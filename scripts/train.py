"""Train the RoboEval policy + value heads, adapted from openpi's scripts/train.py
(https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train.py).

    .venv/bin/python scripts/train.py pi05_roboeval --exp_name <name>
    .venv/bin/python scripts/train.py --config_file <config.pkl>    # a pickled TrainConfig (rl_loop.py)
"""

import concurrent.futures
import dataclasses
import functools
import logging
import os
import platform
import shutil
import sys
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

# Up to 95% of GPU memory; the 'platform' allocator (CUDA malloc) does not preallocate.
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.95')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

# Cap BLAS threads (avoids OpenBLAS thread-creation errors).
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from roboeval_solution.training import checkpoints as _checkpoints
from roboeval_solution.training import config as _config
from roboeval_solution.training import data_loader as _data_loader
from roboeval_solution.training import weight_loaders as _weight_loaders
from roboeval_solution.training import attention_backend as _attention_backend

from roboeval_solution.models.pi_modified import PiModified
from roboeval_solution.models.observation import Observation
from roboeval_solution.training.loss_weights import type_balanced_weighted_mean
from roboeval_solution import transforms as _rbe_transforms

# Consecutive steps with a non-finite loss / gradient (each skipped: params, optimizer and EMA unchanged) before
# training aborts, without another checkpoint save.
MAX_CONSECUTIVE_NONFINITE = 10


_bucket_upload_executor: concurrent.futures.ThreadPoolExecutor | None = None
_bucket_upload_future: concurrent.futures.Future | None = None


def _maybe_submit_bucket_upload(config, step: int, checkpoint_manager) -> None:
    """Fire-and-forget push of a saved step to the bucket: its policy export, plus the full checkpoint every
    `bucket_full_checkpoint_every` steps. Skipped while the previous push is still running.

    The export runs as a `bucket_sync.py export-policy` subprocess on the CPU: this process already owns the GPU
    through JAX.
    """
    global _bucket_upload_executor, _bucket_upload_future
    if _bucket_upload_executor is None:
        _bucket_upload_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bucket_upload",
        )
    if _bucket_upload_future is not None and not _bucket_upload_future.done():
        logging.info("Bucket upload still running, skipping step %d", step)
        return

    import subprocess
    from pathlib import Path

    from roboeval_solution.distributed import bucket

    repo = Path(__file__).resolve().parents[1]

    def _run() -> None:
        try:
            checkpoint_manager.wait_until_finished()
            step_dir = Path(str(config.checkpoint_dir)) / str(step)
            if not step_dir.exists():
                logging.warning("Bucket upload: %s was pruned before upload, skipping", step_dir)
                return
            subprocess.run(
                [sys.executable, str(repo / "scripts" / "bucket_sync.py"), "export-policy", str(step_dir),
                 "--run", config.exp_name],
                check=True, env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
                stdin=subprocess.DEVNULL,
            )
            logging.info("Bucket upload: policy step %d -> policies/%s", step, config.exp_name)
            # Pushed exports live in the bucket: keep only the newest locally.
            exports = repo.parent / "runs" / config.exp_name / "policies"
            for old in sorted(p for p in exports.glob("step_*") if p.is_dir() and p.name != f"step_{step:06d}"):
                shutil.rmtree(old, ignore_errors=True)
            full_every = config.bucket_full_checkpoint_every
            if full_every and step % full_every == 0:
                bucket.push_dir(step_dir, f"checkpoints/{config.name}/{config.exp_name}/{step}")
                logging.info("Bucket upload: full checkpoint step %d -> checkpoints/%s/%s",
                             step, config.name, config.exp_name)
        except Exception as e:  # noqa: BLE001 — never crash the training loop
            logging.warning("Bucket upload failed at step %d: %r", step, e)

    _bucket_upload_future = _bucket_upload_executor.submit(_run)


def _drain_bucket_uploads() -> None:
    if _bucket_upload_future is not None:
        try:
            _bucket_upload_future.result()
        except Exception as e:  # noqa: BLE001
            logging.warning("Bucket upload (final) failed: %r", e)


def init_logging():
    """Compact log format (one-letter levels)."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    # wandb's shared mode (primary writer), so other processes can log into the same run.
    shared = wandb.Settings(mode="shared", x_primary=True, x_label="trainer")
    if resuming and (ckpt_dir / "wandb_id.txt").exists():
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, settings=shared)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            settings=shared,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """The weights `loader` provides for a fresh run: a subset of the param tree (the rest keeps the model's init),
    checked against the model's paths, shapes and dtypes."""
    flat_ref = traverse_util.flatten_dict(params_shape)
    flat = traverse_util.flatten_dict(loader.load(params_shape))
    unknown = sorted("/".join(map(str, k)) for k in flat if k not in flat_ref)
    bad = sorted("/".join(map(str, k)) for k, v in flat.items()
                 if k in flat_ref and (tuple(v.shape) != tuple(flat_ref[k].shape) or v.dtype != flat_ref[k].dtype))
    if unknown or bad:
        raise ValueError(f"weight loader returned params the model does not have: {unknown[:5]}, or with another "
                         f"shape / dtype: {bad[:5]}")
    logging.info("Weight loader: %d of %d params loaded, the rest keep their init", len(flat), len(flat_ref))
    return traverse_util.unflatten_dict(flat)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    # AdamWWithAuxDecay: extra weight decay on the aux-head kernels + biases only.
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule,
        weight_decay_mask=_config.aux_head_weight_decay_mask,
    )
    # Per-group LR multipliers (stateless scaling after AdamW; optimizer state unchanged).
    from roboeval_solution.training import param_groups
    tx = param_groups.with_lr_multipliers(tx, config.lr_multipliers)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# Metrics logged to wandb under train/.
_TRAIN_LOSS_KEYS = {
    "loss", "action_loss", "fast_loss", "success_loss", "action_loss_demo", "action_loss_rollout",
    "action_loss_scripted", "completion_loss", "stage_loss",
    # Togo + Q heads. wm_flow_vector_loss is the WM-flow Q loss; wm_flow_{success,stage,togo}_loss are its blocks.
    "togo_loss",
    "wm_fast_success_loss", "wm_fast_completion_loss", "wm_fast_togo_loss",
    "wm_flow_vector_loss", "wm_flow_success_loss", "wm_flow_stage_loss", "wm_flow_togo_loss",
    "wm_flow_completion_loss",
}
_TRAIN_METRIC_KEYS = {
    "fast_perplexity", "success_pred_mean", "success_pred_std", "demo_frac", "scripted_frac",
    "success_target_mean", "success_mae", "success_valid_frac",
    "success_pred_mean_weighted", "success_target_mean_weighted",
    "completion_pred_mean",
    "completion_pred_mean_weighted", "completion_target_mean_weighted",
    "stage_pred_mean", "stage_accuracy", "stage_valid_frac", "completion_valid_frac",
    "grad_norm", "param_norm", "nonfinite_step", "fast_accuracy",
    # togo_valid_frac: one metric for all three togo heads. wm_flow_t_valid_frac: share of the flow samples with
    # t <= 0.5 (the ones the WM-flow head trains on).
    "togo_mae", "togo_valid_frac", "wm_flow_t_valid_frac",
    # wm_flow_success: the Δsuccess component (true − V̂), range ~[-1, 1].
    "wm_flow_success_pred_mean", "wm_flow_success_pred_std",
    "wm_flow_success_target_mean",
    "action_pad_frac",
    # action flow loss in per-timestep-normalised units, overall and per chunk-step range
    "action_loss_pts", "action_loss_pts_h0", "action_loss_pts_h1_4", "action_loss_pts_h5_14", "action_loss_pts_h15_29",
}


def _structure_metrics(raw_info: dict) -> dict:
    """Convert flat metrics dict into structured wandb sections.

    Returns dict with train/loss_name and train/metric_name keys.
    Drops per-joint losses and other noisy metrics.
    """
    structured = {}
    for key, val in raw_info.items():
        if key in _TRAIN_LOSS_KEYS:
            structured[f"train/{key}"] = val
        elif key in _TRAIN_METRIC_KEYS:
            structured[f"train/{key}"] = val
    return structured


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # Snapshot of the params for the non-finite guard below: the merged model shares its Variables with state.params,
    # so nnx.update(model, ...) overwrites what state.params holds.
    old_params = jax.tree.map(lambda x: x, state.params)
    model = nnx.merge(state.model_def, state.params)
    model.train()

    observation, actions = batch

    @at.typecheck
    def loss_fn(
        model: PiModified, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions
    ):
        losses_dict = model.compute_detailed_loss(rng, observation, actions, num_flow_samples=config.num_flow_samples)
        per_sample_loss = losses_dict["total_loss"]  # [B]
        B = per_sample_loss.shape[0]
        ones_b = jnp.ones_like(per_sample_loss)
        # flow loss on demo vs rollout frames (diagnostic; the mix itself is set by the data loader)
        demo = observation.is_demo.astype(jnp.float32)
        for name, m in (("action_loss_demo", demo), ("action_loss_rollout", 1.0 - demo)):
            losses_dict[name] = (jnp.sum(losses_dict["action_loss"] * m) / jnp.maximum(jnp.sum(m), 1.0)) * ones_b
        losses_dict["demo_frac"] = jnp.mean(demo) * ones_b
        scr = observation.is_scripted.astype(jnp.float32)
        losses_dict["action_loss_scripted"] = (jnp.sum(losses_dict["action_loss"] * scr)
                                               / jnp.maximum(jnp.sum(scr), 1.0)) * ones_b
        losses_dict["scripted_frac"] = jnp.mean(scr) * ones_b
        # Action + FAST losses: plain batch mean; the data loader's sampling sets the mix.
        total_loss = jnp.mean(per_sample_loss)

        # Auxiliary (value) losses: IS-weighted (is_weight = 1 / (N p_i), clipped; data/dataset.py) and task-balanced.
        gt_id = observation.task_id
        is_w = observation.is_weight

        def _aux(loss_key, valid, weights, scale):
            return scale * type_balanced_weighted_mean(losses_dict[loss_key], weights, valid, gt_id)

        def _record_weighted(prefix, pred_key, target, valid, weights):
            """Type-balanced means of predictions and targets under the loss's weights (vs. the unweighted
            `{prefix}_pred_mean`)."""
            pred = losses_dict[pred_key]
            losses_dict[f"{prefix}_pred_mean_weighted"] = (
                type_balanced_weighted_mean(pred, weights, valid, gt_id) * ones_b
            )
            losses_dict[f"{prefix}_target_mean_weighted"] = (
                type_balanced_weighted_mean(target, weights, valid, gt_id) * ones_b
            )

        # success head: BCE on P(success), frames with a finite target.
        valid = jnp.isfinite(observation.success_target)
        w = is_w
        total_loss = total_loss + _aux(
            "success_loss", valid, w, config.model.success_loss_weight,
        )
        _record_weighted("success", "_success_pred_per_sample", observation.success_target, valid, w)

        # completion head: successful rollouts + successful demos, plus failed rollouts at target 0; the dataset emits
        # NaN for every frame without a target, so validity is just finiteness. completion_weight_scale = 0.5 on demo
        # frames.
        valid = jnp.isfinite(observation.completion_target)
        w = is_w * observation.completion_weight_scale
        total_loss = total_loss + _aux(
            "completion_loss", valid, w, config.model.completion_loss_weight,
        )
        _record_weighted("completion", "_completion_pred_per_sample", observation.completion_target, valid, w)
        losses_dict["completion_valid_frac"] = valid.astype(jnp.float32).mean() * ones_b

        # stage head: masked BCE on the per-frame stage targets (rollouts with a recording).
        valid = observation.stage_mask.any(axis=-1)
        total_loss = total_loss + _aux(
            "stage_loss", valid, is_w, config.model.stage_loss_weight,
        )
        losses_dict["stage_valid_frac"] = valid.astype(jnp.float32).mean() * ones_b

        # Togo head (Head 1): per-sample loss already masked + averaged over valid slots in compute_detailed_loss;
        # a sample is valid when any slot is.
        valid = observation.togo_mask.any(axis=-1)
        total_loss = total_loss + _aux(
            "togo_loss", valid, is_w, config.model.togo_loss_weight,
        )

        # WM-FAST head (Head 2, training only): future targets, same validity as the current-frame heads.
        valid = jnp.isfinite(observation.success_future_target)
        w = is_w
        total_loss = total_loss + _aux(
            "wm_fast_success_loss", valid, w, config.model.wm_fast_success_loss_weight,
        )
        valid = jnp.isfinite(observation.completion_future_target)
        total_loss = total_loss + _aux(
            "wm_fast_completion_loss", valid, is_w, config.model.wm_fast_completion_loss_weight,
        )
        valid = observation.togo_future_mask.any(axis=-1)
        total_loss = total_loss + _aux(
            "wm_fast_togo_loss", valid, is_w, config.model.wm_fast_togo_loss_weight,
        )

        # WM-flow head (Head 3): the model zeroes the loss of flow samples with t > 0.5 (mostly noise); the same t-mask
        # gates the weighted-mean denominator.
        wm_flow_t_valid = losses_dict["_wm_flow_t_valid_per_sample"]
        # Value-vector residual (17 components): per-sample masked mean, valid when any component is.
        wm_vec_valid = observation.wm_vector_mask.any(axis=-1)
        valid = wm_vec_valid & wm_flow_t_valid
        total_loss = total_loss + _aux(
            "wm_flow_vector_loss", valid, is_w, config.model.wm_flow_vector_loss_weight,
        )
        valid_s = observation.wm_vector_mask[:, 0] & wm_flow_t_valid
        total_loss = total_loss + _aux(
            "wm_flow_success_loss", valid_s, is_w, config.model.wm_flow_success_loss_weight,
        )
        valid = jnp.isfinite(observation.completion_future_target) & wm_flow_t_valid
        total_loss = total_loss + _aux(
            "wm_flow_completion_loss", valid, is_w, config.model.wm_flow_completion_loss_weight,
        )

        # Logging: the per-sample losses are 0 on invalid samples; the logged values are means over the valid ones.
        def _valid_only_mean_replace(loss_key, valid_bool):
            per_sample = losses_dict[loss_key]
            v = valid_bool.astype(per_sample.dtype)
            v_sum = v.sum()
            scalar = jnp.where(
                v_sum > 0,
                (per_sample * v).sum() / jnp.maximum(v_sum, 1.0),
                0.0,
            )
            # [B]: the info loop's jnp.mean leaves it unchanged
            losses_dict[loss_key] = scalar * jnp.ones_like(per_sample)

        # Head 1: current frame
        kpt_valid_now = observation.togo_mask.any(axis=-1)
        # Heads 2 / 3: future
        kpt_valid_fut = observation.togo_future_mask.any(axis=-1)
        # Future success / completion: valid like their current-frame counterparts (finite targets)
        s_fut_valid = jnp.isfinite(observation.success_future_target)
        c_fut_valid = jnp.isfinite(observation.completion_future_target)
        stage_valid = observation.stage_mask.any(axis=-1)

        def _block_valid(lo, hi):
            return observation.wm_vector_mask[:, lo:hi].any(axis=-1)
        n_stage = config.model.num_stage_outputs

        _valid_only_mean_replace("togo_loss", kpt_valid_now)
        _valid_only_mean_replace("togo_mae_per_sample", kpt_valid_now)
        # WM-FAST heads read the post-FAST token, which training always has: no t-mask
        _valid_only_mean_replace("wm_fast_success_loss", s_fut_valid)
        _valid_only_mean_replace("wm_fast_completion_loss", c_fut_valid)
        _valid_only_mean_replace("wm_fast_togo_loss", kpt_valid_fut)
        # WM-flow heads: also the t ≤ 0.5 mask, as in the loss
        def _with_t(vb):
            return vb & wm_flow_t_valid
        _valid_only_mean_replace("wm_flow_vector_loss", _with_t(wm_vec_valid))
        _valid_only_mean_replace("wm_flow_success_loss", _with_t(_block_valid(0, 1)))
        _valid_only_mean_replace("wm_flow_stage_loss", _with_t(_block_valid(1, 1 + n_stage)))
        _valid_only_mean_replace("wm_flow_togo_loss", _with_t(_block_valid(1 + n_stage, None)))
        _valid_only_mean_replace("wm_flow_completion_loss", _with_t(c_fut_valid))
        _valid_only_mean_replace("stage_loss", stage_valid)
        # success / completion: the per-sample losses are 0 on frames without a target (scripted, recovery, demos):
        # log the mean over the frames that have one.
        _valid_only_mean_replace("success_loss", jnp.isfinite(observation.success_target))
        _valid_only_mean_replace("success_mae", jnp.isfinite(observation.success_target))
        _valid_only_mean_replace("completion_loss", jnp.isfinite(observation.completion_target))
        losses_dict["stage_accuracy"] = losses_dict["stage_accuracy_per_sample"]
        _valid_only_mean_replace("stage_accuracy", stage_valid)

        losses_dict["togo_mae"] = losses_dict["togo_mae_per_sample"]

        # Batch share with togo supervision (one metric for all three togo heads)
        losses_dict["togo_valid_frac"] = (
            kpt_valid_now.astype(jnp.float32).mean() * jnp.ones(B)
        )
        # Share of all num_flow_samples x B flow samples with t ≤ 0.5: ≈ 0.354 = P(t ≤ 0.5) under Beta(1.5, 1)
        losses_dict["wm_flow_t_valid_frac"] = losses_dict["_wm_flow_t_valid_frac_scalar"]

        return total_loss, losses_dict

    train_rng = jax.random.fold_in(rng, state.step)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, losses_dict), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_ema = jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
    )

    # Non-finite loss or gradient: skip the update (params, optimizer state and EMA stay as they were; the step
    # counter advances), so one bad batch can't overwrite the checkpoint and the exports with NaNs. main() aborts
    # after MAX_CONSECUTIVE_NONFINITE skips in a row.
    grad_norm = optax.global_norm(grads)
    finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)

    def keep_if_bad(new, old):
        return jax.tree.map(lambda n, o: jnp.where(finite, n, o), new, old)

    new_state = dataclasses.replace(state, step=state.step + 1, params=keep_if_bad(new_params, old_params),
                                    opt_state=keep_if_bad(new_opt_state, state.opt_state))
    new_state = dataclasses.replace(new_state, ema_params=keep_if_bad(new_ema, state.ema_params))

    # Compute param norm (kernel weights only)
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "param_norm": optax.global_norm(kernel_params),
        "nonfinite_step": (~finite).astype(jnp.float32),
    }
    _AUX_KEYS = (
        "action_loss", "success_loss", "fast_loss", "success_pred_mean", "success_pred_std",
        "action_loss_demo", "action_loss_rollout", "demo_frac", "action_loss_scripted", "scripted_frac",
        "fast_accuracy", "success_target_mean", "success_mae", "success_valid_frac",
        "success_pred_mean_weighted", "success_target_mean_weighted",
        "completion_loss", "completion_pred_mean",
        "completion_pred_mean_weighted", "completion_target_mean_weighted",
        "stage_loss", "stage_pred_mean", "stage_accuracy", "stage_valid_frac", "completion_valid_frac",
        # Togo + WM heads; their *_loss values are valid-only means (loss_fn).
        "togo_loss", "togo_mae", "togo_valid_frac",
        "wm_fast_success_loss", "wm_fast_completion_loss", "wm_fast_togo_loss",
        "wm_flow_vector_loss", "wm_flow_success_loss", "wm_flow_stage_loss", "wm_flow_togo_loss",
        "wm_flow_completion_loss",
        "wm_flow_t_valid_frac",
        "wm_flow_success_pred_mean", "wm_flow_success_pred_std",
        "wm_flow_success_target_mean",
        # Share of padded action timesteps (chunk past the episode end), masked out of action_loss.
        "action_pad_frac",
        "action_loss_pts", "action_loss_pts_h0", "action_loss_pts_h1_4", "action_loss_pts_h5_14", "action_loss_pts_h15_29",
    )
    for key in _AUX_KEYS:
        if key in losses_dict:
            val = losses_dict[key]
            info[key] = val if (isinstance(val, (float, int)) or (hasattr(val, 'ndim') and val.ndim == 0)) else jnp.mean(val)
    if "fast_loss" in info:
        info["fast_perplexity"] = jnp.exp(info["fast_loss"])

    return new_state, info


def raise_open_file_limit() -> None:
    """Soft RLIMIT_NOFILE up to the hard limit (inherited by the data loader workers): the dataset keeps one memmap,
    an open fd, per store and camera in every worker, far above the usual soft limit of 1024 for a large pool."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        logging.info("RLIMIT_NOFILE %d -> %d", soft, hard)


def _release_freed_memory() -> None:
    """After a data loader swap: collect the old loader and hand the freed heap back to the OS (glibc keeps freed
    small blocks in its arenas until malloc_trim)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    raise_open_file_limit()

    backend = _attention_backend.install()
    logging.info(f"Attention backend: {backend}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for JAX RNG: {seed}")

    rng = jax.random.key(seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
    )

    if not any(isinstance(t, _rbe_transforms.TokenizeFASTActions) for t in data_loader.data_config().model_transforms.inputs):
        raise FileNotFoundError("no FAST tokenizer in the assets (scripts/prepare_assets.py): the FAST loss and the "
                                "WM-FAST head need it")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    refresher = None  # started after the train state exists (needs the start step)

    # Aux target fields of the first batch
    obs_check = batch[0]
    for field in ("success_target", "task_id", "completion_target", "stage_target",
                  "togo_target", "wm_vector_target"):
        val = getattr(obs_check, field)
        logging.info("Observation.%s: shape=%s dtype=%s", field, val.shape, val.dtype)

    # Camera views of the first batch
    obs_for_log = batch[0]
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in obs_for_log.images.values()], axis=1))
        for i in range(min(5, len(next(iter(obs_for_log.images.values())))))
    ]
    wandb.log({"media/camera_views": images_to_log}, step=0)

    data_config = data_loader.data_config()
    if data_config.norm_stats is None:
        raise ValueError("norm_stats not found: run scripts/prepare_assets.py --config-name <config> first.")
    norm_stats = data_config.norm_stats

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)

    # Norm-stat-derived model state (flow-noise Cholesky, loss weights: nnx.Intermediate in the params; the
    # inpainting correction matrices: module level), outside jit.
    model = nnx.merge(train_state.model_def, train_state.params)
    model.load_correlation_matrix(norm_stats)
    logging.info("Loaded correlation matrix")
    train_state = dataclasses.replace(
        train_state, model_def=nnx.graphdef(model), params=nnx.state(model)
    )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    if config.data_refresh is not None:
        from roboeval_solution.training import flywheel
        base_sources = flywheel.base_sources_of(config.data.sources)
        refresher = flywheel.DataRefresher(
            config, base_sources, data_sharding,
            lambda cfg, sh: _data_loader.create_data_loader(cfg, sharding=sh),
            start_step).start()
        logging.info("In-process data refresh on: %s", config.data_refresh)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    _t_gpu_window = 0.0
    _t_data_window = 0.0
    _t_window_steps = 0

    consecutive_nonfinite = 0
    for step in pbar:
        _t0 = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready(info)
        _t1 = time.perf_counter()
        infos.append(info)
        if float(info["nonfinite_step"]) > 0:
            consecutive_nonfinite += 1
            logging.warning("Step %d: non-finite loss %s / grad norm %s: update skipped (%d in a row)", step,
                            float(info["loss"]), float(info["grad_norm"]), consecutive_nonfinite)
            if consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
                raise RuntimeError(f"{consecutive_nonfinite} consecutive non-finite steps (last {step}): aborting "
                                   "before the next checkpoint save")
        else:
            consecutive_nonfinite = 0

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            main_metrics = {k: v for k, v in reduced_info.items()
                          if k in _TRAIN_LOSS_KEYS | _TRAIN_METRIC_KEYS | {"loss"}}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in main_metrics.items())
            logging.info(f"Step {step}: {info_str}")

            wandb.log(_structure_metrics(reduced_info), step=step)
            sys.stderr.flush()
            infos = []

        _t2 = time.perf_counter()
        batch = next(data_iter)
        if refresher is not None:
            refresher.set_step(step + 1)
            fresh = refresher.poll()
            if fresh is not None:  # swap to the refreshed pool between steps (same shapes: no recompile)
                data_loader.close()  # else the old producer thread + workers (dataset copies) leak per swap
                data_loader, data_iter, batch = fresh
                _release_freed_memory()
                logging.info("Data loader swapped at step %d", step + 1)
        _t3 = time.perf_counter()

        _t_gpu_window += (_t1 - _t0)
        _t_data_window += (_t3 - _t2)
        _t_window_steps += 1

        if _t_window_steps == 10:
            logging.info(
                f"TIMING [last 10 steps]: "
                f"gpu={_t_gpu_window/10:.3f}s, "
                f"data={_t_data_window/10:.3f}s, "
                f"total={(_t_gpu_window+_t_data_window)/10:.3f}s/step"
            )
            _t_gpu_window = 0.0
            _t_data_window = 0.0
            _t_window_steps = 0

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            if config.bucket_upload:
                _maybe_submit_bucket_upload(config, step, checkpoint_manager)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    _drain_bucket_uploads()

    if config.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    # --config_file: a TrainConfig pickled by rl_loop.py
    if "--config_file" in sys.argv:
        idx = sys.argv.index("--config_file")
        config_path = sys.argv[idx + 1]
        import pickle
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        main(config)
    else:
        main(_config.cli())
