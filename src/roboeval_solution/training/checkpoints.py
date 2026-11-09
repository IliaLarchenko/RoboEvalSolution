"""Checkpoint save / resume (orbax): `train_state` (params, optimizer, step), `params` (the EMA weights, what exports
and the policy server load) and `assets` (norm stats, FAST tokenizer).

Adapted from openpi.training.checkpoints, which cannot be imported here: it imports openpi.training.data_loader
-> lerobot.common (the pre-0.4 lerobot API, not installed). CallbackHandler / CallbackSave and load_norm_stats are
copies; _split_params / _merge_params keep only the EMA case.

Reference: https://github.com/Physical-Intelligence/openpi
"""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import shutil
from typing import Protocol

from etils import epath
import jax

import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.training.utils as training_utils

from roboeval_solution.training import data_loader as _data_loader

# our NormStats (per-timestep std + correlation Cholesky)
from roboeval_solution.shared import normalize as _normalize


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # A resume of a run that never saved its first checkpoint starts fresh (a restore would fail).
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

        # The FAST tokenizer: <root>/outputs/checkpoints/<config>/<exp> -> <root>/outputs/assets/<config>/<asset_id>
        checkpoint_dir = checkpoint_manager.directory
        parts = checkpoint_dir.parts
        if 'checkpoints' in parts:
            idx = parts.index('checkpoints')
            resolved_base = epath.Path(*(parts[:idx] + ('assets',) + (parts[idx + 1],)))
        else:
            resolved_base = checkpoint_dir.parent.parent / 'assets' / checkpoint_dir.parent.name
        fast_tokenizer_source = resolved_base / data_config.asset_id / "fast_tokenizer"
        fast_tokenizer_dest = directory / data_config.asset_id / "fast_tokenizer"
        if fast_tokenizer_source.exists():
            fast_tokenizer_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fast_tokenizer_source, fast_tokenizer_dest, dirs_exist_ok=True)
            logging.info(f"Saved FAST tokenizer to checkpoint: {fast_tokenizer_dest}")
        else:
            logging.warning(f"FAST tokenizer source not found: {fast_tokenizer_source}")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def _check_params_tree(params_dir, live_params) -> None:
    """Raise unless the checkpoint's params item (read from its metadata, no arrays loaded) has exactly the live param
    paths and shapes: a checkpoint of another model tree cannot be resumed (start a new run with
    weight_loaders.PiModifiedWeightLoader instead)."""
    import flax
    import numpy as np

    with ocp.PyTreeCheckpointer() as ckptr:
        meta = ckptr.metadata(str(params_dir)).item_metadata
    if meta is None:
        raise FileNotFoundError(f"no checkpoint params metadata in {params_dir}")
    disk = flax.traverse_util.flatten_dict(meta.tree["params"])
    disk = {"/".join(map(str, k[:-1] if k[-1] == "value" else k)): v for k, v in disk.items()}
    live = {"/".join(map(str, k)): v for k, v in flax.traverse_util.flatten_dict(live_params.to_pure_dict()).items()}
    missing = sorted(set(live) - set(disk))
    extra = sorted(set(disk) - set(live))
    shape = sorted(k for k in set(live) & set(disk)
                   if getattr(disk[k], "shape", None) is not None and tuple(disk[k].shape) != tuple(np.shape(live[k])))
    if missing or extra or shape:
        raise ValueError(f"checkpoint {params_dir} does not match the model's param tree: {len(missing)} live params "
                         f"missing (e.g. {missing[:3]}), {len(extra)} extra on disk (e.g. {extra[:3]}), {len(shape)} "
                         f"with another shape (e.g. {shape[:3]})")


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
) -> training_utils.TrainState:
    """Resume: the full train state (params, optimizer, step) and the EMA params of the latest checkpoint, which must
    have exactly the live param tree."""
    with at.disable_typechecking():
        train_state, params = _split_params(state)
        ckpt_step = checkpoint_manager.latest_step()
        ckpt_dir = checkpoint_manager.directory / str(ckpt_step)
        if not (ckpt_dir / "train_state").exists():
            raise FileNotFoundError(f"{ckpt_dir} has no train_state (an export?): it cannot be resumed; start a new "
                                    "run from its params with weight_loaders.PiModifiedWeightLoader")
        _check_params_tree(ckpt_dir / "params", params)
        restored = checkpoint_manager.restore(
            ckpt_step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
        return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    params = state.ema_params
    train_state = dataclasses.replace(state, ema_params=None)
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Inverse of _split_params: the saved params are the EMA params.
    return dataclasses.replace(train_state, ema_params=params["params"])
