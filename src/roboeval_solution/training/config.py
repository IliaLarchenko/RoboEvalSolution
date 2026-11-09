"""Training configuration (RoboEval): the single live config `pi05_roboeval`.

A fresh run starts from the openpi pi0.5 base weights (weight_loader) after `scripts/prepare_assets.py --config-name
pi05_roboeval` wrote the norm stats + FAST tokenizer; the RL flywheel (scripts/rl_loop.py --role trainer) derives
its per-run config from this one.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import optax
import tyro

import openpi.models.model as _model
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
from openpi.training.config import AssetsConfig
import openpi.transforms as _transforms

from roboeval_solution.models import pi_modified_config
from roboeval_solution.policies import roboeval_policy
from roboeval_solution.shared import normalize as _normalize
from roboeval_solution.training import weight_loaders
from roboeval_solution.training import param_groups as _param_groups
from roboeval_solution import transforms as rbe_transforms
from roboeval_solution.data.dataset import DEFAULT_CAMERA_MAP, RoboEvalDataset, SamplingConfig, StoreSource


_AUX_HEAD_NAMES: tuple[str, ...] = (
    "success_head",
    "task_head",
    "completion_head",
    "stage_head",
    "togo_head",
    "wm_fast_success_head",
    "wm_fast_completion_head",
    "wm_fast_togo_head",
    "wm_flow_completion_head",
    "wm_flow_vector_head",
)


def aux_head_weight_decay_mask(params: Any) -> Any:
    """Boolean PyTree matching ``params``: True on the kernel / bias leaves of the aux heads (_AUX_HEAD_NAMES), the
    ``weight_decay_mask`` of AdamWWithAuxDecay."""
    import jax

    def _path_parts(path):
        parts = []
        for p in path:
            key = getattr(p, "key", None)
            if key is None:
                key = getattr(p, "name", None)
            if key is None:
                key = getattr(p, "idx", None)
            parts.append(str(key) if key is not None else str(p))
        return parts

    def classify(path, _leaf):
        parts = _path_parts(path)
        return any(name in parts for name in _AUX_HEAD_NAMES)

    return jax.tree_util.tree_map_with_path(classify, params)


@dataclasses.dataclass(frozen=True)
class AdamWWithAuxDecay(_optimizer.OptimizerConfig):
    """AdamW (openpi's baseline weight decay 1e-10, i.e. none) with an extra decoupled weight decay on the aux-head
    params only (``weight_decay_mask`` = aux_head_weight_decay_mask)."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10          # openpi default
    aux_head_weight_decay: float = 0.001  # extra decay on the aux-head params
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr,
        weight_decay_mask,
    ) -> optax.GradientTransformation:
        chain: list[optax.GradientTransformation] = [
            optax.clip_by_global_norm(self.clip_gradient_norm),
            optax.scale_by_adam(b1=self.b1, b2=self.b2, eps=self.eps),
            optax.add_decayed_weights(self.weight_decay),
        ]
        chain.append(
            optax.add_decayed_weights(
                self.aux_head_weight_decay, mask=weight_decay_mask,
            )
        )
        chain.append(optax.scale_by_learning_rate(lr))
        return optax.chain(*chain)


@dataclasses.dataclass(frozen=True)
class CosineDecayWithCooldown(_optimizer.CosineDecaySchedule):
    """openpi's warmup + cosine decay, optionally with a cooldown: from step `cooldown_start` a second cosine from the
    schedule's LR at that step, lr_s, down to `cooldown_lr` over `cooldown_steps`, then constant:
        lr(t) = cooldown_lr + 0.5 (lr_s - cooldown_lr) (1 + cos(pi min(1, (t - cooldown_start) / cooldown_steps)))
    (None = off: the plain cosine schedule). rl_loop.py --cooldown-start / --cooldown-steps / --cooldown-lr."""

    cooldown_start: int | None = None
    cooldown_steps: int = 2500
    cooldown_lr: float = 3e-6

    def create(self) -> optax.Schedule:
        base = super().create()
        if self.cooldown_start is None:
            return base
        import jax.numpy as jnp

        start, lo = int(self.cooldown_start), float(self.cooldown_lr)
        lr_s = float(base(start))

        def schedule(step):
            frac = jnp.clip((step - start) / max(int(self.cooldown_steps), 1), 0.0, 1.0)
            return jnp.where(step < start, base(step), lo + 0.5 * (lr_s - lo) * (1.0 + jnp.cos(jnp.pi * frac)))

        return schedule


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """The data pipeline a RoboEvalDataConfig resolves to (create()): transforms, norm stats and the dataset factory."""

    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Norm stats (state: quantile map to [-1, 1]; actions: global z-score; transforms_normalize.py).
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Raw dataset keys -> the policy input keys
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Robot-specific transforms (roboeval_policy.RoboEvalInputs / Outputs).
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model-specific transforms, applied after normalisation.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # RoboEval store-backed data (the RoboEvalDataConfig); the loader builds its RoboEvalDataset.
    roboeval: Any = None


@dataclasses.dataclass(frozen=True)
class RoboEvalDataConfig:
    """RoboEval data from rbe stores (released demos converted + rollouts + scripted generations).

    The dataset (``roboeval_solution.data.dataset.RoboEvalDataset``) emits absolute model-representation action
    chunks (data/action_repr.py): no DeltaActions transform.
    """

    # Dataset name (the norm stats directory name when assets.asset_id is unset).
    repo_id: str = "roboeval"
    # Where the norm stats + FAST tokenizer live: <assets_dir or TrainConfig.assets_dirs>/<asset_id>.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    sources: tuple[StoreSource, ...] = ()
    camera_map: tuple[tuple[str, str], ...] = DEFAULT_CAMERA_MAP
    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
    advantage_weighting_clipping: tuple[float, float] = (-2.0, 2.0)
    # Emit the value-head targets (create() sets it).
    use_value_targets: bool = False

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str) -> dict[str, _transforms.NormStats] | None:
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        image_keys = tuple(k for k, _ in self.camera_map)
        if set(image_keys) != set(model_config.image_keys):
            raise ValueError(f"camera_map keys {image_keys} != model image_keys {model_config.image_keys}")
        from roboeval_solution.data.value_targets import VALUE_TARGET_KEYS

        repack_mapping = {
            **{f"observation/{k}": f"observation.images.{k}" for k in image_keys},
            "observation/state": "observation.state",
            "actions": "action",
            "action_is_pad": "action_is_pad",
            "task_id": "task_id",
            "is_demo": "is_demo",
            "is_scripted": "is_scripted",
            "episode_step": "episode_step",
            **{k: k for k in VALUE_TARGET_KEYS},
            "advantage": "advantage",
        }
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_mapping)])
        data_transforms = _transforms.Group(
            inputs=[roboeval_policy.RoboEvalInputs(image_keys=image_keys)],
            outputs=[roboeval_policy.RoboEvalOutputs()],
        )
        h, w = model_config.image_resolution
        model_transforms = _transforms.Group(inputs=[rbe_transforms.ResizeImagesIfNeeded(h, w)])
        wc = self.advantage_weighting_clipping
        model_transforms = model_transforms.push(
            inputs=[rbe_transforms.AdvantageToWeight(clip_min=wc[0], clip_max=wc[1])])

        asset_id = self.assets.asset_id or self.repo_id
        assets_dir = epath.Path(self.assets.assets_dir or assets_dirs)
        norm_stats = self._load_norm_stats(assets_dir, asset_id)
        # FAST targets (training only; train.py refuses to train without them, the policy server drops the transform)
        tokenizer_path = assets_dir / asset_id / "fast_tokenizer"
        if tokenizer_path.exists():
            model_transforms = model_transforms.push(
                inputs=[rbe_transforms.TokenizeFASTActions(
                    tokenizer_path=str(tokenizer_path),
                    encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                    max_fast_tokens=model_config.max_fast_tokens,
                    norm_stats=norm_stats,
                )])
        else:
            logging.warning(f"FAST tokenizer not found at {tokenizer_path} (scripts/prepare_assets.py)")
        return DataConfig(
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            roboeval=dataclasses.replace(self, use_value_targets=True),
        )

    def make_dataset(self, action_horizon: int) -> RoboEvalDataset:
        return RoboEvalDataset(
            use_value_targets=self.use_value_targets,
            sources=self.sources,
            action_horizon=action_horizon,
            camera_map=self.camera_map,
            sampling=self.sampling,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "RoboEval"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Model config (PiModified).
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_modified_config.PiModifiedConfig)

    # Initial weights of a fresh run (not used on resume).
    weight_loader: weight_loaders.WeightLoader = tyro.MISSING

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=CosineDecayWithCooldown)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=AdamWWithAuxDecay)
    ema_decay: float = 0.99

    # Determines the data to be trained on.
    data: RoboEvalDataConfig = dataclasses.field(default_factory=RoboEvalDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int | None = None
    # Global batch size.
    batch_size: int = 512
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 100_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # FSDP configuration for model sharding across devices.
    fsdp_devices: int = 1

    # Number of flow matching samples per training step
    num_flow_samples: int = 1

    # Per-group LR multipliers (training/param_groups.py), applied after AdamW.
    lr_multipliers: Any = dataclasses.field(default_factory=lambda: _param_groups.LRMultipliers())

    # Push checkpoints to the HF Storage Bucket (distributed/bucket.py, RBE_BUCKET) from `train.py`.
    # Every save: bf16 policy export -> policies/<exp_name>/step_S + LATEST (what collectors pull), then
    # older local exports are deleted (the newest stays); every `bucket_full_checkpoint_every` steps also
    # the full step dir (train state, resumable) -> checkpoints/<name>/<exp_name>/<step>.
    # Runs in a background subprocess (CPU only); skipped if busy.
    bucket_upload: bool = False
    bucket_full_checkpoint_every: int | None = 5000
    # RL flywheel: refresh the rollout pool in-process (training/flywheel.py DataRefresher): new collects ->
    # advantages recomputed + a new data loader built in the background and swapped in between steps, so the
    # trainer never restarts for new data. Set by rl_loop.py --role trainer; None = static data.
    data_refresh: Any = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Every nnx.Param except the 1-row Gemma vocab dummy (PiModified replaces the unused 257K table)."""
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*embedder/input_embedding.*")))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Project root (default: the repository's parent directory): data / outputs / runs live next to the code.
PROJECT_ROOT = os.environ.get("RBE_ROOT", str(pathlib.Path(__file__).resolve().parents[4]))
# Data root for RoboEval stores (override with RBE_DATA_ROOT).
DATA_ROOT = os.environ.get("RBE_DATA_ROOT", f"{PROJECT_ROOT}/data")


_CONFIGS = [
    # ── RoboEval (CoRL 2026 challenge) ──────────────────────────────────────
    # BC on the released demos (all 30 variants, PR upweighted) with the advantage token, value heads and
    # Q heads; the RL flywheel (scripts/rl_loop.py) replaces the sources with demos + collects + recovery
    # runs + scripted generations. Model recipe: PiModifiedConfig defaults (models/pi_modified_config.py).
    TrainConfig(
        name="pi05_roboeval",
        project_name="RoboEval",
        exp_name="bc_v1",
        assets_base_dir=f"{PROJECT_ROOT}/outputs/assets",
        checkpoint_base_dir=f"{PROJECT_ROOT}/outputs/checkpoints",
        model=pi_modified_config.PiModifiedConfig(),
        data=RoboEvalDataConfig(
            repo_id="roboeval",
            # Norm stats + FAST tokenizer: cumulative actions, 30-D state with velocities (scripts/prepare_assets.py).
            assets=AssetsConfig(asset_id="roboeval"),
            # Demos at advantage 1.0 (positives) when trained on alone; the RL loop sets every demo frame to
            # --demo-advantage (default -1).
            sources=(StoreSource(paths=(f"{DATA_ROOT}/stores/demos",), share=1.0, default_advantage=1.0,
                                 name="demos"),),
        ),
        weight_loader=weight_loaders.PiModifiedWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # LR, AdamW betas and EMA follow our BEHAVIOR-1K solution (a low peak LR and betas (0.95, 0.99): the gradients
        # are mostly noise). Clip 7.0: the action loss keeps its per-timestep-normalisation scale (the loss weights are
        # not mean-normalised). Larger batches scale the LR by sqrt(batch / 96) (rl_loop.py --batch-size /
        # --peak-lr / --decay-lr).
        lr_schedule=CosineDecayWithCooldown(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=100_000, decay_lr=2.5e-6),
        optimizer=AdamWWithAuxDecay(b1=0.95, b2=0.99, clip_gradient_norm=7.0),
        ema_decay=0.995,
        lr_multipliers=_param_groups.LRMultipliers(expert=2.0, action_proj=2.0, kv_transform=2.0, fast=2.0,
                                                   embeddings=1.0, value_heads=2.0, pe=2.0),
        num_flow_samples=10,
        # Fits one H200-class GPU (no FSDP, no gradient accumulation).
        batch_size=96,
        num_train_steps=100_000,
        num_workers=12,
        log_interval=10,
        save_interval=500,
        # Locally only the latest checkpoint and the newest policy export are kept; every save goes to the bucket.
        keep_period=None,
        wandb_enabled=True,
        bucket_upload=True,
        bucket_full_checkpoint_every=5000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
