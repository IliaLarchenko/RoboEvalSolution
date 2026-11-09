"""Observation (model inputs + training targets) and `preprocess_observation` (resize + training augmentation)."""

from collections.abc import Sequence
from typing import Generic, TypeVar
import dataclasses

import augmax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.shared import array_typing as at

from roboeval_solution.models.train_aug_config import TrainAugmentationConfig

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


def _is_top_key(key: str) -> bool:
    """Static camera (head / front / external): no left / right in the key."""
    return ("left" not in key and "right" not in key)


def _squeeze_scalar(x):
    """Squeeze a trailing size-1 dimension ([B, 1] -> [B])."""
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim >= 2 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def _squeeze_scalar_int(x):
    """Squeeze a trailing size-1 dimension and cast to int32."""
    if x is None:
        return None
    return _squeeze_scalar(x).astype(np.int32)


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Model inputs (images, state, task, advantage, elapsed time, FAST tokens) and the training targets of the
    value / Q heads (data/value_targets.py; NaN or a False mask = no target)."""

    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]

    # FAST tokens of the action chunk and their mask (training only: the FAST loss and the WM-FAST head).
    fast_tokens: at.Int[ArrayT, "*b t"] | None = None
    fast_token_mask: at.Bool[ArrayT, "*b t"] | None = None

    # Normalised advantage, clipped to [-2, 2] (advantage-token conditioning; training only).
    advantage: at.Float[ArrayT, "*b"] | None = None

    # Episode success (0 / 1) for the success head.
    success_target: at.Float[ArrayT, "*b"] | None = None

    # Aux-loss importance weight 1 / (N p_i) (clipped), undoing the sampling distribution (data/dataset.py).
    is_weight: at.Float[ArrayT, "*b"] | None = None

    # Task family id (data/tasks.TASK_FAMILIES): the task input token and AdaRMS channel.
    task_id: at.Int[ArrayT, "*b"] | None = None
    # 1.0 for demonstration frames, 0.0 otherwise (training metrics only).
    is_demo: at.Float[ArrayT, "*b"] | None = None
    # 1.0 for scripted frames (scripted demos and the scripted part of recovery episodes): selects the scripted_*
    # dropout rates of TrainAugmentationConfig; also used for loss logging.
    is_scripted: at.Float[ArrayT, "*b"] | None = None
    # Env steps since the episode reset at this observation (20 Hz); read only by the value / Q heads, never by the
    # VLM or the action expert.
    episode_step: at.Float[ArrayT, "*b"] | None = None
    # Completion target: t / T on successes, 0 on failures.
    completion_target: at.Float[ArrayT, "*b"] | None = None

    # Togo head: normalised behaviour-to-go at t (width constants.TOGO_HEAD_WIDTH) and its mask.
    togo_target: at.Float[ArrayT, "*b k"] | None = None
    togo_mask: at.Bool[ArrayT, "*b k"] | None = None

    # WM-FAST targets (data/value_targets.FUTURE_HORIZON = 30 steps ahead): episode success, completion at t + 30,
    # togo spent over the chunk togo[t] - togo[t + 30] (and its mask). The WM-flow completion head shares the
    # completion target.
    success_future_target: at.Float[ArrayT, "*b"] | None = None
    completion_future_target: at.Float[ArrayT, "*b"] | None = None
    togo_future_target: at.Float[ArrayT, "*b k"] | None = None
    togo_future_mask: at.Bool[ArrayT, "*b k"] | None = None

    # Stage head: final stage flags [5] and their mask (j < J_f). Distinct dim names: jaxtyping binds a name across
    # fields.
    stage_target: at.Float[ArrayT, "*b ns"] | None = None
    stage_mask: at.Bool[ArrayT, "*b ns"] | None = None
    # WM-flow target psi(t) = [success, stage x5, togo x11] (the model subtracts sg(psi_hat(s_t))) and its mask.
    wm_vector_target: at.Float[ArrayT, "*b nv"] | None = None
    wm_vector_mask: at.Bool[ArrayT, "*b nv"] | None = None
    # Per-sample completion-loss scale (data/value_targets.DEMO_COMPLETION_SCALE on demos). None behaves as 1.0.
    completion_weight_scale: at.Float[ArrayT, "*b"] | None = None

    # [action_horizon] True where the chunk runs past the episode end (padding, hold pose): masked out of the flow
    # loss.
    action_is_pad: at.Bool[ArrayT, "*b ah"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """Convert dict to Observation."""
        # uint8 images -> float32 [-1, 1], in a new dict so the caller's batch is not mutated.
        images = {}
        for key, img in data["image"].items():
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            images[key] = img

        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            fast_tokens=data.get("fast_tokens"),
            fast_token_mask=data.get("fast_token_mask"),
            advantage=_squeeze_scalar(data.get("advantage")),
            success_target=_squeeze_scalar(data.get("success_target")),
            is_weight=_squeeze_scalar(data.get("is_weight")),
            task_id=_squeeze_scalar_int(data.get("task_id")),
            is_demo=_squeeze_scalar(data.get("is_demo")),
            is_scripted=_squeeze_scalar(data.get("is_scripted")),
            episode_step=_squeeze_scalar(data.get("episode_step")),
            completion_target=_squeeze_scalar(data.get("completion_target")),
            togo_target=data.get("togo_target"),
            togo_mask=data.get("togo_mask"),
            success_future_target=_squeeze_scalar(data.get("success_future_target")),
            completion_future_target=_squeeze_scalar(data.get("completion_future_target")),
            togo_future_target=data.get("togo_future_target"),
            togo_future_mask=data.get("togo_future_mask"),
            stage_target=data.get("stage_target"),
            stage_mask=data.get("stage_mask"),
            wm_vector_target=data.get("wm_vector_target"),
            wm_vector_mask=data.get("wm_vector_mask"),
            completion_weight_scale=_squeeze_scalar(data.get("completion_weight_scale")),
            action_is_pad=data.get("action_is_pad"),
        )


def _build_color_chain(cfg: TrainAugmentationConfig) -> augmax.Chain:
    """Per-camera augmax colour chain."""
    return augmax.Chain(augmax.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast,
                                           saturation=cfg.saturation, hue=0.0))


def _build_top_geo_chain(cfg: TrainAugmentationConfig, h: int, w: int) -> augmax.Chain:
    """Static-camera geometric chain (crop + rotate)."""
    return augmax.Chain(augmax.RandomCrop(int(w * cfg.top_crop_frac), int(h * cfg.top_crop_frac)),
                        augmax.Resize(w, h),
                        augmax.Rotate((-cfg.top_rotate_deg, cfg.top_rotate_deg)))


def _cutout_one(rng, image, prob: float, max_area: float):
    """Single rectangular cutout filled with grey (0.5 in the [0, 1] space the augmentations run in)."""
    rng_apply, rng_area, rng_aspect, rng_cy, rng_cx = jax.random.split(rng, 5)
    H, W, _ = image.shape
    apply = jax.random.bernoulli(rng_apply, prob)
    area_frac = jax.random.uniform(rng_area, (), minval=0.0, maxval=max_area)
    aspect = jax.random.uniform(rng_aspect, (), minval=0.3, maxval=3.0)
    area_px = area_frac * H * W
    h = jnp.sqrt(area_px * aspect)
    w = jnp.sqrt(area_px / aspect)
    cy = jax.random.uniform(rng_cy, (), minval=0.0, maxval=float(H))
    cx = jax.random.uniform(rng_cx, (), minval=0.0, maxval=float(W))
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    x0, x1 = cx - w / 2.0, cx + w / 2.0
    ys = jnp.arange(H)[:, None].astype(jnp.float32)
    xs = jnp.arange(W)[None, :].astype(jnp.float32)
    inside = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)  # (H, W)
    mask = inside[..., None] & apply
    return jnp.where(mask, jnp.array(0.5, image.dtype), image)


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool,
    image_keys: Sequence[str],
    image_resolution: tuple[int, int],
    train_aug: TrainAugmentationConfig,
) -> Observation:
    """With `train`, the augmentations of `train_aug` (images already at `image_resolution`): static-camera
    crop / rotate, colour jitter (shared by the cameras), cutout, then camera dropout (zero image, mask False).
    Every non-image field passes through unchanged. Randomness is drawn outside the vmaps (no lax.cond inside)."""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    num_cams = len(image_keys)
    if train:
        # Fixed RNG stream slots (the others are spare): 0 colour, 5 static-camera geometry, 10 cutout,
        # 11 camera dropout
        rngs = jax.random.split(rng, 16)
        rng_color, rng_top_geo, rng_cutout, rng_camdrop = rngs[0], rngs[5], rngs[10], rngs[11]
        batch_size = observation.state.shape[0]
        color_chain = _build_color_chain(train_aug)
        top_geo_chain = _build_top_geo_chain(train_aug, *image_resolution)
        cutout_root_keys = jax.random.split(rng_cutout, num_cams)
        camdrop_root_keys = jax.random.split(rng_camdrop, num_cams)

    out_images = {}
    for cam_idx, key in enumerate(image_keys):
        image = observation.images[key]

        if train:
            image = image / 2.0 + 0.5  # [-1, 1] -> [0, 1] for the augmentations
            # static cameras share one geometric draw per sample
            if _is_top_key(key):
                image = jax.vmap(top_geo_chain)(jax.random.split(rng_top_geo, batch_size), image)
            # colour: the same draw per sample on every camera
            image = jax.vmap(color_chain)(jax.random.split(rng_color, batch_size), image)
            image = jax.vmap(_cutout_one, in_axes=(0, 0, None, None))(
                jax.random.split(cutout_root_keys[cam_idx], batch_size), image,
                train_aug.cutout_prob, train_aug.cutout_max_area,
            )
            image = jnp.clip(image, 0.0, 1.0) * 2.0 - 1.0

        out_images[key] = image

    out_masks = {}
    for key in image_keys:
        out_masks[key] = jnp.asarray(observation.image_masks[key])

    # camera dropout after the per-image ops: the model sees a zero (grey) image with mask False
    if train:
        for cam_idx, key in enumerate(image_keys):
            drop = jax.random.bernoulli(camdrop_root_keys[cam_idx], train_aug.cam_dropout_prob, (batch_size,))
            out_images[key] = jnp.where(drop[:, None, None, None], jnp.zeros_like(out_images[key]), out_images[key])
            out_masks[key] = jnp.logical_and(out_masks[key], jnp.logical_not(drop))

    return dataclasses.replace(observation, images=out_images, image_masks=out_masks)
