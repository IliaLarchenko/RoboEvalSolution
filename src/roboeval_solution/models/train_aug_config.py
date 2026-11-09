"""Training-time augmentation config (no JAX / torch imports). Consumed by `preprocess_observation`
(models/observation.py, images and camera dropout) and `PiModified.compute_detailed_loss` (state / velocity dropout).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainAugmentationConfig:
    """Every field is a JIT-time constant; randomness is drawn outside `jax.vmap` (no lax.cond inside the vmaps)."""

    # ColorJitter (augmax), one draw per sample shared by all cameras.
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5

    # Static cameras (head / front / external) only: random crop to this fraction then resize back, and a rotation
    # of up to +-top_rotate_deg; one draw per sample shared by the static cameras.
    top_crop_frac: float = 0.98
    top_rotate_deg: float = 2.0

    # Cutout: with this probability per image, one rectangle of area U(0, cutout_max_area) x H x W, aspect
    # U(0.3, 3), filled with grey.
    cutout_prob: float = 0.3
    cutout_max_area: float = 0.1

    # Whole-camera dropout, independently per camera and sample: the image is zeroed AND image_mask[cam] = False
    # (its tokens leave attention).
    cam_dropout_prob: float = 0.2

    # State dropout: per sample, the state tokens (bins and the two continuous tokens) are masked out of attention
    # (rows and columns) and hidden from the action expert.
    state_dropout_prob: float = 0.2

    # Velocity dropout: per sample, all velocity state dims (PiModifiedConfig.state_velocity_dims) are hidden at once,
    # their bin tokens and the continuous velocity token, so the policy also works without velocities.
    velocity_dropout_prob: float = 0.3

    # Scripted samples (Observation.is_scripted) use these instead: their actions are close to a function of the
    # proprioception, so heavier dropout keeps the model from reading them off the state instead of the images.
    scripted_state_dropout_prob: float = 0.45
    scripted_velocity_dropout_prob: float = 0.8
