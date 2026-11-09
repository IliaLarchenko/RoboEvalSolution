"""PiModified model configuration. The defaults are the recipe of the `pi05_roboeval` training config."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at

from roboeval_solution.constants import ACTION_DIM, ACTION_HORIZON
from roboeval_solution.models.observation import Observation
from roboeval_solution.models.train_aug_config import TrainAugmentationConfig

if TYPE_CHECKING:
    from roboeval_solution.models.pi_modified import PiModified


@dataclasses.dataclass(frozen=True)
class PiModifiedConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # ee-delta actions: 2 x (3 pos + 3 rotvec) + 2 grippers; chunks of 30 steps (1.5 s at 20 Hz).
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    max_token_len: int = 200  # openpi BaseModelConfig field; no text prompt is tokenised

    # Flow noise ~ N(0, D (beta C + (1 - beta) I) D): C the action-chunk correlation matrix (norm stats), D the
    # per-(timestep, dim) std of the globally normalised actions. Actions are normalised globally (one mean / std per
    # dim); the flow loss is weighted per (h, d) by (global std / per-timestep std)^2, so loss, noise and targets have
    # per-timestep scale (PiModified.load_action_loss_weight / _timestep_noise_scale). Requests can pick their own
    # beta from a grid at serving (precompute_beta_grid).
    correlation_beta: float = 0.5

    # FAST auxiliary loss: the prefix predicts the chunk's FAST tokens (training only).
    fast_loss_weight: float = 0.01
    fast_encoded_dims: str = f"0:{ACTION_DIM}"
    fast_vocab_size: int = 1024
    max_fast_tokens: int = 96

    # Inpainting at inference: the observed (anchor) coords are soft-pinned by this fraction per flow step (1.0 =
    # hard pin), the correlated correction of the free coords is capped per step (normalised units), anchors clipped
    # to +-inpaint_anchor_clip; no correction once t <= the request's time_threshold_inpaint.
    inpaint_pin_alpha: float = 0.7
    inpaint_correction_step_cap: float = 0.1
    inpaint_anchor_clip: float = 3.0

    # Advantage conditioning (pi0.6* RECAP): one learned token in the prefix plus an AdaRMS vector in the action
    # expert. Training: advantage < 0 -> always neutral (token hidden, AdaRMS term off); advantage >= 0 -> neutral
    # with P ramping from neutral_prob_at_zero (A = 0) to neutral_prob_at_max (A >= max_adv_for_neutral).
    # Inference: always conditioned; classifier-free guidance v = v_uncond + cfg_scale (v_cond - v_uncond), cfg_scale
    # per request.
    neutral_prob_at_zero: float = 0.5
    neutral_prob_at_max: float = 0.1
    max_adv_for_neutral: float = 2.0

    # Value heads on a learned query token after the images (success / completion / stage / togo).
    success_loss_weight: float = 0.1
    # Completion = t / T on successes, 0 on failures (completion x P(success)).
    completion_loss_weight: float = 0.1
    # Stage head: P(stage j reached by the episode end), masked BCE (mask j < J_f).
    num_stage_outputs: int = 5
    stage_loss_weight: float = 0.05
    # Togo head: the normalised behaviour-to-go (score_reward.TOGO_KEYS; constants.TOGO_HEAD_WIDTH), raw linear output.
    togo_loss_weight: float = 0.05

    # Q heads.
    #   WM-FAST: a query token after the FAST tokens (training only): success, completion at t + 30 and the togo
    #   spent over the chunk, conditioned on the chunk's FAST tokens.
    #   WM-flow: a query token at the tail of the action-expert suffix: the value-vector residual
    #   psi(t) - sg(psi_hat(s_t)), psi = [success, stage x5, togo x11] (wm_flow_vector_dim 17), and completion at
    #   t + 30; loss weight (1 - t), flow samples with t > 0.5 excluded. Inference reads it at the last denoising
    #   step (best-of-N ranking in scripts/serve_policy.py). Component 0 (the Delta success) also has its own loss
    #   term on top of its share of the vector loss.
    wm_fast_success_loss_weight: float = 0.05
    wm_fast_completion_loss_weight: float = 0.05
    wm_fast_togo_loss_weight: float = 0.05
    wm_flow_vector_dim: int = 17
    wm_flow_vector_loss_weight: float = 0.05
    wm_flow_success_loss_weight: float = 0.05
    wm_flow_completion_loss_weight: float = 0.05

    # Elapsed episode time for the value / Q heads only (late fusion). phi(t) = [u, sin 2 pi f_k u, cos 2 pi f_k u],
    # u = episode_step / aux_time_max_steps, f_k geometric in [0.5, 20] cycles per u. Each head group's hidden state
    # (success query: success / stage / togo; WM-FAST token; WM-flow token) becomes h' = h + W2 gelu(W1 [h; phi]),
    # W2 zero-initialised. The VLM and the action tokens never see t. The completion head keeps the raw h:
    # completion = t / T on successes, so with the clock it would read t instead of the scene's progress.
    aux_time_max_steps: float = 800.0  # longest budget (StackSingleBookShelf, 40 s at 20 Hz)
    aux_time_num_freqs: int = 8
    aux_time_hidden: int = 256

    # SigLIP So400m/16 on native 256x256 renders keeps pi0.5's 16x16 token grid; the pi0.5 /14 stem is adjoint-resized
    # at load (training/weight_loaders.py) and the whole vision tower is trained.
    siglip_variant: str = "So400m/16"
    image_resolution: tuple[int, int] = (256, 256)
    # Model-side camera keys (data/dataset.py DEFAULT_CAMERA_MAP). Each camera adds 256 image tokens and has its own
    # identity-initialised Linear on the shared SigLIP output (camera_proj_<key>).
    image_keys: tuple[str, ...] = ("head_rgb", "front_rgb", "external_rgb", "left_rgb", "right_rgb")

    # Training-time image + state augmentation (models/train_aug_config.py).
    train_aug: TrainAugmentationConfig = dataclasses.field(default_factory=TrainAugmentationConfig)

    # State: one token per dim = 256-bin embedding (dedicated table, the Gemma vocab is unused) + per-dim embedding +
    # Linear(sincos(raw normalised value)), preceded by two continuous tokens Linear(non-velocity dims) and
    # Linear(velocities). state_dim must match the data's state (data/action_repr.make_state: 14 arm qpos, 2 grippers,
    # 14 arm qvel); state_velocity_dims = [start, end) of the velocities (hidden together by velocity dropout).
    state_dim: int = 30
    state_velocity_dims: tuple[int, int] = (16, 30)
    state_sincos_dim: int = 256
    state_sincos_min_period: float = 0.01
    state_sincos_max_period: float = 4.0

    def get_fast_dim_ranges(self) -> list[tuple[int, int]]:
        """Parse fast_encoded_dims into list of ranges."""
        ranges = []
        for range_str in self.fast_encoded_dims.split(','):
            start, end = map(int, range_str.strip().split(':'))
            ranges.append((start, end))
        return ranges

    @property
    @override
    def model_type(self):
        return "pi_modified"

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiModified":
        from roboeval_solution.models.pi_modified import PiModified

        return PiModified(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple["Observation", _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *self.image_resolution, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        f32 = lambda *shape: jax.ShapeDtypeStruct([batch_size, *shape], jnp.float32)  # noqa: E731

        with at.disable_typechecking():
            observation_spec = Observation(
                images={k: image_spec for k in self.image_keys},
                image_masks={k: image_mask_spec for k in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
                advantage=f32(),
                task_id=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                episode_step=f32(),
            )

        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
