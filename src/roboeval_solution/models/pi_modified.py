"""PiModified: the pi0.5-based RoboEval policy + value model (adapted from the openpi Pi0.5 implementation)."""

import dataclasses
import logging

import numpy as np
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at

from roboeval_solution.models import pi_modified_config
from roboeval_solution.models.observation import Observation, preprocess_observation
from roboeval_solution.constants import NUM_TASKS
from roboeval_solution.models import flow_sampling as _flow_sampling

logger = logging.getLogger("roboeval_solution")



# Set by load_correlation_matrix, kept outside the NNX module (out of the state / graphdef); one slot, the last loaded
# model. Unshrunk action covariance (host), for PiModified.precompute_beta_grid:
_sigma_raw: np.ndarray | None = None
_noise_scale: np.ndarray | None = None  # flat [H * D] per-(h, d) flow-noise std (_timestep_noise_scale)


def _bce_with_logits(logits, target):
    """BCE from logits in float32 via log_sigmoid: no log(0) cliff and no bf16 precision loss at saturated
    predictions. The heads stay bf16; only the loss is upcast."""
    z = logits.astype(jnp.float32)
    y = target.astype(jnp.float32)
    return -(y * jax.nn.log_sigmoid(z) + (1.0 - y) * jax.nn.log_sigmoid(-z))


class KVCacheTransform(nnx.Module):
    """Learned cross-layer mixing of the prefix KV cache for the action expert.

    Each destination layer's K (and, independently, V) is a linear combination of all source layers plus a bias.
    Identity init (coeffs = I, bias = 0): the model starts as plain layer-by-layer attention.
    """
    
    def __init__(self, num_layers: int, head_dim: int, num_kv_heads: int):
        # coeffs [dest_layer, src_layer]; bias [layer, num_kv_heads, head_dim]
        self.k_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        
        self.k_bias = nnx.Param(jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32))
        
        self.v_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        self.v_bias = nnx.Param(jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32))
    
    def __call__(self, kv_cache: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(cache_k, cache_v), each [num_layers, batch, seq_len, num_kv_heads, head_dim] -> the mixed pair, same
        shape and dtype."""
        cache_k, cache_v = kv_cache
        
        # The float32 coefficients upcast the einsum; the result is cast back to the cache dtype.
        original_dtype = cache_k.dtype
        
        # k_new[dest] = sum_src(k_coeffs[dest, src] * cache_k[src]) + k_bias[dest]
        k_new = jnp.einsum('ds,sbtkh->dbtkh', self.k_coeffs.value, cache_k)
        k_new = k_new + self.k_bias.value[:, None, None, :, :]
        
        v_new = jnp.einsum('ds,sbtkh->dbtkh', self.v_coeffs.value, cache_v)
        v_new = v_new + self.v_bias.value[:, None, None, :, :]
        
        k_new = k_new.astype(original_dtype)
        v_new = v_new.astype(original_dtype)
        
        return (k_new, v_new)


def _identity_kernel_init(key, shape, dtype=jnp.float32):
    """Identity kernel (module level: nnx keeps kernel_init in the graph metadata, which must be
    the same object in every model instance)."""
    return jnp.eye(shape[0], shape[1], dtype=dtype)


def masked_action_loss(action_loss: jax.Array, action_is_pad: jax.Array) -> jax.Array:
    """[B, H, D] weighted flow loss -> [B]: padded steps (the chunk runs past the episode end, hold pose: not actions
    the episode took) are masked out and each sample's sum is divided by the constant H x D, so every valid (frame,
    step, dim) element weighs the same (a near-end chunk with k valid steps carries k / H of a full chunk's weight)."""
    action_loss = action_loss * (~action_is_pad).astype(action_loss.dtype)[..., None]
    return jnp.mean(action_loss, axis=(-2, -1))


def aux_time_features(episode_step: jax.Array, max_steps: float, num_freqs: int) -> jax.Array:
    """[B] env steps since reset -> [B, 1 + 2 num_freqs]: u = step / max_steps and sin / cos(2π f_k u), f_k
    geometric in [0.5, 20] cycles per unit u (the value / Q heads' time input, PiModifiedConfig.aux_time_*)."""
    u = episode_step.astype(jnp.float32) / max_steps
    freqs = jnp.geomspace(0.5, 20.0, num_freqs, dtype=jnp.float32)
    ang = 2.0 * jnp.pi * u[:, None] * freqs[None, :]
    return jnp.concatenate([u[:, None], jnp.sin(ang), jnp.cos(ang)], axis=-1)


class AuxTimeFusion(nnx.Module):
    """h' = h + W2 gelu(W1 [h; φ(t)]): the elapsed-time features concatenated to the hidden state a head group reads.
    W2 is zero-initialised, so h' = h (and every head output) is unchanged until training moves it."""

    def __init__(self, width: int, time_dim: int, hidden: int, rngs: nnx.Rngs):
        self.inp = nnx.Linear(width + time_dim, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, h: jax.Array, time_feat: jax.Array) -> jax.Array:
        x = jnp.concatenate([h.astype(jnp.float32), time_feat.astype(jnp.float32)], axis=-1)
        return (h.astype(jnp.float32) + self.out(jax.nn.gelu(self.inp(x)))).astype(h.dtype)


class PiModified(_model.BaseModel):
    def __init__(self, config: pi_modified_config.PiModifiedConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        from roboeval_solution.constants import TOGO_HEAD_WIDTH as _TOGO_W

        # Gemma with AdaRMS in the action expert (pi0.5)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        # The 257K-row Gemma vocab table is never used (no text prompt; the state has its own 256-row table) but the
        # submodule allocates it: it is replaced by a 1-row dummy (frozen, TrainConfig.trainable_filter) to save memory.
        llm.embedder['input_embedding'].value = jnp.zeros((1, paligemma_config.width), dtype=jnp.float32)

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=config.siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # The action expert attends to learned per-layer mixes of the VLM layers' KV (identity at init).
        self.kv_transform = KVCacheTransform(
            num_layers=paligemma_config.depth,
            head_dim=paligemma_config.head_dim,
            num_kv_heads=paligemma_config.num_kv_heads,
        )

        # 256-bin state token table (x sqrt(width), the Gemma embedding convention)
        self.state_embedding = nnx.Embed(num_embeddings=256, features=paligemma_config.width, rngs=rngs)

        # pi0.5 action-expert projections
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Set from the norm stats by load_correlation_matrix (nnx.Intermediate: not trained, but part of the state and
        # so of the checkpoints): the Cholesky factor of the flow-noise covariance [H*D, H*D], the per-(h, d) flow-loss
        # weight [H, D] and the factor converting the loss to per-timestep-normalised units for the action_loss_pts*
        # metrics [H, D].
        flat_dim = config.action_horizon * config.action_dim
        self.action_correlation_cholesky = nnx.Intermediate(jnp.eye(flat_dim))
        self.correlation_beta = config.correlation_beta  # shrinkage towards the identity
        self.action_loss_weight = nnx.Intermediate(jnp.ones((config.action_horizon, config.action_dim)))
        self.action_pts_scale = nnx.Intermediate(jnp.ones((config.action_horizon, config.action_dim)))

        # FAST auxiliary: token embedding (prefix width) and the projection back to the FAST vocab
        self.fast_token_embedding = nnx.Embed(
            num_embeddings=config.fast_vocab_size,
            features=paligemma_config.width,
            rngs=rngs
        )
        self.fast_token_proj = nnx.Linear(paligemma_config.width, config.fast_vocab_size, rngs=rngs)

        # Advantage conditioning (pi0.6* style): the token is advantage_embeddings[0]; row 1 is never used (kept: it is
        # part of the checkpointed param tree). The AdaRMS vector (action-expert width, zero-init) is added to the
        # action expert's conditioning when the token is active.
        self.advantage_embeddings = nnx.Embed(
            num_embeddings=2,
            features=paligemma_config.width,
            rngs=rngs,
        )
        self.advantage_adarms_vec = nnx.Param(
            jnp.zeros((1, action_expert_config.width), dtype=jnp.float32)
        )

        # Task conditioning: a token in the prefix (with the images, attention group 0) and a zero-init AdaRMS channel
        # (action-expert width) per task.
        self.task_input_embedding = nnx.Embed(
            num_embeddings=NUM_TASKS,
            features=paligemma_config.width,
            rngs=rngs,
        )
        self.task_adarms_embedding = nnx.Embed(
            num_embeddings=NUM_TASKS,
            features=action_expert_config.width,
            embedding_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        # Value heads on a learned query token after the images: success, task (untrained), completion, stage, togo.
        self.success_query_token = nnx.Param(jax.random.normal(rngs(), (1, paligemma_config.width)) * 0.02)
        self.success_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
        self.task_head = nnx.Linear(paligemma_config.width, NUM_TASKS, rngs=rngs)
        self.completion_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
        self.stage_head = nnx.Linear(paligemma_config.width, config.num_stage_outputs, rngs=rngs)
        self.togo_head = nnx.Linear(paligemma_config.width, _TOGO_W, rngs=rngs)

        # WM-FAST (training only): a query token after the FAST tokens (a new attention group: it sees everything
        # before it, nothing sees it); success, completion at t + 30 and the togo spent over the chunk. Its gradients
        # reach the VLM.
        self.wm_fast_query_token = nnx.Param(
            jax.random.normal(rngs(), (1, paligemma_config.width)) * 0.02
        )
        self.wm_fast_success_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
        self.wm_fast_completion_head = nnx.Linear(paligemma_config.width, 1, rngs=rngs)
        self.wm_fast_togo_head = nnx.Linear(paligemma_config.width, _TOGO_W, rngs=rngs)

        # WM-flow: a query token at the tail of the action-expert suffix (bidirectional with the noisy action tokens):
        # the value-vector residual psi(t) - sg(psi_hat(s_t)), psi = [success, stage x S, togo x K], and completion at
        # t + 30.
        if config.wm_flow_vector_dim != 1 + config.num_stage_outputs + _TOGO_W:
            raise ValueError(f"wm_flow_vector_dim {config.wm_flow_vector_dim} != 1 + "
                             f"{config.num_stage_outputs} + {_TOGO_W}")
        self.wm_flow_query_token = nnx.Param(
            jax.random.normal(rngs(), (1, action_expert_config.width)) * 0.02
        )
        self.wm_flow_vector_head = nnx.Linear(action_expert_config.width, config.wm_flow_vector_dim, rngs=rngs)
        self.wm_flow_completion_head = nnx.Linear(action_expert_config.width, 1, rngs=rngs)

        # Elapsed time for the value / Q heads only (late fusion, zero-init: identity at step 0).
        tdim = 1 + 2 * config.aux_time_num_freqs
        self.aux_time_value = AuxTimeFusion(paligemma_config.width, tdim, config.aux_time_hidden, rngs)
        self.aux_time_wm_fast = AuxTimeFusion(paligemma_config.width, tdim, config.aux_time_hidden, rngs)
        self.aux_time_wm_flow = AuxTimeFusion(action_expert_config.width, tdim, config.aux_time_hidden, rngs)

        # Input encodings (identity / zero init, except the continuous state tokens).
        # Per-camera Linear on the shared SigLIP output (one attribute per camera: nnx.Dict graph metadata is not
        # stable across instances).
        for key in config.image_keys:
            setattr(self, f"camera_proj_{key}", nnx.Linear(
                paligemma_config.width, paligemma_config.width,
                kernel_init=_identity_kernel_init, bias_init=nnx.initializers.zeros, rngs=rngs))
        # One embedding per state dimension added to its bin token.
        self.state_dim_embedding = nnx.Param(jnp.zeros((config.state_dim, paligemma_config.width), jnp.float32))
        # W . sincos(x_j) on the raw (unclipped) normalised value, added to bin token j.
        self.state_value_proj = nnx.Linear(
            config.state_sincos_dim, paligemma_config.width,
            kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs)
        # Two continuous tokens: Linear(non-velocity dims) and Linear(velocities), so state dropout and velocity
        # dropout can hide them separately.
        v0, v1 = config.state_velocity_dims
        self.state_continuous_proj = nnx.Linear(config.state_dim - (v1 - v0), paligemma_config.width, rngs=rngs)
        self.velocity_continuous_proj = nnx.Linear(v1 - v0, paligemma_config.width, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _compute_advantage_token(
        self,
        advantage: at.Float[at.Array, " b"],
        rng: at.KeyArrayLike | None,
        *,
        inference: bool = False,
    ) -> tuple[at.Float[at.Array, "b d"], at.Bool[at.Array, " b"]]:
        """The advantage token (advantage_embeddings[0]) and its per-sample mask (True = active, False = neutral).

        Inference: always active. Training: advantage < 0 -> neutral; advantage >= 0 -> neutral with P ramping from
        ``neutral_prob_at_zero`` (A = 0) to ``neutral_prob_at_max`` (A >= ``max_adv_for_neutral``). A neutral token is
        zeroed and masked out of attention (and its AdaRMS term is off, embed_suffix).
        """
        batch_size = advantage.shape[0]
        token = self.advantage_embeddings(jnp.zeros((batch_size,), dtype=jnp.int32)).astype(self.config.dtype)
        if inference:
            return token, jnp.ones((batch_size,), dtype=jnp.bool_)

        is_negative = advantage < 0.0
        t = jnp.minimum(advantage / self.config.max_adv_for_neutral, 1.0)
        t = jnp.maximum(t, 0.0)
        p_neutral = (
            (1.0 - t) * self.config.neutral_prob_at_zero
            + t * self.config.neutral_prob_at_max
        )
        uniform = jax.random.uniform(rng, (batch_size,))
        use_neutral = is_negative | (uniform < p_neutral)

        token = jnp.where(use_neutral[:, None], 0.0, token)
        return token, ~use_neutral

    @staticmethod
    def _action_stds(norm_stats: dict) -> tuple[np.ndarray, np.ndarray]:
        """(per-timestep std [H', D'], global std [D']) of the actions, float64."""
        stats = norm_stats["actions"]
        pt_std, g_std = stats.per_timestamp_std, stats.std
        if pt_std is None or g_std is None:
            raise ValueError("norm_stats['actions'] needs std and per_timestamp_std (scripts/prepare_assets.py)")
        return np.asarray(pt_std, np.float64), np.asarray(g_std, np.float64)

    def load_action_loss_weight(self, norm_stats: dict):
        """Actions are normalised globally (one mean / std per dim, all timesteps in the same units); each (h, d) of
        the flow loss is weighted by (global_std[d] / per_timestamp_std[h, d])^2, the MSE weight that, with the flow
        noise scaled to the same per-timestep std (_timestep_noise_scale), puts loss, noise and targets on the
        per-timestep scale. The weights are not mean-normalised, so the action loss is not down-weighted against the
        aux heads."""
        pt_std, g_std = self._action_stds(norm_stats)
        H, D = self.action_horizon, self.action_dim
        w = ((g_std[:D][None, :] + 1e-6) / (pt_std[:H, :D] + 1e-6)) ** 2
        self.action_pts_scale.value = jnp.asarray(w, jnp.float32)
        self.action_loss_weight.value = jnp.asarray(w, jnp.float32)
        logger.info("Action loss weight (global / per-timestep std)^2: min %.3f max %.3f, h=0 mean %.3f, "
                    "h=-1 mean %.3f", w.min(), w.max(), w[0].mean(), w[-1].mean())

    def _timestep_noise_scale(self, norm_stats: dict) -> np.ndarray:
        """Flat [H * D] std of each (h, d) in the globally normalised action units (per_timestamp_std / std): the
        per-element scale of the correlated flow noise."""
        pt_std, g_std = self._action_stds(norm_stats)
        k = (pt_std[: self.action_horizon, : self.action_dim] + 1e-6) / (g_std[: self.action_dim] + 1e-6)
        logger.info("Flow noise scaled to per-timestep std: h=0 mean %.3f, h=-1 mean %.3f (min %.3f max %.3f)",
                    k[0].mean(), k[-1].mean(), k.min(), k.max())
        return k.reshape(-1)

    def load_correlation_matrix(self, norm_stats: dict):
        """Set the norm-stat-derived state (call after init / restore, outside jit): the flow-loss weights and the
        flow-noise covariance Sigma_reg = D (beta C + (1 - beta) I) D, C the action-chunk correlation matrix
        (norm_stats['actions'].action_correlation_cholesky, scripts/prepare_assets.py), D the per-(h, d) noise std
        (_timestep_noise_scale)."""
        self.load_action_loss_weight(norm_stats)
        if not isinstance(norm_stats, dict) or 'actions' not in norm_stats:
            raise ValueError("norm_stats must be a dict with an 'actions' entry (roboeval_solution.shared.normalize.load)")
        chol_matrix = norm_stats['actions'].action_correlation_cholesky
        if chol_matrix is None:
            raise ValueError("norm_stats['actions'] has no action_correlation_cholesky (scripts/prepare_assets.py)")
        expected_dim = self.action_horizon * self.action_dim
        L = jnp.array(chol_matrix)
        if L.shape != (expected_dim, expected_dim):
            raise ValueError(f"action_correlation_cholesky has shape {L.shape}, expected ({expected_dim}, {expected_dim}) "
                             f"(action_horizon={self.action_horizon} x action_dim={self.action_dim})")

        Sigma = L @ L.T
        global _sigma_raw, _noise_scale
        _sigma_raw = np.asarray(Sigma, np.float64)
        _noise_scale = self._timestep_noise_scale(norm_stats)
        beta = self.correlation_beta
        Sigma_reg = (beta * Sigma + (1 - beta) * jnp.eye(Sigma.shape[0])) * jnp.asarray(np.outer(_noise_scale, _noise_scale))
        L_reg = jnp.linalg.cholesky(Sigma_reg)
        if not bool(jnp.all(jnp.isfinite(L_reg))):
            raise RuntimeError(f"Cholesky of the regularised noise covariance failed (beta={beta:.2f}); lower "
                               "correlation_beta")
        self.action_correlation_cholesky.value = L_reg
        logger.info(f"Loaded correlation matrix {L_reg.shape} (beta={beta:.2f} shrinkage)")

    def generate_correlated_noise(
        self,
        rng: at.KeyArrayLike,
        batch_size: int,
    ) -> at.Float[at.Array, "b {self.action_horizon} {self.action_dim}"]:
        """[B, H, D] flow noise ~ N(0, Sigma_reg) (load_correlation_matrix)."""
        if _sigma_raw is None:
            raise RuntimeError("correlated noise needs load_correlation_matrix() first")
        flat_dim = self.action_horizon * self.action_dim
        standard_normal = jax.random.normal(rng, (batch_size, flat_dim))
        correlated_flat = standard_normal @ self.action_correlation_cholesky.value.T
        return correlated_flat.reshape(batch_size, self.action_horizon, self.action_dim)

    def _correction_matrices(self, Sigma: np.ndarray, max_initial: int) -> np.ndarray:
        """[max_initial + 1, F, F] inpainting correction matrices Σ_UO Σ_OO^-1 for covariance Sigma."""
        flat_dim = self.action_horizon * self.action_dim
        all_C = np.zeros((max_initial + 1, flat_dim, flat_dim), dtype=np.float32)
        for n in range(1, max_initial + 1):
            O_list = [t * self.action_dim + d for t in range(n) for d in range(self.action_dim)]
            O_set = set(O_list)
            U_list = [i for i in range(flat_dim) if i not in O_set]
            O_idx = np.array(O_list, dtype=np.int32)
            U_idx = np.array(U_list, dtype=np.int32)

            Sigma_OO = Sigma[np.ix_(O_idx, O_idx)]
            Sigma_UO = Sigma[np.ix_(U_idx, O_idx)]
            eps = 1e-6 * max(np.mean(np.diag(Sigma_OO)), 1.0)
            Sigma_OO_reg = Sigma_OO + eps * np.eye(len(O_idx))

            # correction_small: [|U|, |O|] = Sigma_UO @ Sigma_OO^{-1}
            correction_small = np.linalg.solve(Sigma_OO_reg, Sigma_UO.T).T

            # Scatter into full-space [F, F]: C_full[o, u] = correction_small[u_local, o_local],
            # so that delta @ C_full gives corrections at U positions
            all_C[n][np.ix_(O_idx, U_idx)] = correction_small.T
        logger.info(f"Correction matrices precomputed: {all_C.shape} ({all_C.nbytes / 1024 / 1024:.1f} MB)")
        return all_C

    def precompute_beta_grid(self, betas, max_initial: int = 20) -> tuple:
        """Inference: Cholesky factors and correction matrices of D (beta * Σ + (1 - beta) * I) D for every beta,
        so requests can pick their own shrinkage (sample_actions(correlation_beta=..., beta_tables=...)).
        Betas whose regularised covariance is not positive definite are dropped."""
        if _sigma_raw is None:
            raise RuntimeError("precompute_beta_grid needs load_correlation_matrix() first")
        eye = np.eye(_sigma_raw.shape[0])
        kept, chols, corrs = [], [], []
        for b in betas:
            S = (float(b) * _sigma_raw + (1.0 - float(b)) * eye) * np.outer(_noise_scale, _noise_scale)
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                logger.warning("correlation beta %.2f: not positive definite, skipped", b)
                continue
            kept.append(float(b))
            chols.append(L.astype(np.float32))
            corrs.append(self._correction_matrices(L @ L.T, max_initial))
        tables = (jnp.asarray(kept, jnp.float32), jnp.asarray(np.stack(chols)), jnp.asarray(np.stack(corrs)))
        logger.info("Correlated-noise beta grid %s (%.0f MB)", kept, sum(t.nbytes for t in tables) / 2 ** 20)
        return tables

    @at.typecheck
    def _time_fuse(self, name: str, h: jax.Array, observation: Observation) -> jax.Array:
        """h fused with the elapsed-time features by the AuxTimeFusion `name`."""
        if observation.episode_step is None:
            raise ValueError("the value / Q heads need observation.episode_step (env steps since the episode reset)")
        feat = aux_time_features(observation.episode_step, self.config.aux_time_max_steps,
                                 self.config.aux_time_num_freqs)
        return getattr(self, name)(h, feat)

    def embed_prefix(
        self,
        obs: Observation,
        advantage_token,  # [B, D] advantage embedding (zeroed where neutral)
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        int,  # num_image_tokens: the success query token's position
        int,  # advantage_token_idx
        int | None,  # wm_fast_token_idx: position of the WM-FAST token (None without FAST tokens)
        tuple[int, int],  # state_token_range: [start, end) of the state tokens (continuous + bins)
    ]:
        """
        Prefix: images + success_query + task + [state_cont, velocity_cont, state bins x state_dim] + advantage
        + [FAST tokens] + [WM-FAST query].

        Attention groups (cumsum of ar_mask; a token sees the groups <= its own):
          - group 0: images, success query, task: bidirectional among themselves
          - group 1: state tokens: see group 0 + themselves, invisible to group 0
          - group 2: advantage: sees groups 0-1 + itself (the per-sample neutral mask is applied by the caller)
          - groups 3..: FAST tokens, causal (training only)
          - last group: the WM-FAST query, sees everything before it (training only)
        """
        input_mask = []
        ar_mask = []
        tokens = []

        vision_train_mode = not self.deterministic
        num_image_tokens = 0
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=vision_train_mode)
            image_tokens = getattr(self, f"camera_proj_{name}")(image_tokens).astype(image_tokens.dtype)
            num_image_tokens += image_tokens.shape[1]
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        batch_size_local = obs.state.shape[0]
        # Value query token, right after the images (position num_image_tokens).
        vq = jnp.broadcast_to(self.success_query_token.value, (batch_size_local, 1, self.success_query_token.value.shape[-1]))
        tokens.append(vq)
        input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
        ar_mask += [False]

        # Task token: in group 0 with the images, visible to every token like a language instruction.
        if obs.task_id is None:
            raise ValueError("observation.task_id is required (the task input token)")
        task_token = self.task_input_embedding(jnp.clip(obs.task_id, 0, NUM_TASKS - 1))[:, None, :]  # [B, 1, D]
        tokens.append(task_token)
        input_mask.append(jnp.ones(task_token.shape[:2], dtype=jnp.bool_))
        ar_mask += [False]

        # State: one token per dim = 256-bin embedding (x sqrt(width), Gemma convention) + W.sincos(raw value)
        # + per-dim embedding, preceded by the two continuous tokens.
        b, s_dim = obs.state.shape
        if s_dim != self.config.state_dim:
            raise ValueError(f"state has {s_dim} dims, config.state_dim={self.config.state_dim}")
        discretized_state = jnp.digitize(obs.state, bins=jnp.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = jnp.clip(discretized_state, 0, 255)
        scale = jnp.sqrt(float(self.state_embedding.features))
        state_tokens = jnp.concatenate(
            [self.state_embedding(discretized_state[:, i:i + 1]) * scale for i in range(s_dim)], axis=1)  # [B, S, D]
        feat = posemb_sincos(
            obs.state.astype(jnp.float32).reshape(-1), self.config.state_sincos_dim,
            min_period=self.config.state_sincos_min_period, max_period=self.config.state_sincos_max_period,
        ).reshape(b, s_dim, self.config.state_sincos_dim)  # on the RAW normalised value (no clip)
        state_tokens = state_tokens + self.state_value_proj(feat).astype(state_tokens.dtype)
        state_tokens = state_tokens + self.state_dim_embedding.value.astype(state_tokens.dtype)[None]
        # Continuous tokens on the raw normalised state (digitize / clip discard everything outside [-1, 1]):
        # [Linear(non-velocity dims), Linear(velocities)].
        raw = obs.state.astype(jnp.float32)
        v0, v1 = self.config.state_velocity_dims
        non_vel = jnp.concatenate([raw[:, :v0], raw[:, v1:]], axis=-1)
        cont = [self.state_continuous_proj(non_vel)[:, None, :], self.velocity_continuous_proj(raw[:, v0:v1])[:, None, :]]
        state_tokens = jnp.concatenate([c.astype(state_tokens.dtype) for c in cont] + [state_tokens], axis=1)
        state_token_start = sum(t.shape[1] for t in tokens)
        state_token_range = (state_token_start, state_token_start + state_tokens.shape[1])  # incl. the continuous ones
        tokens.append(state_tokens)
        input_mask.append(jnp.ones((b, state_tokens.shape[1]), dtype=jnp.bool_))
        ar_mask += [True] + [False] * (state_tokens.shape[1] - 1)  # group 1

        # Advantage token: group 2.
        advantage_token_idx = sum(t.shape[1] for t in tokens)
        tokens.append(advantage_token[:, None, :])  # [B, 1, D]
        input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
        ar_mask += [True]

        # FAST tokens (training): teacher forcing [BOS, tok_0, ..., tok_{T-2}], causal, with their own embedding.
        wm_fast_token_idx = None
        if obs.fast_tokens is not None:
            fast_tokens = obs.fast_tokens  # [B, T]
            bos_token = jnp.zeros((fast_tokens.shape[0], 1), dtype=jnp.int32)
            shifted_tokens = jnp.concatenate([bos_token, fast_tokens[:, :-1]], axis=1)
            bos_mask = jnp.ones((fast_tokens.shape[0], 1), dtype=jnp.bool_)
            shifted_mask = jnp.concatenate([bos_mask, obs.fast_token_mask[:, :-1]], axis=1)
            tokens.append(self.fast_token_embedding(shifted_tokens))  # [B, T, D]
            input_mask.append(shifted_mask)
            ar_mask += [True] * shifted_tokens.shape[1]

            # WM-FAST query: a new group after FAST (sees everything before it, invisible to all of it). The action
            # expert drops FAST and this token from the KV cache.
            wm_fast_token_idx = sum(t.shape[1] for t in tokens)
            wm_tok = jnp.broadcast_to(
                self.wm_fast_query_token.value,
                (batch_size_local, 1, self.wm_fast_query_token.value.shape[-1]),
            )
            tokens.append(wm_tok)
            input_mask.append(jnp.ones((batch_size_local, 1), dtype=jnp.bool_))
            ar_mask += [True]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_image_tokens, advantage_token_idx, wm_fast_token_idx, state_token_range

    @at.typecheck
    def embed_suffix(
        self,
        obs: Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        advantage_mask: at.Bool[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        """Action-expert suffix [a_0 .. a_{H-1}, WM-flow query] and its AdaRMS conditioning vector

            c = time_emb + task_adarms_embedding[task] + advantage_adarms_vec * advantage_mask,

        which modulates every RMSNorm of the action expert (the CFG unconditional pass subtracts the advantage term
        itself, sample_actions).
        """
        action_tokens = self.action_in_proj(noisy_actions)
        # flow-matching timestep: sine-cosine embedding -> MLP (pi0.5 AdaRMS conditioning)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        adarms_cond = time_emb  # [B, action_expert_width]

        # task channel (zero-init)
        adarms_cond = adarms_cond + self.task_adarms_embedding(
            jnp.clip(obs.task_id, 0, NUM_TASKS - 1)).astype(adarms_cond.dtype)

        # advantage channel: one vector, gated by the same per-sample mask as the prefix token
        adv_vec = self.advantage_adarms_vec.value.astype(adarms_cond.dtype)  # [1, D]
        adarms_cond = adarms_cond + jnp.where(advantage_mask[:, None], adv_vec, jnp.zeros_like(adv_vec))

        # Action tokens: one new group (the prefix does not see them), bidirectional among themselves. The WM-flow
        # query sits at the TAIL in the same group (bidirectional with the actions); action outputs are read from
        # suffix_out[:, :action_horizon].
        batch_size_local = noisy_actions.shape[0]
        wm_tok = jnp.broadcast_to(
            self.wm_flow_query_token.value,
            (batch_size_local, 1, self.wm_flow_query_token.value.shape[-1]),
        )
        tokens = jnp.concatenate([action_tokens, wm_tok], axis=1)
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1) + [False])
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Unused: training calls compute_detailed_loss()."""
        raise NotImplementedError("Use compute_detailed_loss() instead")

    @override
    def compute_detailed_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, num_flow_samples: int = 1
    ) -> dict[str, at.Float[at.Array, "*b"]]:
        """Per-sample losses and metrics ([B] each; train.py weights and aggregates them).

        The prefix (with the FAST tokens and the WM-FAST query in training) runs once; FAST and WM-FAST leave the KV
        cache, then num_flow_samples (t, noise) draws of the action expert reuse it and their losses are averaged.
        The value / Q head losses are returned separately (train.py adds them IS-weighted and task-balanced);
        total_loss = action flow loss + fast_loss_weight x FAST loss.
        """
        losses = {}

        preprocess_rng, rng = jax.random.split(rng)
        observation = preprocess_observation(
            preprocess_rng, observation, train=True,
            train_aug=self.config.train_aug,
            image_resolution=tuple(self.config.image_resolution),
            image_keys=tuple(self.config.image_keys),
        )
        batch_size = actions.shape[0]
        aug = self.config.train_aug

        # Advantage token (stochastic neutral masking).
        adv_rng, rng = jax.random.split(rng)
        advantage_token, advantage_mask = self._compute_advantage_token(observation.advantage, adv_rng, inference=False)

        # Per-sample dropout probabilities: scripted samples use their own (TrainAugmentationConfig).
        scripted = observation.is_scripted > 0.5

        def drop_prob(base: float, for_scripted: float):
            return jnp.where(scripted, for_scripted, base)

        # Per-sample velocity dropout (all velocity dims together).
        vdrop_rng, rng = jax.random.split(rng)
        velocity_drop = jax.random.bernoulli(
            vdrop_rng, drop_prob(aug.velocity_dropout_prob, aug.scripted_velocity_dropout_prob), (batch_size,))

        # Prefix (incl. the FAST tokens + WM-FAST query when the observation has FAST tokens).
        (prefix_tokens, prefix_mask, prefix_ar_mask, num_image_tokens, adv_token_idx, wm_fast_token_idx,
         state_token_range) = self.embed_prefix(observation, advantage_token)

        # Prefix KV cache (attention groups: embed_prefix).
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # [B, P] prefix tokens hidden by state / velocity dropout: their KV still carries the dropped input,
        # so the action expert must not attend to them either
        prefix_hidden = jnp.zeros(prefix_mask.shape, bool)

        # Per-sample state dropout: zero attention rows+columns for state tokens
        sdrop_rng, rng = jax.random.split(rng)
        state_drop = jax.random.bernoulli(
            sdrop_rng, drop_prob(aug.state_dropout_prob, aug.scripted_state_dropout_prob), (batch_size,))
        s_start, s_end = state_token_range
        keep = jnp.logical_not(state_drop)  # [B]
        prefix_attn_mask = prefix_attn_mask.at[:, :, s_start:s_end].set(
            prefix_attn_mask[:, :, s_start:s_end] & keep[:, None, None]
        )
        # Zero rows: state tokens cannot attend to anything
        prefix_attn_mask = prefix_attn_mask.at[:, s_start:s_end, :].set(
            prefix_attn_mask[:, s_start:s_end, :] & keep[:, None, None]
        )
        prefix_hidden = prefix_hidden.at[:, s_start:s_end].set(state_drop[:, None])

        # Velocity dropout: hide the velocity tokens the same way (rows + columns): the continuous velocity token
        # (right after the continuous state token) and the per-dim velocity bin tokens.
        hide = velocity_drop[:, None] & self._velocity_tokens(state_token_range, prefix_attn_mask.shape[-1])[None, :]
        prefix_attn_mask = prefix_attn_mask & ~hide[:, None, :] & ~hide[:, :, None]
        prefix_hidden = prefix_hidden | hide

        # Neutral advantage: the token's row and column leave attention
        prefix_attn_mask = prefix_attn_mask.at[:, :, adv_token_idx].set(
            prefix_attn_mask[:, :, adv_token_idx] & advantage_mask[:, None]
        )
        prefix_attn_mask = prefix_attn_mask.at[:, adv_token_idx, :].set(
            prefix_attn_mask[:, adv_token_idx, :] & advantage_mask[:, None]
        )

        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache_full = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions_prefix
        )

        # Value heads on the success query token. success / stage / togo read it with the elapsed time
        # (value_hidden); completion reads it raw (completion = t / T on successes: with the clock it would just read
        # t off it). Their stop-gradient predictions are the WM-flow residual baseline psi_hat(s_t).
        success_hidden = prefix_out[:, num_image_tokens, :]  # [B, D]
        value_hidden = self._time_fuse("aux_time_value", success_hidden, observation)
        sp_logit = self.success_head(value_hidden).squeeze(-1)  # [B] (bf16 logit)
        sp_pred = jax.nn.sigmoid(sp_logit)  # [B]
        # raw per-sample outputs (underscore keys: train.py's weighted metrics, not logged directly)
        losses["_success_pred_per_sample"] = sp_pred  # [B]

        success_target = observation.success_target  # [B]
        valid = jnp.isfinite(success_target)  # NaN = no target
        safe_target = jnp.where(valid, success_target, 0.0)
        bce = _bce_with_logits(sp_logit, safe_target)
        losses["success_loss"] = jnp.where(valid, bce, 0.0)  # [B]
        n_valid = jnp.maximum(valid.sum(), 1.0)
        pred_mean = jnp.where(valid, sp_pred, 0.0).sum() / n_valid
        losses["success_pred_mean"] = pred_mean * jnp.ones(batch_size)
        pred_var = jnp.where(valid, (sp_pred - pred_mean) ** 2, 0.0).sum() / jnp.maximum(n_valid - 1, 1.0)
        losses["success_pred_std"] = jnp.where(n_valid > 1, jnp.sqrt(pred_var), 0.0) * jnp.ones(batch_size)
        losses["success_target_mean"] = safe_target.sum() / n_valid * jnp.ones(batch_size)
        losses["success_mae"] = jnp.where(valid, jnp.abs(sp_pred - safe_target), 0.0)  # [B]
        losses["success_valid_frac"] = (valid.sum() / batch_size) * jnp.ones(batch_size)

        comp_pred = jax.nn.sigmoid(self.completion_head(success_hidden).squeeze(-1))  # [B]
        losses["_completion_pred_per_sample"] = comp_pred  # [B]

        # Completion: MSE on frames with a target (t / T on successes, 0 on failures)
        comp_target = observation.completion_target  # [B]
        comp_valid = jnp.isfinite(comp_target)
        comp_safe = jnp.where(comp_valid, comp_target, 0.0)
        comp_mse = (comp_pred - comp_safe) ** 2
        losses["completion_loss"] = jnp.where(comp_valid, comp_mse, 0.0)  # [B]
        losses["completion_pred_mean"] = jnp.mean(comp_pred) * jnp.ones(batch_size)

        # Stage head: masked BCE over the final stage flags (mask j < J_f), per-sample mean over valid slots.
        st_logit = self.stage_head(value_hidden)  # [B, 5] (bf16 logits)
        stage_pred = jax.nn.sigmoid(st_logit.astype(jnp.float32))  # [B, 5]
        st_mask_f = observation.stage_mask.astype(jnp.float32)
        st_safe = jnp.where(observation.stage_mask, observation.stage_target, 0.0)
        st_bce = _bce_with_logits(st_logit, st_safe) * st_mask_f  # [B, 5]
        st_n = jnp.maximum(st_mask_f.sum(axis=-1), 1.0)
        losses["stage_loss"] = st_bce.sum(axis=-1) / st_n  # [B]
        losses["stage_accuracy_per_sample"] = (
            ((stage_pred > 0.5) == (st_safe > 0.5)).astype(jnp.float32) * st_mask_f
        ).sum(axis=-1) / st_n  # [B]
        losses["stage_pred_mean"] = jnp.mean(stage_pred) * jnp.ones(batch_size)

        # Togo head: normalised behaviour-to-go regression (raw linear output), per-sample mean over valid slots.
        kpt_pred = self.togo_head(value_hidden)  # [B, 11]
        togo_pred = kpt_pred.astype(jnp.float32)
        kpt_target = observation.togo_target  # [B, 11]
        kpt_mask = observation.togo_mask       # [B, 11] bool
        kpt_mask_f = kpt_mask.astype(kpt_pred.dtype)
        safe_target = jnp.where(kpt_mask, kpt_target, 0.0)
        slot_sq = (kpt_pred - safe_target) ** 2 * kpt_mask_f  # [B, 11]
        n_valid_per_sample = jnp.maximum(kpt_mask_f.sum(axis=-1), 1.0)
        losses["togo_loss"] = slot_sq.sum(axis=-1) / n_valid_per_sample  # [B]
        losses["togo_mae_per_sample"] = (
            jnp.abs(kpt_pred - safe_target) * kpt_mask_f
        ).sum(axis=-1) / n_valid_per_sample  # [B]

        # FAST loss (teacher forcing) from the prefix outputs: FAST occupies [P - T - 1, P - 1), the WM-FAST query
        # sits last.
        wm_fast_tail = 1
        fast_targets = observation.fast_tokens
        fast_token_mask = observation.fast_token_mask

        fast_len = fast_targets.shape[1]
        fast_start_idx = prefix_tokens.shape[1] - fast_len - wm_fast_tail
        fast_outputs = prefix_out[:, fast_start_idx:fast_start_idx + fast_len, :]  # [B, T, D]

        fast_logits = self.fast_token_proj(fast_outputs)  # [B, T, vocab_size]

        pred_logits = fast_logits  # [B, T, vocab]
        target_tokens = fast_targets  # [B, T]
        loss_mask = fast_token_mask  # [B, T]

        log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs,
            target_tokens[:, :, None],
            axis=-1
        ).squeeze(-1)  # [B, T]

        fast_token_loss = -target_log_probs  # [B, T]

        # Mean over the valid tokens
        masked_loss = fast_token_loss * loss_mask  # [B, T]
        num_valid_tokens = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1)  # [B]
        losses["fast_loss"] = jnp.sum(masked_loss, axis=-1) / num_valid_tokens  # [B]

        # Accuracy on the valid tokens
        pred_tokens = jnp.argmax(pred_logits, axis=-1)
        correct = (pred_tokens == target_tokens) * loss_mask
        losses["fast_accuracy"] = jnp.sum(correct, axis=-1) / num_valid_tokens

        fast_loss_value = self.config.fast_loss_weight * jnp.mean(losses["fast_loss"])

        # WM-FAST head: success, completion at t + 30 and the togo spent over the chunk, from the query after FAST.
        wm_fast_raw = prefix_out[:, wm_fast_token_idx, :]  # [B, D]; completion reads it without the clock
        wm_fast_hidden = self._time_fuse("aux_time_wm_fast", wm_fast_raw, observation)
        wm_fast_s_logit = self.wm_fast_success_head(wm_fast_hidden).squeeze(-1)  # [B] (bf16)
        wm_fast_c_pred = jax.nn.sigmoid(self.wm_fast_completion_head(wm_fast_raw).squeeze(-1))  # [B]
        wm_fast_k_pred = self.wm_fast_togo_head(wm_fast_hidden)  # [B, 11] togo spent over the chunk

        # Future success (stable BCE from logits, NaN-masked)
        s_target = observation.success_future_target
        s_valid = jnp.isfinite(s_target)
        s_safe = jnp.where(s_valid, s_target, 0.0)
        s_bce = _bce_with_logits(wm_fast_s_logit, s_safe)
        losses["wm_fast_success_loss"] = jnp.where(s_valid, s_bce, 0.0)

        # Future completion (MSE, NaN-masked)
        c_target = observation.completion_future_target
        c_valid = jnp.isfinite(c_target)
        c_safe = jnp.where(c_valid, c_target, 0.0)
        c_mse = (wm_fast_c_pred - c_safe) ** 2
        losses["wm_fast_completion_loss"] = jnp.where(c_valid, c_mse, 0.0)

        # Togo spent over the chunk, togo[t] - togo[t+30] (masked MSE)
        k_target = observation.togo_future_target
        k_mask = observation.togo_future_mask
        k_mask_f = k_mask.astype(wm_fast_k_pred.dtype)
        k_safe = jnp.where(k_mask, k_target, 0.0)
        k_sq = (wm_fast_k_pred - k_safe) ** 2 * k_mask_f
        k_denom = jnp.maximum(k_mask_f.sum(axis=-1), 1.0)
        losses["wm_fast_togo_loss"] = k_sq.sum(axis=-1) / k_denom

        # The action expert attends to neither FAST nor WM-FAST: drop both from the end of the KV cache
        # [layers, batch, seq_len, num_kv_heads, head_dim].
        tail_drop = fast_len + wm_fast_tail
        cache_k, cache_v = kv_cache_full
        cache_k = cache_k[:, :, :-tail_drop, :, :]
        cache_v = cache_v[:, :, :-tail_drop, :, :]
        kv_cache_for_actions = (cache_k, cache_v)
        prefix_len_for_actions = prefix_tokens.shape[1] - tail_drop
        prefix_mask_for_actions = prefix_mask[:, :-tail_drop]
        # suffix RoPE positions start after the valid prefix tokens, as in sample_actions (camera dropout
        # leaves gaps in prefix_mask); counted before the advantage / dropout masking, like at inference
        num_prefix_valid = jnp.sum(prefix_mask_for_actions, axis=-1)[:, None]  # [B, 1]

        # Tokens hidden by state / velocity dropout are hidden from the action expert too
        prefix_mask_for_actions = prefix_mask_for_actions & ~prefix_hidden[:, :prefix_len_for_actions]

        # A neutral advantage token is hidden from the action expert too (it sits before FAST: same index)
        prefix_mask_for_actions = prefix_mask_for_actions.at[:, adv_token_idx].set(
            prefix_mask_for_actions[:, adv_token_idx] & advantage_mask
        )

        # Cross-layer KV mixing for the action expert
        kv_cache_for_actions = self.kv_transform(kv_cache_for_actions)

        # WM-flow value-vector residual target psi(t) - sg(psi_hat(s_t)), psi = [success, stage x S, togo x K], the
        # baseline the prefix heads' stop-gradient predictions; the same for every flow sample.
        V = self.config.wm_flow_vector_dim
        n_stage = self.config.num_stage_outputs
        psi_hat = jax.lax.stop_gradient(jnp.concatenate(
            [sp_pred.astype(jnp.float32).reshape(batch_size, 1),
             stage_pred.astype(jnp.float32).reshape(batch_size, n_stage),
             togo_pred.astype(jnp.float32).reshape(batch_size, V - 1 - n_stage)], axis=-1))  # [B, V]
        v_mask = observation.wm_vector_mask & jnp.isfinite(observation.wm_vector_target)  # [B, V]
        v_resid_target = jnp.where(v_mask, observation.wm_vector_target.astype(jnp.float32) - psi_hat, 0.0)
        v_mask_f = v_mask.astype(jnp.float32)

        def process_one_flow_sample(time, noise):
            """One flow sample (t [B], noise [B, H, D]) on the cached prefix -> (action_loss [B, H, D],
            wm_v_sq [B, V] (masked, (1 - t)-weighted squared residual errors), wm_c_loss [B], t_valid [B],
            wm_v_pred [B, V]). The suffix is [a_0 .. a_{H-1}, WM-flow query]: actions are read from the first H
            outputs, the WM-flow heads from the last."""
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time, advantage_mask=advantage_mask
            )
            # the suffix attends to the prefix (without FAST / WM-FAST) and to itself
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(
                prefix_mask_for_actions, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            suffix_positions = num_prefix_valid + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache_for_actions,
                adarms_cond=[None, adarms_cond]
            )

            v_t = self.action_out_proj(suffix_out[:, :self.action_horizon])
            action_loss_i = jnp.square(v_t - u_t)  # [B, H, D]

            # WM-flow head: loss weight (1 - t); flow samples with t > 0.5 (mostly noise) are excluded, and t_valid
            # tells train.py which samples contributed (its valid-only metric denominators).
            # t in [0.001, 1.0]: t=1 -> pure noise, t=0 -> clean.
            _WM_FLOW_T_MAX = 0.5
            t_valid_i = time <= _WM_FLOW_T_MAX  # [B] bool
            wm_flow_raw = suffix_out[:, -1, :]  # [B, D_action]; completion reads it without the clock
            wm_flow_hidden = self._time_fuse("aux_time_wm_flow", wm_flow_raw, observation)
            wm_v_pred = self.wm_flow_vector_head(wm_flow_hidden).astype(jnp.float32)  # [B, V] residual, raw
            wm_c_pred = jax.nn.sigmoid(self.wm_flow_completion_head(wm_flow_raw).squeeze(-1))  # [B]
            w = jnp.where(t_valid_i, 1.0 - time, 0.0).astype(jnp.float32)  # [B]
            wm_v_sq_i = (wm_v_pred - v_resid_target) ** 2 * v_mask_f * w[:, None]  # [B, V]

            c_target = observation.completion_future_target
            c_valid = jnp.isfinite(c_target) & t_valid_i
            c_safe = jnp.where(c_valid, c_target, 0.0)
            c_mse = (wm_c_pred - c_safe) ** 2
            wm_c_loss_i = jnp.where(c_valid, c_mse, 0.0) * w.astype(c_mse.dtype)
            return action_loss_i, wm_v_sq_i, wm_c_loss_i, t_valid_i, wm_v_pred

        # (t, noise) of the N flow samples (models/flow_sampling.py), vectorised over N
        flow_t, flow_noise = _flow_sampling.sample_t_and_noise(
            rng, num_flow_samples, batch_size, self.generate_correlated_noise)
        with at.disable_typechecking():
            (
                all_action_losses, all_wm_v_sq, all_wm_c, all_t_valid, all_wm_v_pred,
            ) = jax.vmap(process_one_flow_sample)(flow_t, flow_noise)

        # Average over the flow samples
        action_loss = jnp.mean(all_action_losses, axis=0)  # [B, H, D]
        # Unweighted loss in per-timestep-normalised units (comparable across normalisations), per horizon.
        pts_loss = action_loss * self.action_pts_scale.value[None].astype(action_loss.dtype)
        pts_valid = (~observation.action_is_pad).astype(action_loss.dtype)
        for name, (h0, h1) in (("", (0, None)), ("_h0", (0, 1)), ("_h1_4", (1, 5)), ("_h5_14", (5, 15)),
                               ("_h15_29", (15, None))):
            v = pts_valid[:, h0:h1]
            losses["action_loss_pts" + name] = (pts_loss[:, h0:h1].mean(-1) * v).sum(-1) / jnp.maximum(v.sum(-1), 1.0)
        action_loss = action_loss * self.action_loss_weight.value[None].astype(action_loss.dtype)
        losses["action_loss"] = masked_action_loss(action_loss, observation.action_is_pad)
        # fraction of the batch's chunk steps past the episode end (~action_horizon / mean episode length)
        losses["action_pad_frac"] = observation.action_is_pad.astype(jnp.float32).mean() * jnp.ones(batch_size)

        # WM-flow losses are averaged over each sample's valid (t <= 0.5) flow samples, not all N, so the invalid ones
        # do not dilute the gradient. A sample without any valid flow sample gets loss 0 and is marked invalid
        # (``_wm_flow_t_valid_per_sample``).
        all_t_valid_f = all_t_valid.astype(jnp.float32)
        valid_count_per_sample = all_t_valid_f.sum(axis=0)  # [B]
        has_any_valid = valid_count_per_sample > 0              # [B] bool
        valid_count_safe = jnp.maximum(valid_count_per_sample, 1.0)
        # Value-vector residual: per-sample mean over the valid components (wm_flow_vector_loss) and per block
        # (success, stage, togo).
        v_sq = jnp.sum(all_wm_v_sq, axis=0) / valid_count_safe[:, None]  # [B, V]

        def _block(lo, hi):
            n = v_mask_f[:, lo:hi].sum(axis=-1)
            return jnp.where(has_any_valid & (n > 0), v_sq[:, lo:hi].sum(axis=-1) / jnp.maximum(n, 1.0), 0.0)

        losses["wm_flow_vector_loss"] = _block(0, V)
        losses["wm_flow_success_loss"] = _block(0, 1)
        losses["wm_flow_stage_loss"] = _block(1, 1 + n_stage)
        losses["wm_flow_togo_loss"] = _block(1 + n_stage, V)
        losses["wm_flow_completion_loss"] = jnp.where(
            has_any_valid, jnp.sum(all_wm_c, axis=0) / valid_count_safe, 0.0
        )
        # Per-sample valid flag (at least one low-noise flow sample): train.py's metric denominators.
        losses["_wm_flow_t_valid_per_sample"] = has_any_valid  # [B] bool
        # Fraction of valid flow samples over all N x B (P(t <= 0.5) ~ 0.35 under Beta(1.5, 1)).
        _wm_flow_t_valid_frac_scalar = all_t_valid_f.mean()
        losses["_wm_flow_t_valid_frac_scalar"] = _wm_flow_t_valid_frac_scalar * jnp.ones(batch_size)

        # Δsuccess (component 0) diagnostics: prediction mean / std over the valid flow samples, target mean over the
        # samples that contributed to the gradient.
        all_wm_s_pred = all_wm_v_pred[..., 0]  # [N, B]
        total_t_valid_scalar = all_t_valid_f.sum()
        wm_s_pred_sum = (all_wm_s_pred * all_t_valid_f).sum()
        wm_s_pred_mean_scalar = jnp.where(
            total_t_valid_scalar > 0,
            wm_s_pred_sum / jnp.maximum(total_t_valid_scalar, 1.0),
            0.0,
        )
        losses["wm_flow_success_pred_mean"] = wm_s_pred_mean_scalar * jnp.ones(batch_size)
        wm_s_pred_var_sum = ((all_wm_s_pred - wm_s_pred_mean_scalar) ** 2 * all_t_valid_f).sum()
        wm_s_pred_var = jnp.where(
            total_t_valid_scalar > 1,
            wm_s_pred_var_sum / jnp.maximum(total_t_valid_scalar - 1, 1.0),
            0.0,
        )
        losses["wm_flow_success_pred_std"] = jnp.where(
            total_t_valid_scalar > 1, jnp.sqrt(wm_s_pred_var), 0.0
        ) * jnp.ones(batch_size)
        tgt_valid_f = (v_mask[:, 0] & has_any_valid).astype(jnp.float32)
        n_tgt_valid = tgt_valid_f.sum()
        tgt_mean = jnp.where(n_tgt_valid > 0, (v_resid_target[:, 0] * tgt_valid_f).sum() / jnp.maximum(n_tgt_valid, 1.0), 0.0)
        losses["wm_flow_success_target_mean"] = tgt_mean * jnp.ones(batch_size)

        # Total loss (the value / Q head losses are added in train.py's train_step)
        losses["total_loss"] = losses["action_loss"] + fast_loss_value
        
        return losses

    def _velocity_tokens(self, state_token_range: tuple[int, int], length: int) -> jax.Array:
        """[length] bool: the prefix positions of the velocity tokens (the continuous velocity token right after the
        continuous state token, and the per-dim bins of the velocity dims), the ones velocity dropout hides."""
        v0, v1 = self.config.state_velocity_dims
        s0 = state_token_range[0]
        return jnp.zeros(length, bool).at[jnp.array([s0 + 1] + list(range(s0 + 2 + v0, s0 + 2 + v1)))].set(True)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        num_steps: at.Int[at.Array, " b"],  # [B] flow steps per element
        # Mask-based inpainting (supports batching with different inpaint lengths per element):
        inpaint_targets: at.Float[at.Array, "b ah ad"],  # [B, H, D] padded target values
        inpaint_mask: at.Float[at.Array, "b flat"],      # [B, H * D] 1=observed
        inpaint_lengths: at.Int[at.Array, " b"],         # [B] num initial actions per element
        time_threshold_inpaint: at.Float[at.Array, " b"],  # [B]
        inpaint_pin_alpha: at.Float[at.Array, " b"],  # [B]
        inpaint_correction_step_cap: at.Float[at.Array, " b"],  # [B]
        noise_temperature: at.Float[at.Array, " b"],  # [B]
        cfg_scale: at.Float[at.Array, " b"],  # [B] Classifier-Free Guidance scale
        correlation_beta: at.Float[at.Array, " b"],  # [B], snapped to the nearest beta_tables entry
        beta_tables: tuple,  # (betas [G], cholesky [G, F, F], corrections [G, n, F, F]) from
                             # precompute_beta_grid; passed as arguments, not baked constants
        num_candidates: int = 1,  # best-of-N: the prefix runs once on the B requests, then the KV cache and the
                                  # observation are tiled to B*N for the flow loop (independent noise per
                                  # candidate). Static under JIT.
        hide_velocity: at.Bool[at.Array, " b"],  # [B]: velocity tokens hidden (as velocity dropout in training)
        hide_cameras: at.Bool[at.Array, "b c"],  # [B, cameras in image_keys order]: camera dropped
                                                 # (as camera dropout in training)
        time_warp: at.Float[at.Array, " b"],  # [B]: flow time grid t_i = u_i^p, u_i = 1 - i / num_steps (1 = uniform)
    ) -> _model.Actions:
        observation = preprocess_observation(
            None, observation, train=False,
            train_aug=self.config.train_aug,
            image_resolution=tuple(self.config.image_resolution),
            image_keys=tuple(self.config.image_keys),
        )
        # as camera dropout in training: a zero (grey) image whose tokens leave attention
        hc = jnp.asarray(hide_cameras, bool)
        images, masks = dict(observation.images), dict(observation.image_masks)
        for c, key in enumerate(self.config.image_keys):
            images[key] = jnp.where(hc[:, c, None, None, None], jnp.zeros_like(images[key]), images[key])
            masks[key] = masks[key] & ~hc[:, c]
        observation = dataclasses.replace(observation, images=images, image_masks=masks)
        # t=1 is noise, t=0 is target (opposite of pi0 paper convention)
        # num_steps [B]: per-request step counts. The batch runs max(num_steps) iterations and each
        # element stops updating after its own count.
        B_unique = observation.state.shape[0]  # rows of unique requests
        flat_dim = self.action_horizon * self.action_dim

        if observation.fast_tokens is not None:
            raise ValueError(
                "FAST tokens must not be provided during inference (sample_actions). "
                "FAST tokens are only used during training for auxiliary loss. "
                "Set observation.fast_tokens=None before calling sample_actions."
            )

        # Phase 1: the prefix at B_unique (once per request).
        # Advantage token: always active at inference (the CFG uncond pass hides it from the suffix below).
        adv_token_inference, adv_mask_inference = self._compute_advantage_token(
            jnp.zeros(B_unique, dtype=jnp.float32), rng=None, inference=True
        )

        # KV cache of the prefix (no FAST tokens at inference, so no WM-FAST token either).
        prefix_tokens, prefix_mask, prefix_ar_mask, num_image_tokens, adv_token_idx, _, state_token_range = \
            self.embed_prefix(observation, adv_token_inference)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # Hidden velocity: those tokens' rows and columns leave attention, in the prefix and for the action expert,
        # exactly as velocity dropout in training; RoPE positions stay those of the full prefix (prefix_mask).
        vel_hidden = (jnp.asarray(hide_velocity, bool)[:, None]
                      & self._velocity_tokens(state_token_range, prefix_mask.shape[-1])[None, :])  # [B, P]
        prefix_attn_mask = prefix_attn_mask & ~vel_hidden[:, None, :] & ~vel_hidden[:, :, None]

        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # Prefix-derived prediction heads (still at B_unique), by served key.
        preds: dict[str, jax.Array] = {}
        success_hidden = prefix_out[:, num_image_tokens, :]  # [B_unique, D]
        value_hidden = self._time_fuse("aux_time_value", success_hidden, observation)  # + elapsed time
        preds["success_pred"] = jax.nn.sigmoid(self.success_head(value_hidden).squeeze(-1))
        preds["completion_pred"] = jax.nn.sigmoid(self.completion_head(success_hidden).squeeze(-1))  # no clock
        preds["stage_pred"] = jax.nn.sigmoid(self.stage_head(value_hidden).astype(jnp.float32))  # [B_unique, 5]
        preds["togo_pred"] = self.togo_head(value_hidden).astype(jnp.float32)  # [B_unique, 11]

        # Cross-layer KV mixing for the action expert
        kv_cache = self.kv_transform(kv_cache)

        # Phase 2: tile the prefix outputs to B_unique * num_candidates. The prefix is identical for the N candidates
        # of a request, so the KV cache, masks and observation are tiled instead of rerunning the VLM; the candidates
        # diverge through independent noise drawn at B*N below.
        def _rep(x):
            return jnp.repeat(x, num_candidates, axis=0)

        observation = jax.tree.map(
            lambda x: jnp.repeat(x, num_candidates, axis=0)
                if (hasattr(x, "shape") and getattr(x, "ndim", 0) > 0
                    and x.shape[0] == B_unique)
                else x,
            observation,
        )
        # KV cache layout [layers, batch, t, kv_heads, head_dim]: the batch axis is 1.
        kv_cache = jax.tree.map(
            lambda x: jnp.repeat(x, num_candidates, axis=1), kv_cache,
        )
        prefix_mask = _rep(prefix_mask)
        vel_hidden = _rep(vel_hidden)
        adv_mask_inference = _rep(adv_mask_inference)
        preds = {k: _rep(v) for k, v in preds.items()}
        noise_temperature = _rep(noise_temperature)
        time_warp = _rep(time_warp)
        time_threshold_inpaint = _rep(time_threshold_inpaint)
        cfg_scale = _rep(cfg_scale)
        correlation_beta = _rep(correlation_beta)
        inpaint_pin_alpha = _rep(inpaint_pin_alpha)
        inpaint_correction_step_cap = _rep(inpaint_correction_step_cap)
        num_steps = _rep(num_steps)
        inpaint_targets = _rep(inpaint_targets)
        inpaint_mask = _rep(inpaint_mask)
        inpaint_lengths = _rep(inpaint_lengths)

        batch_size = B_unique * num_candidates  # post-tile leading axis

        # Phase 3: noise and inpainting setup at batch_size (= B*N).
        # Per-request correlated-noise shrinkage: nearest entry of the precomputed grid.
        beta_grid, cholesky_grid, correction_grid = beta_tables
        cb = jnp.asarray(correlation_beta, jnp.float32)
        beta_idx = jnp.argmin(jnp.abs(cb[:, None] - beta_grid[None, :]), axis=1)  # [B*N]
        _, noise_rng = jax.random.split(rng)
        z = jax.random.normal(noise_rng, (batch_size, flat_dim))
        noise = jnp.einsum("bj,bij->bi", z, cholesky_grid[beta_idx]).reshape(
            batch_size, self.action_horizon, self.action_dim)

        # Noise temperature [B*N].
        noise_temp = jnp.asarray(noise_temperature)
        noise = noise * jnp.sqrt(noise_temp[:, None, None])

        noise_flat = noise.reshape(batch_size, flat_dim)
        targets_flat = inpaint_targets.reshape(batch_size, flat_dim)
        fixed_z = noise_flat * inpaint_mask
        x0 = targets_flat * inpaint_mask
        C_batch = correction_grid[beta_idx, inpaint_lengths]

        # Classifier-free guidance: the unconditional pass hides the advantage token from the suffix and drops the
        # advantage AdaRMS term (= a neutral sample in training).
        cfg_scale = jnp.asarray(cfg_scale)
        # prefix tokens the action expert attends to (positions still count every valid prefix token)
        prefix_keys = prefix_mask & ~vel_hidden
        # Unconditional mask: same as prefix_keys but advantage token = False
        prefix_mask_uncond = prefix_keys.at[:, adv_token_idx].set(False)

        tti = jnp.asarray(time_threshold_inpaint)  # [B*N]
        # Soft-pin fraction and trust-region cap: [B*N] -> [B*N, 1]
        pin_alpha = jnp.asarray(inpaint_pin_alpha, jnp.float32)[:, None]
        step_cap = jnp.asarray(inpaint_correction_step_cap, jnp.float32)[:, None]
        steps_b = jnp.asarray(num_steps, jnp.int32)  # [B*N]
        max_steps = jnp.max(steps_b)
        # warped time grid (per element): t_i = u_i^p, u_i = 1 - i / num_steps; p = 1 is the uniform grid
        warp = jnp.asarray(time_warp, jnp.float32)

        _V = self.config.wm_flow_vector_dim

        def step(carry):
            x_t, wm_v_c, i = carry
            n = steps_b.astype(jnp.float32)
            time = jnp.clip(1.0 - i / n, 0.0, 1.0) ** warp
            dt = jnp.clip(1.0 - (i + 1) / n, 0.0, 1.0) ** warp - time
            dt_x = dt[:, None, None]  # broadcast over [B, H, D]

            # Model forward pass (conditional).
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time,
                advantage_mask=adv_mask_inference,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(prefix_keys, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond])
            # actions: the first action_horizon suffix outputs (the WM-flow token is last)
            v_cond = self.action_out_proj(suffix_out[:, :self.action_horizon])

            # WM-flow prediction (conditional pass); float32 like the while_loop carry placeholder.
            # Value-vector residual [success, stage x5, togo x11] (raw, no activation).
            wm_raw_c = suffix_out[:, -1, :]
            wm_hid_c = self._time_fuse("aux_time_wm_flow", wm_raw_c, observation)
            wm_v_c_new = self.wm_flow_vector_head(wm_hid_c).astype(jnp.float32)

            # Classifier-free guidance: the unconditional pass
            prefix_attn_mask_uncond = einops.repeat(prefix_mask_uncond, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask_uncond = jnp.concatenate([prefix_attn_mask_uncond, suffix_attn_mask], axis=-1)

            adv_vec = self.advantage_adarms_vec.value.astype(adarms_cond.dtype)
            adv_sub = jnp.where(adv_mask_inference[:, None], adv_vec, jnp.zeros_like(adv_vec))
            adarms_cond_uncond = adarms_cond - adv_sub

            (_, suffix_out_uncond), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask_uncond,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond_uncond],
            )
            v_uncond = self.action_out_proj(suffix_out_uncond[:, :self.action_horizon])

            v_t = v_uncond + cfg_scale[:, None, None] * (v_cond - v_uncond)

            x_t_new = x_t + dt_x * v_t

            # Mask-based inpainting correction (vectorized, supports different inpaint lengths per element)
            time_new = time + dt
            tn = time_new[:, None]

            # Anchor path at the observed positions: x_t[O] = (1-t)*x0[O] + t*z[O]
            x_desired = (1.0 - tn) * x0 + tn * fixed_z  # [B, H * D]
            x_flat = x_t_new.reshape(batch_size, flat_dim)

            # Delta at observed positions (zero at free positions due to mask)
            delta = (x_desired - x_flat) * inpaint_mask  # [B, H * D]

            # Soft pin: move the observed coords `inpaint_pin_alpha` of the way to the anchor path each step
            # (1.0 = hard set, which carries the previous chunk's slow tail into the new chunk's head).
            applied = pin_alpha * delta
            x_flat = x_flat + applied

            # Correlated correction: propagate the APPLIED O delta to U positions
            # correction[b, i] = sum_j applied[b, j] * C_batch[b, j, i]
            correction = jnp.einsum('bj,bji->bi', applied, C_batch)  # [B, H * D]
            # Only at the unobserved positions
            correction = correction * (1.0 - inpaint_mask)

            # Trust region: scale (never skip) the correction so its largest per-step component stays within
            # step_cap (per element).
            max_corr = jnp.max(jnp.abs(correction), axis=-1, keepdims=True)  # [B, 1]
            scale = jnp.clip(step_cap / (max_corr + 1e-9), 0.0, 1.0)
            x_flat = x_flat + correction * scale

            x_corrected = x_flat.reshape(batch_size, self.action_horizon, self.action_dim)

            # Per-element threshold: only apply correction when time_new > tti[b]
            should_correct = (time_new > tti)  # [B]
            x_t_new = jnp.where(should_correct[:, None, None], x_corrected, x_t_new)

            # elements past their own step count stay as they are
            active = i < steps_b
            x_t_new = jnp.where(active[:, None, None], x_t_new, x_t)
            wm_v_c_new = jnp.where(active[:, None], wm_v_c_new, wm_v_c)
            return x_t_new, wm_v_c_new, i + 1

        def cond(carry):
            *_, i = carry
            return i < max_steps

        # Initial WM carry (conditional pass): a zero placeholder, overwritten by every step.
        init_carry = (noise, jnp.zeros((batch_size, _V), dtype=jnp.float32), jnp.int32(0))
        x_0, wm_flow_resid, _ = jax.lax.while_loop(cond, step, init_carry)

        # Prefix-head predictions were computed at B_unique and tiled to B * num_candidates. The WM-flow prediction
        # is from the FINAL loop iteration (the suffix at t = 1 / num_steps) of the conditional pass.
        preds["wm_flow_resid"] = wm_flow_resid  # [B*N, 17]: Q - V residual [dsucc, dstage x5, dtogo x11]

        return x_0, preds
