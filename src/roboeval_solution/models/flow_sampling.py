"""Variance-reduced (t, noise) draws for the K flow-matching samples per observation (training only).

Unbiased: every t is marginally Beta(1.5, 1) (scaled to [0.001, 1]) and every noise draw is marginally N(0, Sigma), as
with an i.i.d. sampler. Only the joint distribution of the K draws changes (t stratified over its CDF, antithetic
+-noise pairs), which lowers the variance of the K-sample loss and gradient.
"""
import jax
import jax.numpy as jnp

_BETA_A = 1.5  # Beta(1.5, 1): CDF t**1.5, inverse u**(2/3)


def _scale(t):
    return t * 0.999 + 0.001


def _stratified_t(rng, n_bins, batch_size):
    """[n_bins, B]: one t per CDF-quantile bin (bin i covers u in [i/n, (i+1)/n))."""
    u = (jnp.arange(n_bins, dtype=jnp.float32)[:, None] + jax.random.uniform(rng, (n_bins, batch_size))) / n_bins
    return _scale(u ** (1.0 / _BETA_A))


def sample_t_and_noise(rng, num_samples, batch_size, noise_fn):
    """-> (t [K, B], noise [K, B, H, D]): stratified t, antithetic noise pairs. `noise_fn(rng, batch_size)` draws one
    [B, H, D] noise batch (the model's correlated-noise generator)."""
    t_rng, n_rng = jax.random.split(rng)
    K = num_samples
    if K % 2:
        raise ValueError(f"antithetic flow sampling needs an even num_flow_samples, got {K}")
    half = jax.vmap(lambda r: noise_fn(r, batch_size))(jax.random.split(n_rng, K // 2))  # [K/2, B, H, D]
    # pair j -> samples (2j, 2j+1) with +noise_j / -noise_j
    noise = jnp.stack([half, -half], axis=1).reshape((K,) + half.shape[1:])
    t = _stratified_t(t_rng, K, batch_size)  # adjacent bins share the +- pair
    return t, noise
