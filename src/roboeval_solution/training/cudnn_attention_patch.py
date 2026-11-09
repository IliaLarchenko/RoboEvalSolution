"""cuDNN flash attention for the long VLM prefix pass (training and inference).

The stock gemma.Attention materialises [B, 8, T, S] float32 scores; for the ~1300-token prefix
(5 cameras x 256 image tokens) that is the largest non-GEMM cost of a forward pass. Here the prefix
pass (no KV cache, long sequence) goes through cuDNN's fused flash attention with the same boolean
mask; the action-expert passes (suffix tokens over the cached prefix) keep the original math.

JAX lets cuDNN take head_dim 256 only on Hopper and later GPUs. install() verifies the kernel against the
reference math (forward and, for training, backward) and keeps the stock attention if it does not match.

Same parameters / names as gemma.Attention, so checkpoints load unchanged. Install before the
model is constructed:

    from roboeval_solution.training import cudnn_attention_patch
    cudnn_attention_patch.install()
"""
from __future__ import annotations

import logging

import flax.linen as nn
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

_INSTALLED = False
MIN_PREFIX_LEN = 256  # shorter (suffix / tiny) passes keep the original math


def _reference(q, k, v, attn_mask, head_dim, num_kv_heads, dtype):
    """openpi gemma.Attention math (q not yet scaled)."""
    import einops

    q = q * head_dim ** -0.5
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
    masked = jnp.where(attn_mask[:, :, None, :, :], logits, -2.3819763e38)
    probs = jax.nn.softmax(masked, axis=-1).astype(dtype)
    enc = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
    return einops.rearrange(enc, "B T K G H -> B T (K G) H")


def cudnn_attention(q, k, v, attn_mask, head_dim):
    """cuDNN fused attention, exact w.r.t. the reference also for query rows with no allowed key
    (padded FAST / WM tokens, dropped state or velocity tokens): the reference softmax over the equal
    -2.38e38 fill is uniform over all keys, i.e. mean(v). cuDNN returns NaN gradients for such rows,
    so they are fully opened (finite) and their output is replaced by mean(v).
    q [B, T, N, H], k / v [B, S, K, H], attn_mask [B, 1, T, S] bool."""
    row_any = jnp.any(attn_mask, axis=-1, keepdims=True)  # [B, 1, T, 1]
    safe_mask = attn_mask | ~row_any
    enc = jax.nn.dot_product_attention(q, k, v, mask=safe_mask, scale=head_dim ** -0.5, implementation="cudnn")
    n_rep = q.shape[2] // k.shape[2]
    v_mean = jnp.repeat(jnp.mean(v.astype(jnp.float32), axis=1), n_rep, axis=1).astype(enc.dtype)  # [B, N, H]
    empty = ~row_any[:, 0, :, :, None]  # [B, T, 1, 1]
    return jnp.where(empty, v_mean[:, None], enc)


def _test_mask(B, T):
    """Prefix-like mask: images bidirectional, then causal, plus fully-masked rows / columns (padding)."""
    grp = jnp.where(jnp.arange(T) < 1281, 0, jnp.arange(T) - 1280)
    mask = (grp[None, :] <= grp[:, None])
    valid = jnp.arange(T) < T - 7  # last 7 tokens padded: their rows and columns are empty
    mask = mask & valid[None, :] & valid[:, None]
    mask = mask.at[1290].set(False)  # one more empty row in the middle (dropped token)
    return mask[None, None].repeat(B, 0)


def _self_test() -> float:
    """Max abs difference of cuDNN vs reference on a prefix-shaped problem with a block mask."""
    B, T, N, K, H = 2, 1300, 8, 1, 256
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, T, N, H)).astype(jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, T, K, H)).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, T, K, H)).astype(jnp.bfloat16)
    mask = _test_mask(B, T)
    ref = jax.jit(lambda q, k, v: _reference(q, k, v, mask, H, K, jnp.bfloat16))(q, k, v)
    out = jax.jit(lambda q, k, v: cudnn_attention(q, k, v, mask, H))(q, k, v)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    return err if err == err else float("inf")  # NaN -> inf


def _self_test_grad() -> float:
    """Max relative error of d(sum(out * w))/d(q, k, v), cuDNN vs reference (training backward pass)."""
    B, T, N, K, H = 2, 1300, 8, 1, 256
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (B, T, N, H)).astype(jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (B, T, K, H)).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), (B, T, K, H)).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.fold_in(key, 3), (B, T, N, H)).astype(jnp.float32)
    mask = _test_mask(B, T)

    def loss(f):
        return lambda q, k, v: jnp.sum(f(q, k, v).astype(jnp.float32) * w)

    ref = jax.jit(jax.grad(loss(lambda q, k, v: _reference(q, k, v, mask, H, K, jnp.bfloat16)), argnums=(0, 1, 2)))
    cud = jax.jit(jax.grad(loss(lambda q, k, v: cudnn_attention(q, k, v, mask, H)), argnums=(0, 1, 2)))
    errs = []
    for a, b in zip(ref(q, k, v), cud(q, k, v)):
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        e = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-6))
        errs.append(e if e == e else float("inf"))  # NaN -> inf
    return max(errs)


def install(training: bool = False) -> bool:
    """Install the patch; returns False (and leaves gemma untouched) if cuDNN can't run it exactly.
    ``training=True`` also checks the backward pass (training through the cuDNN prefix attention)."""
    if _INSTALLED == "cudnn":
        return True
    try:
        err = _self_test()
        gerr = _self_test_grad() if training else 0.0
    except Exception as e:  # unsupported GPU / cuDNN
        logger.warning("cuDNN attention unavailable (%s); keeping the stock attention", e)
        return False
    if not err < 0.05:  # bf16 outputs of O(1): a broken kernel is off by O(1)
        logger.warning("cuDNN attention self-test max err %.3g; keeping the stock attention", err)
        return False
    if not gerr < 0.05:
        logger.warning("cuDNN attention gradient self-test max rel err %.3g; keeping the stock attention", gerr)
        return False
    logger.info("cuDNN attention self-test ok (max abs err %.2g, grad rel err %.2g vs reference)", err, gerr)
    _install_class()
    logger.info("Installed cuDNN prefix attention (replaced gemma.Attention)")
    return True


def _install_class() -> None:
    """Replace gemma.Attention (same parameters / names)."""
    global _INSTALLED
    from openpi.models import gemma as _gemma
    from openpi.models.gemma import _apply_rope, _name, lora

    class PatchedAttention(_gemma.Attention):
        @nn.compact
        def __call__(self, xs, positions, attn_mask, kv_cache):
            assert all(c.head_dim == self.configs[0].head_dim for c in self.configs)
            assert all(c.num_heads == self.configs[0].num_heads for c in self.configs)
            assert all(c.num_kv_heads == self.configs[0].num_kv_heads for c in self.configs)
            dtype = next(x.dtype for x in xs if x is not None)

            qkvs = []
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is None:
                    continue
                if config.num_kv_heads == config.num_heads:
                    qkv_einsum = lora.Einsum(
                        shape=(3, config.num_heads, config.width, config.head_dim),
                        name=_name("qkv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
                else:
                    q_einsum = lora.Einsum(
                        shape=(config.num_heads, config.width, config.head_dim),
                        name=_name("q_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    q = q_einsum("BTD,NDH->BTNH", x)
                    kv_einsum = lora.Einsum(
                        shape=(2, config.num_kv_heads, config.width, config.head_dim),
                        name=_name("kv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                    qkvs.append((q, k, v))

            q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
            q = _apply_rope(q, positions=positions)
            k = _apply_rope(k, positions=positions)
            assert q.dtype == k.dtype == v.dtype == dtype

            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)

            if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
                raise ValueError(f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: "
                                 f"{q.shape} and {k.shape}")
            head_dim, n_kv = self.configs[0].head_dim, self.configs[0].num_kv_heads
            if kv_cache is None and q.shape[1] >= MIN_PREFIX_LEN:
                encoded = cudnn_attention(q, k, v, attn_mask, head_dim)
            else:
                encoded = _reference(q, k, v, attn_mask, head_dim, n_kv, dtype)

            out = []
            start = 0
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is not None:
                    end = start + x.shape[1]
                    out_einsum = lora.Einsum(
                        shape=(config.num_heads, config.head_dim, config.width),
                        name=_name("attn_vec_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                        lora_config=config.lora_configs.get("attn"),
                    )
                    out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                    start = end
                else:
                    out.append(None)
            return out, (k, v)

    _gemma.Attention = PatchedAttention
    _INSTALLED = "cudnn"
