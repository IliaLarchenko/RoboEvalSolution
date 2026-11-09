"""Patch-embedding resize for the SigLIP stem: pi0.5's So400m/14 (224x224 -> 16x16 tokens) consumes native 256x256
images as So400m/16 (same 16x16 token grid, same position embeddings, same compute).

The adjoint resize w_new = D^T w_old, D the antialiased new->old downsampling (what the 224 preprocessing did),
reproduces "downsample then old stem" per patch, so native 256 frames give the pretrained tokens (closer than the
FlexiViT pseudo-inverse resize of Beyer et al., CVPR 2023); the trainable stem then learns to use the extra detail.
"""
from __future__ import annotations

import numpy as np


def _resize_matrix(old_hw: tuple[int, int], new_hw: tuple[int, int]) -> np.ndarray:
    """[new_h*new_w, old_h*old_w] matrix of antialiased bilinear resizing (jax.image.resize)."""
    import jax
    import jax.numpy as jnp

    n_old = old_hw[0] * old_hw[1]
    basis = jnp.eye(n_old, dtype=jnp.float32).reshape(n_old, *old_hw)
    resized = jax.vmap(lambda b: jax.image.resize(b, new_hw, method="bilinear", antialias=True))(basis)
    return np.asarray(resized).reshape(n_old, -1).T


def pi_resize_patch_kernel(kernel: np.ndarray, new_hw: tuple[int, int]) -> np.ndarray:
    """Adjoint-resize a conv patch-embedding kernel [h, w, c_in, c_out] to [new_h, new_w, c_in, c_out]."""
    kernel = np.asarray(kernel, np.float32)
    old_hw = kernel.shape[:2]
    if tuple(old_hw) == tuple(new_hw):
        return kernel
    h, w, ci, co = kernel.shape
    down = _resize_matrix(tuple(new_hw), old_hw)  # [n_old, n_new]
    return (down.T @ kernel.reshape(h * w, ci * co)).reshape(*new_hw, ci, co)
