"""Per-loss weight composition. Auxiliary losses are weighted by the per-sample `is_weight` = 1 / (N p_i), which
unbiases the weighted sampling of data/dataset.py, and balanced over the tasks of the batch."""

from __future__ import annotations

import jax.numpy as jnp

from roboeval_solution.constants import NUM_TASKS


def type_balanced_weighted_mean(loss, weights, valid, task_id):
    """Weighted mean within each task, then unweighted mean across tasks.

    For each task t present in the batch (task_id in [0, NUM_TASKS)):
        ℓ_t = Σ_{i∈t, valid} w_i · loss_i / Σ_{i∈t, valid} w_i
    Final loss = mean_{t present} ℓ_t. NaN-safe: ``loss`` is zeroed at invalid positions before the weighted sum.
    """
    safe_loss = jnp.where(valid, loss, 0.0)
    type_losses = []
    type_present = []
    for t in range(NUM_TASKS):
        mask_t = (task_id == t) & valid
        w_t = jnp.where(mask_t, weights, 0.0)
        w_t_sum = jnp.maximum(w_t.sum(), 1e-8)
        type_loss = (safe_loss * w_t).sum() / w_t_sum
        type_losses.append(type_loss)
        type_present.append(mask_t.any())

    type_stack = jnp.stack(type_losses)
    present = jnp.stack(type_present)
    return jnp.where(present, type_stack, 0.0).sum() / jnp.maximum(
        present.sum().astype(jnp.float32), 1.0
    )
