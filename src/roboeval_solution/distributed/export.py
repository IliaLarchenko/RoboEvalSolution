"""Export a training checkpoint step as a servable policy dir (params in bf16 + assets).

The server restores params as bf16 anyway (policy_config.create_trained_policy), so exporting in
bf16 halves the size moved through the bucket.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def export_policy(step_dir: str | Path, out_dir: str | Path) -> Path:
    """step_dir = <checkpoint_dir>/<step> (orbax: params/, assets/); writes out_dir atomically."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never touch the training GPUs
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp

    from openpi.models import model as _model

    step_dir, out_dir = Path(step_dir), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    params = _model.restore_params(step_dir / "params", restore_type=np.ndarray, dtype=jnp.dtype("bfloat16"))
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save((tmp / "params").resolve(), {"params": params})
    shutil.copytree(step_dir / "assets", tmp / "assets")
    (tmp / "EXPORT_INFO").write_text(f"source={step_dir.resolve()}\ndtype=bfloat16\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return out_dir
