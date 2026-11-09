"""Initial weights of a fresh training run (TrainConfig.weight_loader; never used on resume).

PiModifiedWeightLoader(params_path) loads either
  * the openpi pi0.5 base checkpoint (pi05_roboeval's fresh-run init): the VLM, the SigLIP tower, the action expert
    and its in / out / time projections, with three adaptations: the SigLIP /14 patch stem is adjoint-resized to /16
    (native 256 px input), the action projections are cut from pi0.5's 32 action dims to ours (same-rank
    intersection), and the unused 257K-row Gemma vocab table becomes PiModified's 1-row dummy. Modules pi0.5 lacks
    keep the model's own init (identity camera projections, zero-init encodings and AdaRMS channels, random heads);
  * a PiModified checkpoint (an exported / saved `params` item): every param of the current tree must be present
    with its shape.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import dataclasses
import logging

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

from openpi.training.weight_loaders import WeightLoader

logger = logging.getLogger(__name__)

# Top-level modules of a PiModified that come from pi0.5 (everything else is PiModified-only).
PI05_MODULES = ("PaliGemma", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
# nnx.Intermediate state set from the norm stats after init (PiModified.load_correlation_matrix), never loaded.
INTERMEDIATES = ("action_correlation_cholesky", "action_loss_weight", "action_pts_scale")
GEMMA_EMBED_KEY = "PaliGemma/llm/embedder/input_embedding"


def _adapt_pi05(key: str, v: np.ndarray, ref_shape: tuple, ref_dtype) -> np.ndarray | None:
    """A pi0.5 param for our model, or None (keep the model's init) when it cannot be adapted."""
    if v.shape == ref_shape:
        return v.astype(ref_dtype) if v.dtype != ref_dtype else v
    if key == GEMMA_EMBED_KEY:
        # the 257K-row vocab table -> the 1-row dummy (state tokens use their own 256-row table)
        return np.zeros(ref_shape, dtype=ref_dtype)
    if key.endswith("img/embedding/kernel") and v.ndim == 4 and v.shape[2:] == tuple(ref_shape[2:]):
        # SigLIP stem /14 -> /16: adjoint resize, D^T w with D the antialiased 256->224 resize (models/flexi.py),
        # reproduces the pretrained 224-pipeline tokens on native 256 frames
        from roboeval_solution.models.flexi import pi_resize_patch_kernel

        logger.info(f"Adjoint-resized patch embedding '{key}': {v.shape} -> {tuple(ref_shape)}")
        return pi_resize_patch_kernel(v, tuple(ref_shape[:2])).astype(ref_dtype)
    if v.ndim == len(ref_shape) and v.ndim > 0:
        # same rank (pi0.5's 32 action dims vs ours): the intersection, zeros elsewhere
        slices = tuple(slice(0, min(s_l, s_r)) for s_l, s_r in zip(v.shape, ref_shape))
        merged = np.zeros(ref_shape, dtype=ref_dtype)
        merged[slices] = np.array(v[slices]).astype(ref_dtype)
        logger.info(f"'{key}': loaded {v.shape} into {ref_shape} (intersection, rest zero)")
        return merged
    logger.warning(f"Shape mismatch at '{key}': loaded {v.shape} vs model {ref_shape}: keeping the model's init")
    return None


@dataclasses.dataclass(frozen=True)
class PiModifiedWeightLoader(WeightLoader):
    """pi0.5 base or same-tree PiModified checkpoint -> the loaded subset of the param tree (see the module doc)."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        """`params`: the model's (abstract) pure-dict param tree. Returns the subset to load ("/"-path unflattened);
        params left out keep the model's init."""
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(download.maybe_download(self.params_path))
        flat_loaded = flax.traverse_util.flatten_dict(restored["params"])
        if all(kp[-1] == "value" for kp in flat_loaded if len(kp) >= 2):  # nnx.State layout (PiModified saves)
            flat_loaded = {kp[:-1]: v for kp, v in flat_loaded.items() if len(kp) >= 2}
        flat_loaded = {"/".join(map(str, k)): v for k, v in flat_loaded.items()}
        flat_ref = {k: v for k, v in flax.traverse_util.flatten_dict(params, sep="/").items()
                    if k.split("/")[0] not in INTERMEDIATES}

        if any(k.split("/")[0] not in PI05_MODULES for k in flat_loaded):
            # a PiModified checkpoint: the exact current tree
            missing = sorted(k for k in flat_ref if k not in flat_loaded)
            wrong = sorted(k for k in flat_ref if k in flat_loaded and tuple(np.shape(flat_loaded[k])) != tuple(flat_ref[k].shape))
            if missing or wrong:
                raise ValueError(f"{self.params_path} is not a checkpoint of this model: {len(missing)} params missing "
                                 f"(e.g. {missing[:5]}), {len(wrong)} with another shape (e.g. {wrong[:5]})")
            logger.info("Loading a PiModified checkpoint (%d params)", len(flat_ref))
            result = {k: np.asarray(flat_loaded[k]).astype(flat_ref[k].dtype) for k in flat_ref}
        else:
            logger.info("Loading the pi0.5 base checkpoint (PiModified-only modules keep their init)")
            result = {}
            for k, ref in flat_ref.items():
                if k.split("/")[0] not in PI05_MODULES:
                    continue
                if k not in flat_loaded:
                    raise ValueError(f"pi0.5 checkpoint {self.params_path} has no '{k}'")
                v = _adapt_pi05(k, np.asarray(flat_loaded[k]), tuple(ref.shape), ref.dtype)
                if v is not None:
                    result[k] = v
            fresh = sorted({k.split("/")[0] for k in flat_ref if k not in result})
            logger.info("Modules at their fresh init: %s", fresh)
        return flax.traverse_util.unflatten_dict(result, sep="/")
