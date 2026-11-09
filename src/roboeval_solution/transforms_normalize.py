"""Normalize / Unnormalize transforms (training data loader and the policy server).

State (1-D) is quantile-mapped to [-1, 1] with its q01 / q99 (the envelope written by scripts/prepare_assets.py).
Actions (2-D chunks) are z-scored with one global mean / std per dimension; the per-timestep scale lives in the
model (the flow loss weighted by (global std / per-timestep std)^2 and the flow noise scaled to the per-timestep std,
pi_modified.py).
"""

import dataclasses

from openpi.transforms import DataTransformFn, DataDict, apply_tree, pad_to_dim
from openpi.shared import array_typing as at
from roboeval_solution.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats]

    def __call__(self, data: DataDict) -> DataDict:
        return apply_tree(data, self.norm_stats, self._normalize, strict=False)

    @staticmethod
    def _normalize(x, stats: NormStats):
        if x.ndim < 2:  # state
            q01 = stats.q01[..., : x.shape[-1]]
            q99 = stats.q99[..., : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return (x - mean) / (std + 1e-6)


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats]

    def __call__(self, data: DataDict) -> DataDict:
        return apply_tree(data, self.norm_stats, self._unnormalize, strict=True)

    @staticmethod
    def _unnormalize(x, stats: NormStats):
        if x.ndim < 2:  # state
            q01 = stats.q01[..., : x.shape[-1]]
            q99 = stats.q99[..., : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean
