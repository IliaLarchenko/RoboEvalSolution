"""Data transforms on top of OpenPI's: TokenizeFASTActions, AdvantageToWeight, ResizeImagesIfNeeded."""

import dataclasses
from pathlib import Path

import numpy as np

from openpi.transforms import (
    DataTransformFn,
    DataDict,
)

from roboeval_solution.shared.normalize import NormStats

# FAST tokenizers by tokenizer_path (module level: the transforms are frozen dataclasses).
_TOKENIZER_CACHE: dict[str, object] = {}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTActions(DataTransformFn):
    """FAST tokens of the action chunk: the PiModified auxiliary FAST loss target (training only).

    The globally z-scored chunk is mapped back to raw actions and re-normalised with the global q01 / q99 to [-1, 1]
    (the range the tokenizer was fit on, scripts/prepare_assets.py), then tokenised, truncated / padded to
    ``max_fast_tokens``. Creates ``fast_tokens`` [max_fast_tokens] and ``fast_token_mask``.
    """

    # FAST tokenizer directory (<assets>/<asset_id>/fast_tokenizer)
    tokenizer_path: str
    # Action dimensions to encode: [(start, end), ...]
    encoded_dim_ranges: list[tuple[int, int]]
    # Normalization stats (global mean / std to undo, q01 / q99 to map to [-1, 1])
    norm_stats: dict[str, NormStats]
    max_fast_tokens: int = 180

    def _get_tokenizer(self):
        """Lazy load tokenizer (called in each worker process)."""
        if self.tokenizer_path in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[self.tokenizer_path]

        import importlib.util

        tokenizer_dir = Path(self.tokenizer_path)
        processor_file = tokenizer_dir / "processing_action_tokenizer.py"

        if not processor_file.exists():
            raise FileNotFoundError(f"Tokenizer processing file not found at {processor_file}")

        spec = importlib.util.spec_from_file_location("processing_action_tokenizer", processor_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        UniversalActionProcessor = module.UniversalActionProcessor
        tokenizer = UniversalActionProcessor.from_pretrained(str(tokenizer_dir))

        _TOKENIZER_CACHE[self.tokenizer_path] = tokenizer
        return tokenizer
    
    def __call__(self, data: DataDict) -> DataDict:
        actions = data["actions"]  # [H, D], globally z-scored
        
        encoded_chunks = []
        for start, end in self.encoded_dim_ranges:
            encoded_chunks.append(actions[:, start:end])
        encoded_actions = np.concatenate(encoded_chunks, axis=-1)  # [H, D_encoded]

        if np.isnan(encoded_actions).any():
            raise ValueError("NaN in the action chunk passed to TokenizeFASTActions")

        # Global z-score -> raw actions -> global quantile range [-1, 1] (clipped to [q01, q99]), as for the fit
        action_stats = self.norm_stats['actions']
        encoded_dims = np.array([d for start, end in self.encoded_dim_ranges for d in range(start, end)])
        raw = encoded_actions * (action_stats.std[encoded_dims] + 1e-6) + action_stats.mean[encoded_dims]
        q01 = action_stats.q01[encoded_dims]
        q99 = action_stats.q99[encoded_dims]
        raw = np.clip(raw, q01, q99)
        q_range = q99 - q01
        # a near-constant dimension maps to 0 instead of huge values
        encoded_actions = np.where(q_range > 1e-4, 2.0 * (raw - q01) / np.maximum(q_range, 1e-4) - 1.0, 0.0)

        tokenizer = self._get_tokenizer()
        tokens = tokenizer(encoded_actions[None])[0]
        
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        
        # Truncate or pad to max_fast_tokens
        if len(tokens) > self.max_fast_tokens:
            tokens = tokens[:self.max_fast_tokens]
            mask = np.ones(self.max_fast_tokens, dtype=bool)
        else:
            mask = np.concatenate([
                np.ones(len(tokens), dtype=bool),
                np.zeros(self.max_fast_tokens - len(tokens), dtype=bool)
            ])
            tokens = np.pad(tokens, (0, self.max_fast_tokens - len(tokens)), constant_values=0)
        
        data["fast_tokens"] = tokens.astype(np.int32)
        data["fast_token_mask"] = mask

        return data


@dataclasses.dataclass(frozen=True)
class AdvantageToWeight:
    """Clips the ``advantage`` key to [clip_min, clip_max] (the advantage-token conditioning input). The sampling
    priority exp(clipped advantage) is applied by the dataset (StoreSource.prioritized), not here."""

    clip_min: float = -2.0
    clip_max: float = 2.0

    def __call__(self, data: DataDict) -> DataDict:
        adv_raw = data.get("advantage")
        if adv_raw is None:
            return data

        data["advantage"] = np.clip(
            np.asarray(adv_raw, dtype=np.float32), self.clip_min, self.clip_max
        ).astype(np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class ResizeImagesIfNeeded(DataTransformFn):
    """openpi ResizeImages, skipped for images already at the target size (native 256 renders)."""

    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        from openpi.shared import image_tools

        data["image"] = {
            k: v if tuple(np.shape(v)[-3:-1]) == (self.height, self.width)
            else image_tools.resize_with_pad(v, self.height, self.width)
            for k, v in data["image"].items()
        }
        return data
