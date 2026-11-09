"""Normalization statistics (norm_stats.json, written by scripts/prepare_assets.py)."""

import json
import pathlib

import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile (state: lower end of the [-1, 1] map)
    q99: numpydantic.NDArray | None = None  # 99th quantile (state: upper end)

    # Actions: per-timestep std [action_horizon, action_dim] (the flow loss weights and the noise scale,
    # PiModified.load_action_loss_weight / _timestep_noise_scale) and the Cholesky factor of the chunk correlation
    # matrix [H * D, H * D] (correlated flow noise).
    per_timestamp_std: numpydantic.NDArray | None = None
    action_correlation_cholesky: numpydantic.NDArray | None = None


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
