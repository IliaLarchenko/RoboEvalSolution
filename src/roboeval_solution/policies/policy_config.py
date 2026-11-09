"""Build a served policy from a trained checkpoint / export (params + assets)."""

import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.download as download

from roboeval_solution import transforms as rbe_transforms
from roboeval_solution.policies.pi_modified_policy import PiModifiedPolicy
from roboeval_solution.training import checkpoints as _checkpoints
from roboeval_solution.training import config as _config
from roboeval_solution.transforms_normalize import Normalize, Unnormalize


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    sample_kwargs: dict[str, Any] | None = None,
) -> PiModifiedPolicy:
    """PiModified policy from `<checkpoint_dir>/params` (+ `assets/` norm stats) with the config's transforms."""
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    loaded_params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)  # the server runs bf16
    model = train_config.model.load(loaded_params)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    if norm_stats is None:
        raise ValueError(f"no norm stats in {checkpoint_dir}/assets")
    model.load_correlation_matrix(norm_stats)
    logging.info("Loaded correlation matrix for inference")

    # FAST tokenisation is a training-only target
    model_inputs = [t for t in data_config.model_transforms.inputs if not isinstance(t, rbe_transforms.TokenizeFASTActions)]
    # (data_config.repack_transforms is skipped: it maps the training 'actions' column)
    input_transforms = [
        *data_config.data_transforms.inputs,
        Normalize(norm_stats),  # as in training (training/data_loader.py)
        *model_inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        Unnormalize(norm_stats),
        *data_config.data_transforms.outputs,
    ]
    return PiModifiedPolicy(model, transforms=input_transforms, output_transforms=output_transforms,
                            sample_kwargs=sample_kwargs)
