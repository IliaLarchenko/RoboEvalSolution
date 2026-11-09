"""Training data loading: RoboEval rbe stores (data/dataset.py) -> transforms -> batched torch loader.

Protocol definitions, TransformedDataset and TorchDataLoader are adapted from openpi.training.data_loader
(not imported directly: it pulls in an old lerobot API).
"""

import logging
import multiprocessing
import os
import queue
import threading
import time
import typing
from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar

import jax
import numpy as np
import torch

import openpi.transforms as _transforms

from roboeval_solution.models.observation import Observation
from roboeval_solution.transforms_normalize import Normalize

T_co = TypeVar("T_co", covariant=True)



# ── Protocol definitions (copied from openpi.training.data_loader) ──────────

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""
    def __getitem__(self, index: SupportsIndex) -> T_co: ...
    def __len__(self) -> int: ...


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""
    def data_config(self): ...
    def __iter__(self) -> Iterator[T_co]: ...


# ── Utility classes (copied from openpi.training.data_loader) ───────────────

class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)




def _collate_fn(items):
    """Collate batch elements into batched numpy arrays. A None field is a data bug: raise (a NaN fill would reach
    the losses, e.g. a NaN advantage reads as positive)."""
    for i, item in enumerate(items):
        bad = [key for key, value in item.items() if value is None]
        if bad:
            raise ValueError(f"None in fields {bad} of batch item {i}")
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate GPU memory."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class TorchDataLoader:
    """Torch data loader adapted from openpi: weighted sampling, background prefetch, close()."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding,
        num_workers: int = 0,
        seed: int = 0,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) > dataset size ({len(dataset)}).")

        self._sharding = sharding
        self._stop = threading.Event()  # set by close(): ends the producer thread and the torch iterator
        self._thread: threading.Thread | None = None

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        # Items are drawn with replacement from the dataset's per-item sampling weights (RoboEvalDataset).
        weights = getattr(dataset, "_dataset", dataset).weights
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.float64), num_samples=len(weights),
            replacement=True, generator=generator)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def _raw_iter(self):
        """Iterate over raw batches (no prefetch)."""
        while not self._stop.is_set():
            data_iter = iter(self._data_loader)
            while not self._stop.is_set():
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
                )

    def __iter__(self):
        buf: queue.Queue = queue.Queue(maxsize=2)
        _DONE = object()

        def _put(item) -> bool:
            # a bounded put that gives up once close() was called (an abandoned consumer never drains buf)
            while not self._stop.is_set():
                try:
                    buf.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def _producer():
            try:
                for batch in self._raw_iter():
                    if not _put(batch):
                        return
            except Exception as e:
                _put(e)
            finally:
                _put(_DONE)

        thread = threading.Thread(target=_producer, daemon=True)
        self._thread = thread
        thread.start()
        while True:
            item = buf.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self, timeout: float = 30.0) -> None:
        """Stop the producer thread and shut the (persistent) worker processes down. Required when a loader
        is replaced (DataRefresher swap): otherwise the producer blocks on a full queue forever and keeps
        the loader, its torch iterator and every worker (each holding the dataset arrays) alive."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        it = getattr(self._data_loader, "_iterator", None)  # persistent_workers keeps the worker iterator here
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
        self._data_loader._iterator = None


# ── Data loading ────────────────────────────────────────────────────

class DataLoaderImpl(DataLoader):
    """Yields (Observation, actions) batches."""

    def __init__(self, data_config, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self):
        return self._data_config

    def close(self) -> None:
        self._data_loader.close()

    def __iter__(self):
        for batch in self._data_loader:
            obs = Observation.from_dict(batch)
            actions = batch["actions"]
            yield obs, actions




def create_dataset(data_config, action_horizon: int) -> Dataset:
    """The RoboEval store dataset of a RoboEvalDataConfig (sources, sampling weights, value targets)."""
    if getattr(data_config, "roboeval", None) is None:
        raise ValueError("data config has no RoboEval store dataset (RoboEvalDataConfig)")
    return data_config.roboeval.make_dataset(action_horizon)


def transform_dataset(dataset: Dataset, data_config) -> Dataset:
    """Transform the dataset: repack, data transforms, normalisation (transforms_normalize.Normalize), model
    transforms."""
    if data_config.norm_stats is None:
        raise ValueError(
            "Normalization stats not found. "
            "Run `scripts/prepare_assets.py --config-name <config>`."
        )
    norm_stats = data_config.norm_stats

    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        Normalize(norm_stats),
        *data_config.model_transforms.inputs,
    ]

    return TransformedDataset(dataset, transforms_list)


def create_data_loader(
    config,
    *,
    sharding,
) -> DataLoader:
    """Create the training data loader."""
    data_config = config.data.create(config.assets_dirs, config.model)

    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")

    dataset = create_dataset(data_config, action_horizon=config.model.action_horizon)
    dataset = transform_dataset(dataset, data_config)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        num_workers=config.num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)
