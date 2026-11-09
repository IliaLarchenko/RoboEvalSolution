"""TorchDataLoader.close() stops the producer thread and the persistent workers."""
import time

import jax
import numpy as np

from roboeval_solution.training.data_loader import TorchDataLoader


class _Toy:
    weights = np.full(64, 1 / 64)

    def __len__(self):
        return 64

    def __getitem__(self, i):
        return {"x": np.full((3,), i, np.float32)}


def test_close_stops_producer_and_workers():
    sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B"))
    loader = TorchDataLoader(_Toy(), 4, sharding=sharding, num_workers=2)
    it = iter(loader)
    next(it)
    time.sleep(1.0)  # producer fills the queue and blocks on put
    workers = list(loader._data_loader._iterator._workers)
    assert all(w.is_alive() for w in workers)
    loader.close(timeout=10)
    assert not loader._thread.is_alive()
    assert not any(w.is_alive() for w in workers)
