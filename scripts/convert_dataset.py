"""Convert the released LeRobot v3 dataset (image features in parquet) into rbe stores.

One store per task variation:  <out>/<VariationName>/  (PNG bytes copied, no re-encode).

    python scripts/convert_dataset.py --src data/roboeval_ee_delta_20hz_scaled --out data/stores/demos --workers 8

Runs in either venv (needs numpy, pandas, pyarrow).
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.data.store import StoreReader, StoreWriter  # noqa: E402
from roboeval_solution.data.tasks import split_variation_name  # noqa: E402

CAM_COLUMNS = {"head": "cam_head", "front": "cam_front", "external": "cam_external",
               "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}  # in the stores' camera order


def convert_one(args) -> str:
    src_dir, out_dir = args
    cameras = list(CAM_COLUMNS)
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    name = src_dir.name
    family, variation = split_variation_name(name)
    if (out_dir / "meta.json").exists():
        r = StoreReader(out_dir, load_arrays=False)
        return f"{name}: exists ({r.meta['num_episodes']} eps, {r.num_frames} frames)"
    t0 = time.time()
    files = sorted(glob.glob(str(src_dir / "data" / "chunk-*" / "*.parquet")))
    ep_meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(src_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))])
    expected_len = dict(zip(ep_meta["episode_index"].astype(int), ep_meta["length"].astype(int)))
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    writer = StoreWriter(tmp_dir, cameras=cameras, fps=20,
                         info={"source": "roboeval_ee_delta_20hz_scaled", "variation_name": name,
                               "family": family, "variation": variation})
    cols = ["episode_index", "frame_index", "state", "state_gripper", "actions", "reward", "uuid"]
    cols += [CAM_COLUMNS[c] for c in cameras]
    pending = None  # frames of the episode currently being assembled

    def emit(df: pd.DataFrame) -> None:
        df = df.sort_values("frame_index")
        ep = int(df.episode_index.iloc[0])
        n = len(df)
        if expected_len.get(ep, n) != n:
            raise RuntimeError(f"{name} ep {ep}: {n} frames, meta says {expected_len.get(ep)}")
        if not np.array_equal(df.frame_index.values, np.arange(n)):
            raise RuntimeError(f"{name} ep {ep}: non-contiguous frame_index")
        arrays = {
            "state": np.stack(df.state.values).astype(np.float32),
            "state_gripper": np.stack(df.state_gripper.values).astype(np.float32),
            "actions": np.stack(df.actions.values).astype(np.float32),
            "reward": np.asarray([np.asarray(r).reshape(-1)[0] for r in df.reward.values], np.float32),
        }
        images = {c: [d["bytes"] for d in df[CAM_COLUMNS[c]].values] for c in cameras}
        uuid = df.uuid.iloc[0]
        uuid = uuid[0] if isinstance(uuid, (list, np.ndarray)) else uuid
        writer.add_episode(arrays, images, {
            "family": family, "variation": variation, "success": True, "source": "demo",
            "uuid": str(uuid), "src_episode_index": ep,
        })

    for f in files:
        df = pd.read_parquet(f, columns=cols)
        for ep, g in df.groupby("episode_index", sort=True):
            if pending is not None and int(pending.episode_index.iloc[0]) != int(ep):
                emit(pending)
                pending = None
            pending = g if pending is None else pd.concat([pending, g])
    if pending is not None:
        emit(pending)
    writer.close()
    if writer.num_episodes != len(expected_len):
        raise RuntimeError(f"{name}: wrote {writer.num_episodes} episodes, meta has {len(expected_len)}")
    tmp_dir.rename(out_dir)
    return f"{name}: {writer.num_episodes} eps, {writer.num_frames} frames in {time.time() - t0:.0f}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    src = Path(a.src)
    dirs = sorted(d for d in src.iterdir() if (d / "meta" / "info.json").exists())
    jobs = [(str(d), str(Path(a.out) / d.name)) for d in dirs]
    print(f"converting {len(jobs)} variations with {a.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(convert_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
