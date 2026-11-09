"""Flat episode store: numpy arrays + append-only encoded-image blobs.

Layout of one store directory::

    meta.json                 format, fps, cameras, num_frames, array specs, free-form info
    episodes.json             one dict per episode: start, length, family, variation, success, ...
    arrays/<name>.npy         per-frame arrays, first axis = frame (state, actions, episode_index, ...)
    images/<cam>.bin          concatenated encoded images (PNG/JPEG), frame order
    images/<cam>.offsets.npy  int64 [num_frames + 1] byte offsets into <cam>.bin

Random access to any frame's image is one memmap slice + one decode, so the
training loader needs no video decoding or parquet row-group reads. The writer
is used both for converting the released dataset and by rollout workers; its
``flush()`` makes everything written so far readable (arrays/meta are replaced
atomically, image blobs are append-only).

Only numpy (+ optional cv2 for decoding) is imported so both the training venv
and the simulator venv can use this module.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

FORMAT = "rbe_store_v1"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


class StoreWriter:
    """Append episodes to a store (an existing one in `root` is overwritten). Not process-safe: one writer per
    directory."""

    def __init__(self, root: str | Path, cameras: Sequence[str], fps: int = 20,
                 info: Optional[Dict[str, Any]] = None):
        self.root = Path(root)
        (self.root / "arrays").mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.cameras = list(cameras)
        self.fps = fps
        self.info = dict(info or {})
        self._arrays: Dict[str, List[np.ndarray]] = {}
        self._episodes: List[Dict[str, Any]] = []
        self._num_frames = 0
        self._bins = {c: open(self.root / "images" / f"{c}.bin", "wb") for c in self.cameras}
        self._offsets = {c: [0] for c in self.cameras}

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_episode(self, arrays: Dict[str, np.ndarray], images: Dict[str, Sequence[bytes]],
                    episode_info: Dict[str, Any]) -> int:
        """Add one episode. ``arrays[name]`` has the frame axis first; ``images[cam]`` is a list
        of encoded images (one per frame). Returns the episode index within this store."""
        lengths = {k: len(v) for k, v in arrays.items()} | {c: len(images[c]) for c in self.cameras}
        n = next(iter(lengths.values()))
        if any(v != n for v in lengths.values()):
            raise ValueError(f"inconsistent episode lengths: {lengths}")
        if self._episodes and set(arrays) | {"episode_index", "frame_index"} != set(self._arrays):
            raise ValueError(f"array keys changed: {sorted(arrays)} vs {sorted(self._arrays)}")
        ep_idx = len(self._episodes)
        for k, v in arrays.items():
            self._arrays.setdefault(k, []).append(np.asarray(v))
        self._arrays.setdefault("episode_index", []).append(np.full(n, ep_idx, np.int32))
        self._arrays.setdefault("frame_index", []).append(np.arange(n, dtype=np.int32))
        for c in self.cameras:
            f, offs = self._bins[c], self._offsets[c]
            for b in images[c]:
                f.write(b)
                offs.append(offs[-1] + len(b))
        info = dict(episode_info)
        info.update(episode_index=ep_idx, start=self._num_frames, length=n)
        self._episodes.append(info)
        self._num_frames += n
        return ep_idx

    def flush(self) -> None:
        """Make all episodes added so far visible to readers."""
        for f in self._bins.values():
            f.flush()
        for c in self.cameras:
            _atomic_save_npy(self.root / "images" / f"{c}.offsets.npy", np.asarray(self._offsets[c], np.int64))
        specs = {}
        for k, parts in self._arrays.items():
            arr = np.concatenate(parts) if parts else np.zeros((0,))
            self._arrays[k] = [arr]
            _atomic_save_npy(self.root / "arrays" / f"{k}.npy", arr)
            specs[k] = {"shape": list(arr.shape[1:]), "dtype": str(arr.dtype)}
        _atomic_write_text(self.root / "episodes.json", json.dumps(self._episodes, default=_json_default))
        meta = {
            "format": FORMAT,
            "fps": self.fps,
            "cameras": self.cameras,
            "num_frames": self._num_frames,
            "num_episodes": len(self._episodes),
            "arrays": specs,
            "info": self.info,
        }
        _atomic_write_text(self.root / "meta.json", json.dumps(meta, indent=1, default=_json_default))

    def close(self) -> None:
        self.flush()
        for f in self._bins.values():
            f.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


class StoreReader:
    """Read-only view of a store. Picklable; memmaps are opened lazily per process."""

    def __init__(self, root: str | Path, load_arrays: bool = True):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{self.root}: not a {FORMAT} store")
        self.episodes: List[Dict[str, Any]] = json.loads((self.root / "episodes.json").read_text())
        self.num_frames = int(self.meta["num_frames"])
        self.cameras = list(self.meta["cameras"])
        self.arrays: Dict[str, np.ndarray] = {}
        if load_arrays:
            for k in self.meta["arrays"]:
                self.arrays[k] = np.load(self.root / "arrays" / f"{k}.npy")[: self.num_frames]
        self._mm: Dict[str, np.memmap] = {}
        self._offsets: Dict[str, np.ndarray] = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mm"], d["_offsets"] = {}, {}
        return d

    def __len__(self) -> int:
        return self.num_frames

    def with_arrays(self, keys) -> "StoreReader":
        """A shallow copy holding only the per-frame arrays in `keys` (the arrays themselves shared, not copied)."""
        r = copy.copy(self)
        r.arrays = {k: v for k, v in self.arrays.items() if k in keys}
        r._mm, r._offsets = {}, {}
        return r

    def _open(self, cam: str):
        if cam not in self._mm:
            self._offsets[cam] = np.load(self.root / "images" / f"{cam}.offsets.npy")
            self._mm[cam] = np.memmap(self.root / "images" / f"{cam}.bin", dtype=np.uint8, mode="r")
        return self._mm[cam], self._offsets[cam]

    def image_bytes(self, cam: str, frame: int) -> np.ndarray:
        mm, offs = self._open(cam)
        return mm[offs[frame]: offs[frame + 1]]

    def image(self, cam: str, frame: int) -> np.ndarray:
        """Decoded RGB uint8 [H, W, 3]."""
        import cv2

        buf = np.asarray(self.image_bytes(cam, frame))
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot decode {self.root}/{cam}[{frame}]")
        return img[..., ::-1]


def write_array(root: str | Path, name: str, arr: np.ndarray) -> None:
    """Add or replace a per-frame array of an existing (closed) store, e.g. 'advantage'."""
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    arr = np.asarray(arr)
    if len(arr) != meta["num_frames"]:
        raise ValueError(f"{name}: {len(arr)} rows, store has {meta['num_frames']} frames")
    _atomic_save_npy(root / "arrays" / f"{name}.npy", arr)
    meta["arrays"][name] = {"shape": list(arr.shape[1:]), "dtype": str(arr.dtype)}
    _atomic_write_text(root / "meta.json", json.dumps(meta, indent=1, default=_json_default))


def encode_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    """Encode an RGB uint8 [H, W, 3] (or CHW) image as JPEG bytes (lossy, smaller than PNG)."""
    import cv2

    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def list_stores(paths: Iterable[str | Path]) -> List[Path]:
    """Expand directories into store roots (a dir with meta.json, or its children that have one)."""
    out = []
    for p in paths:
        p = Path(p)
        if (p / "meta.json").exists():
            out.append(p)
        else:
            out.extend(sorted(q.parent for q in p.glob("*/meta.json")))
    return out
