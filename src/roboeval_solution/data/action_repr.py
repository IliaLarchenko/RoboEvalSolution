"""Action / state representations shared by training (dataset) and inference (policy server).

Env action (ee-delta mode, 14-D):
    [L dpos(3), L drotvec(3), R dpos(3), R drotvec(3), L grip, R grip]
The env composes each step's delta onto FK(ctrl) (the commanded joint targets):
    pos' = pos + dpos ;  R' = R(drotvec) * R      (world-frame left multiplication)
Gripper commands are absolute: 0 = open, 1 = close.

Model action representation ("cumulative"): pose of step k relative
to the commanded pose at chunk start, i.e. the composition of deltas 0..k (pos summed, rotations
composed). Exactly invertible back to per-step deltas, smooth targets, no in-chunk drift from
re-integrating noisy per-step predictions.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import RotationSpline

ACTION_DIM = 14
POSE_SLICES = ((0, 3, 3, 6), (6, 9, 9, 12))  # (pos_lo, pos_hi, rot_lo, rot_hi) per arm
GRIP = slice(12, 14)


def delta_to_cumulative(a: np.ndarray) -> np.ndarray:
    """[H, 14] per-step deltas -> [H, 14] cumulative relative poses (grippers unchanged)."""
    a = np.asarray(a, np.float64)
    out = a.copy()
    H = len(a)
    rv = np.stack([a[:, r0:r1] for _, _, r0, r1 in POSE_SLICES], axis=1)  # [H, 2, 3]
    mats = R.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(H, 2, 3, 3)
    cum = np.empty_like(mats)
    acc = np.broadcast_to(np.eye(3), (2, 3, 3))
    for k in range(H):  # R_k = r_k * R_{k-1}  (world-frame left multiplication)
        acc = mats[k] @ acc
        cum[k] = acc
    cum_rv = R.from_matrix(cum.reshape(-1, 3, 3)).as_rotvec().reshape(H, 2, 3)
    for i, (p0, p1, r0, r1) in enumerate(POSE_SLICES):
        out[:, p0:p1] = np.cumsum(a[:, p0:p1], axis=0)
        out[:, r0:r1] = cum_rv[:, i]
    return out.astype(np.float32)


def cumulative_to_delta(c: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_to_cumulative`."""
    c = np.asarray(c, np.float64)
    out = c.copy()
    for p0, p1, r0, r1 in POSE_SLICES:
        out[:, p0:p1] = np.diff(c[:, p0:p1], axis=0, prepend=np.zeros((1, 3)))
        cum = R.from_rotvec(c[:, r0:r1])
        prev = R.concatenate([R.identity(), cum[:-1]])
        out[:, r0:r1] = (cum * prev.inv()).as_rotvec()
    return out.astype(np.float32)


def to_model(raw: np.ndarray) -> np.ndarray:
    return delta_to_cumulative(raw)


def to_env(model_actions: np.ndarray) -> np.ndarray:
    return cumulative_to_delta(model_actions)


def rebase_cumulative(c: np.ndarray, k: int) -> np.ndarray:
    """Re-express cumulative poses relative to step k-1 (the pose reached after executing k
    actions), dropping the first k. Used to turn the unexecuted tail of a chunk into
    inpainting targets for the next chunk."""
    c = np.asarray(c, np.float64)
    if k <= 0:
        return c.astype(np.float32)
    out = c[k:].copy()
    for p0, p1, r0, r1 in POSE_SLICES:
        out[:, p0:p1] = c[k:, p0:p1] - c[k - 1, p0:p1]
        base = R.from_rotvec(c[k - 1, r0:r1])
        out[:, r0:r1] = (R.from_rotvec(c[k:, r0:r1]) * base.inv()).as_rotvec()
    return out.astype(np.float32)


def resample_deltas(a: np.ndarray, k: int, n: int) -> np.ndarray:
    """Play the first k per-step env deltas of a chunk over n env steps (speed k / n): the commanded pose
    trajectory relative to the chunk start is sampled at model times j*k/n (j = 1..n) on cubic splines
    through the chunk's poses (positions: CubicSpline, rotations: RotationSpline; integer speed ratios
    sample the poses exactly); gripper commands are held from the model step covering each sample. Ends
    at exactly the pose of k model steps, so the chunk tail after step k stays a valid continuation."""
    a = np.asarray(a, np.float32)
    if n == k:
        return a[:k].copy()
    c = delta_to_cumulative(a[:k]).astype(np.float64)  # pose after model steps 1..k
    t_src = np.arange(k + 1, dtype=np.float64)  # model time 0..k (0 = chunk start)
    t_dst = np.arange(1, n + 1, dtype=np.float64) * k / n
    t_dst[-1] = float(k)  # exact end pose (no rounding drift)
    out = np.zeros((n, ACTION_DIM), np.float64)
    for p0, p1, r0, r1 in POSE_SLICES:
        pos = np.vstack([np.zeros((1, 3)), c[:, p0:p1]])
        out[:, p0:p1] = CubicSpline(t_src, pos, axis=0)(t_dst)
        rots = R.concatenate([R.identity(), R.from_rotvec(c[:, r0:r1])])
        out[:, r0:r1] = RotationSpline(t_src, rots)(t_dst).as_rotvec()
    out[:, GRIP] = a[np.clip(np.ceil(t_dst - 1e-9).astype(int) - 1, 0, k - 1), GRIP]
    return cumulative_to_delta(out)


# ── state ───────────────────────────────────────────────────────────────────
# proprioception (36) = qpos(18) + qvel(18); per arm: 7 arm joints + 2 finger joints.
# Order: [L arm 7, L fingers 2, R arm 7, R fingers 2].
_ARM_QPOS = np.r_[0:7, 9:16]
_ARM_QVEL = 18 + _ARM_QPOS


def make_state(proprio: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    """Model state vector from env proprioception (…, 36) and gripper state (…, 2): 14 arm joints + 2 gripper
    openings + 14 arm joint velocities = 30."""
    proprio = np.asarray(proprio, np.float32)
    grippers = np.asarray(grippers, np.float32)
    return np.concatenate([proprio[..., _ARM_QPOS], grippers, proprio[..., _ARM_QVEL]], axis=-1)
