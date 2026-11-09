"""Where a failed rollout leaves the scripted policy's path, for recovery's "track" stage (sim/recovery.py).

A failed episode is replayed bit-exactly from its seed and stored actions, and the family's scripted policy runs from
the same seed (REFS parameter draws: the defaults, then random). Every step records both hands' positions, the
gripper commands and the task object's state. The policy's run is aligned to each complete scripted run by dynamic
time warping (open end: a failed run need not finish the path), and the cheapest alignment is the reference. Per
policy frame the deviation components are the left / right hand distance to the matched scripted frame, the two
gripper mismatches and the object distance (metre-equivalent: obj_scale). The onset is the first frame from which a
component stays above its threshold for PERSIST frames (0.5 s).

THRESHOLDS are the 97.5th percentile of each component over full-reward policy rollouts of the same seeds, with
floors (3 cm hands, 2 cm object, gripper 0.5 = a mismatch held 0.5 s). A gripper threshold of 1.0 means its timing
never counts (successful rollouts disagree with the scripted timing too often).

The track stage branches at the onset: the scripted controller of the reference run, advanced on its own episode to
the matched frame, takes over the policy's state and steers it back onto the path (scripts/rollout.py, job
"track_j")."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

REFS = 3
PERSIST = 10
GRIP_M = 0.05  # a gripper mismatch counts as 5 cm in the alignment
MIN_TAIL = 20  # frames of budget beyond the reference's remaining path
THRESHOLDS = {  # left / right hand (m), gripper L / R (mismatch), object (m-equivalent)
    "CubeHandover": dict(L=0.222, R=0.181, gL=0.5, gR=1.0, obj=0.168),
    "LiftPot": dict(L=0.125, R=0.123, gL=1.0, gR=1.0, obj=0.086),
    "LiftTray": dict(L=0.094, R=0.123, gL=0.5, gR=1.0, obj=0.069),
    "PackBox": dict(L=0.183, R=0.122, gL=0.5, gR=0.5, obj=0.020),
    "PickSingleBookFromTable": dict(L=0.304, R=0.340, gL=1.0, gR=1.0, obj=0.045),
    "RotateValve": dict(L=0.083, R=0.051, gL=0.5, gR=1.0, obj=0.020),
    "StackSingleBookShelf": dict(L=0.291, R=0.119, gL=1.0, gR=0.5, obj=0.110),
    "StackTwoBlocks": dict(L=0.236, R=0.240, gL=1.0, gR=0.5, obj=0.197),
}


def obj_scale(fam: str) -> np.ndarray:
    """Metres per unit of each object feature: positions 1; tilt 10 deg = 1 cm, yaw 20 deg = 1 cm; valve state
    0.1 = 1 cm; PackBox flap states (object_state: normalised joint positions) 0.001."""
    if fam == "PackBox":
        return np.array([0.001, 0.001])
    if fam == "RotateValve":
        return np.array([0.1, 0.1])
    return np.array([1, 1, 1, 0.001, 0.0005] * (2 if fam == "StackTwoBlocks" else 1))


def object_state(env, fam: str) -> np.ndarray:
    """The task object's state: body position, tilt and yaw (deg) per object; PackBox flap angles; valve states."""
    from roboeval_solution.sim.scripted_common import pose

    def body(b):
        p, q = pose(b)
        z = q.apply([0, 0, 1.0])
        return list(p) + [float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))), float(np.degrees(q.as_euler("xyz")[2]))]

    if fam == "LiftPot":
        return np.asarray(body(env.kitchenpot.body), np.float32)
    if fam == "LiftTray":
        return np.asarray(body(env.breakfast_tray.body), np.float32)
    if fam == "StackTwoBlocks":
        return np.asarray(body(env.blocks[0].body) + body(env.blocks[1].body), np.float32)
    if fam == "CubeHandover":
        return np.asarray(body(env.cube.body), np.float32)
    if fam in ("PickSingleBookFromTable", "StackSingleBookShelf"):
        return np.asarray(body(env.books[0].body), np.float32)
    if fam == "PackBox":
        from roboeval_solution.sim.scripted_packbox import flap_states

        return np.asarray(flap_states(env), np.float32)
    if fam == "RotateValve":
        return np.asarray([float(np.mean(v.get_state())) for v in env.valves], np.float32)
    raise ValueError(fam)


def frame(env, fam: str, act) -> np.ndarray:
    """One alignment feature row: hand positions (6), gripper commands x GRIP_M (2), scaled object state."""
    from roboeval_solution.sim.scripted_common import ee_sites

    ee = ee_sites(env)[:, :3].reshape(6)
    g = (np.asarray(act[12:14]) > 0.5).astype(np.float64) * GRIP_M
    return np.concatenate([ee, g, object_state(env, fam) * obj_scale(fam)])


def policy_path(env, fam: str, seed: int, actions: np.ndarray) -> np.ndarray:
    env.reset(seed=seed)
    rows = [frame(env, fam, np.zeros(14))]
    for a in actions:
        _, _, term, trunc, _ = env.step(a)
        rows.append(frame(env, fam, a))
        if env.success or term or trunc:
            break
    return np.asarray(rows)


def ref_seed(seed: int, k: int) -> tuple[int, bool]:
    """(scripted parameter seed, randomize) of reference run k (k = 0: the defaults)."""
    return int(seed) * 7 + k, k > 0


def scripted_path(env, fam: str, seed: int, k: int, budget: int) -> tuple[np.ndarray, bool]:
    from roboeval_solution.sim import scripted

    ps, rnd = ref_seed(seed, k)
    ctl, _ = scripted.make(scripted.for_family(fam), seed=ps, randomize=rnd)
    env.reset(seed=seed)
    rows = [frame(env, fam, np.zeros(14))]
    for t in range(budget):
        a = np.asarray(ctl(env, t), np.float32)
        a[12:14] = (a[12:14] > 0.5).astype(np.float32)
        _, _, term, trunc, _ = env.step(a)
        rows.append(frame(env, fam, a))
        if env.success or term or trunc:
            break
    flags = dict(env._evaluator._stage_flags)
    return np.asarray(rows), bool(env.success) and all(flags.values())


def dtw(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """For each frame of a, the matched frame of b (the last one its row passes through), and the mean cost. Cost per
    pair: left + right hand distance + gripper mismatches + object distance."""
    c = (np.linalg.norm(a[:, None, 0:3] - b[None, :, 0:3], axis=2)
         + np.linalg.norm(a[:, None, 3:6] - b[None, :, 3:6], axis=2)
         + np.abs(a[:, None, 6:8] - b[None, :, 6:8]).sum(2) + np.linalg.norm(a[:, None, 8:] - b[None, :, 8:], axis=2))
    n, m = c.shape
    d = np.full((n, m), np.inf)
    d[0] = np.cumsum(c[0])
    for i in range(1, n):  # d[i, j] = c[i, j] + min(d[i-1, j], d[i-1, j-1], d[i, j-1]), the last term by a prefix min
        x = c[i] + np.minimum(d[i - 1], np.concatenate([[np.inf], d[i - 1][:-1]]))
        s = np.cumsum(c[i])
        d[i] = s + np.minimum.accumulate(x - s)
    i, j = n - 1, int(np.argmin(d[-1]))  # open end
    match = np.zeros(n, int)
    match[i] = j
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin([d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]]))
            i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        match[i] = max(match[i], j)
    return match, float(d[-1].min() / n)


def components(a: np.ndarray, b: np.ndarray, match: np.ndarray) -> dict:
    bm = b[match]
    return {"L": np.linalg.norm(a[:, 0:3] - bm[:, 0:3], axis=1), "R": np.linalg.norm(a[:, 3:6] - bm[:, 3:6], axis=1),
            "gL": np.abs(a[:, 6] - bm[:, 6]) / GRIP_M, "gR": np.abs(a[:, 7] - bm[:, 7]) / GRIP_M,
            "obj": np.linalg.norm(a[:, 8:] - bm[:, 8:], axis=1)}


def onset(comp: dict, thr: dict) -> tuple[int | None, list]:
    """First frame from which a component stays over its threshold for PERSIST frames, and the components that do."""
    over = {k: comp[k] > thr[k] for k in thr}
    for t in range(max(0, len(comp["L"]) - PERSIST)):
        hit = [k for k in ("obj", "L", "R", "gL", "gR") if over[k][t: t + PERSIST].all()]
        if hit:
            return t, hit
    return None, []


def locate(env, fam: str, seed: int, actions: np.ndarray, budget: int) -> dict | None:
    """The track branch of one failed episode: {t_b, ref_k, j, ref_len, what}, or None (no complete reference run, no
    onset, or too little budget left for the rest of the reference path at the onset)."""
    a = policy_path(env, fam, seed, actions)
    best = None
    for k in range(REFS):
        b, ok = scripted_path(env, fam, seed, k, budget)
        if ok:
            match, cost = dtw(a, b)
            if best is None or cost < best[3]:
                best = (k, b, match, cost)
    if best is None:
        return None
    k, b, match, _ = best
    t, what = onset(components(a, b, match), THRESHOLDS[fam])
    if t is None or t < 1:
        return None
    ref_len = len(b) - 1

    def fits(tb):
        return (1 <= tb < len(a) - 1 and int(match[tb]) < ref_len - 1
                and budget - tb >= ref_len - int(match[tb]) + MIN_TAIL)

    if not fits(t):
        return None
    return {"t_b": int(t), "ref_k": k, "j": int(match[t]), "ref_len": ref_len, "what": what}


def _work(job):
    global _ENV
    fam, seed, budget, actions, key = job
    if _ENV is None or _ENV[0] != fam:  # jobs come sorted by family: one env per worker at a time
        if _ENV is not None:
            _ENV[1].close()
        from rollout import make_env_any, make_interface  # noqa: scripts/ is on sys.path (rollout.py)

        _ENV = (fam, make_env_any(fam, "PositionAndOrientation", make_interface([], 64)))
    try:
        return key, locate(_ENV[1], fam, seed, actions, budget)
    except Exception as e:  # one episode's replay never stops the run
        return key, {"error": repr(e)}


def _init(path):
    import sys

    global _ENV
    _ENV = None
    sys.path.insert(0, path)


def find_track(targets: list[dict], workers: int = 8) -> dict:
    """{index in targets: track branch} for failed targets (sim/recovery.failed_episodes entries); replays and scripted
    references run in a process pool (CPU only)."""
    jobs = sorted(((f["family"], int(f["seed"]), int(f["budget"]), f["actions"], i)
                   for i, f in enumerate(targets)), key=lambda j: j[0])
    out = {}
    if not jobs:
        return out
    scripts = str(Path(__file__).resolve().parents[3] / "scripts")
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(scripts,)) as pool:
        for key, res in pool.imap_unordered(_work, jobs, chunksize=4):
            if res and "error" not in res:
                out[key] = res
    return out


def summary(found: dict, targets: list[dict]) -> str:
    by = {}
    for i, f in enumerate(targets):
        b = by.setdefault(f["family"], [0, 0])
        b[0] += 1
        b[1] += i in found
    return json.dumps({k: f"{v[1]}/{v[0]}" for k, v in sorted(by.items())})
