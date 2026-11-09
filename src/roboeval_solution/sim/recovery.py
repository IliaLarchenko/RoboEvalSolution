"""Hard-case recovery from recorded rollouts (scripts/rollout.py --recover).

Targets: failed episodes, and successful ones that missed a stage (task progression < 1). A target is branched where
the policy's own recorded P(success) drops (failures), else at the first drop of its predicted P(stage j reached) for
a stage j it never reached (t_drop; t_peak = the P̂ maximum in the 3 s before it), or earlier at the first observable
stall event (event_frame: a gripper closed on nothing, the hands hovering, RotateValve's wrist twist blocked while
holding): P̂ drops only ~15-40 frames after such an event. The env replays the original actions bit-exactly from the
seed (clock, budget, stage flags and metric accumulators inherited), optional non-policy actions follow (undo the last
moves / a random ee shift), then the policy takes over.

Stages (how the branch state is made):
  drop      at t_drop
  back2s    2 s before t_drop
  peak      at t_peak, the P̂ peak in the PEAK_WINDOW frames (3 s) before the drop (skipped when < 10 frames
            before it)
  reverse   at t_drop, the last 30 actions (1.5 s) undone in reverse order
  perturb   at t_drop, a random 2–5 cm / ≤ 8° ee shift per arm over 10 steps
  retry     at t_drop (also when the object was dropped), the family's scripted policy takes over the failed state
            (sim/scripted.Retry: both hands open, rise clear, then a fresh scripted attempt from the props as they
            lie); the served policy is still queried every chunk for its head outputs, stored like a rollout's
  track     at the frame a failed episode leaves the scripted policy's path (sim/deviation.py), the scripted
            controller of the matched reference run, advanced to the matched frame, takes over and steers back onto
            the path. Targets: every failed episode with an onset (failed_episodes, rollout.py --recover-track)
Stages that start at the failure itself are skipped when the object was dropped there (a slip recorded within
[t_drop − 1.5 s, t_drop + 0.5 s]).
The cascade's "|"-separated groups (rollout.py: "drop,perturb,reverse,peak,back2s|retry") run as independent chains
per target. A try counts (ends its chain, is stored) only when it is complete (counts()). The targets no stage
recovered are listed in the run's left.jsonl.

A recovery run's recovery.json names the collect it branched ("collect"), which pairs the two in training
(training/flywheel.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

AT_FAILURE = ("drop", "perturb", "reverse")  # stages that start at the failure point
REVERSE_STEPS = 30  # reverse: actions undone (1.5 s)
BACK_STEPS = 40  # back2s: branch 2 s before the drop
PERTURB_DIST, PERTURB_DEG = (0.02, 0.05), 8.0  # perturb: ee shift per arm (m), max rotation (deg)
# peak: t_peak is the P̂ argmax within this many frames before the drop (3 s at 20 Hz). Over the whole prefix it is
# mostly near the episode start, i.e. a restart, not a recovery.
PEAK_WINDOW = 60
# retry: minimum frames left at the branch: the Retry entry (open, rise, settle: ~40) + the scripted policy's p95
# episode length
RETRY_LEFT = {"LiftPot": 110, "LiftTray": 115, "RotateValve": 125, "StackTwoBlocks": 135, "CubeHandover": 150,
              "PickSingleBookFromTable": 150, "StackSingleBookShelf": 195, "PackBox": 205}
SERVER_KEYS = ("inpaint_snap", "inpaint_pin_alpha", "inpaint_correction_step_cap", "time_threshold_inpaint",
               "noise_temperature", "correlation_beta", "num_steps", "cfg_scale", "num_candidates", "hide_velocity",
               "hide_front", "hide_external", "time_warp", "rank_full")
CLIENT_KEYS = ("replan", "keep", "gripper_threshold", "exec_speed", "stall_boost")
META = "recovery.json"


def source_collect(run_dir: str | Path) -> str:
    """Name of the collect a recovery run branched (its recovery.json's "collect")."""
    return json.loads((Path(run_dir) / META).read_text())["collect"]


def find_drop(p: np.ndarray, drop: float, min_t: int, max_t: int, min_peak: float = 0.6,
              peak_window: int = PEAK_WINDOW) -> tuple[int, int] | None:
    """(t_peak, t_drop) of the first P̂ drop of at least `drop` below its running max (and below 0.5), from a
    running max of at least `min_peak` (early-episode noise of P̂ is not a drop); t_peak = the P̂ argmax within the
    `peak_window` frames before t_drop (t_drop itself included)."""
    run = np.maximum.accumulate(p)
    cand = np.flatnonzero((run - p >= drop) & (p < 0.5) & (run >= min_peak))
    cand = cand[(cand >= min_t) & (cand <= max_t)]
    if not len(cand):
        return None
    t_drop = int(cand[0])
    lo = max(t_drop - int(peak_window), 0)
    return lo + int(np.argmax(p[lo: t_drop + 1])), t_drop


TWIST_WINDOW, TWIST_MIN_DEG, TWIST_RATIO = 10, 15.0, 3.0  # RotateValve blocked twist (event_frame)


def event_frame(fam: str, actions: np.ndarray, opening: np.ndarray, state_ee: np.ndarray,
                after: int = 0) -> tuple[int, str] | None:
    """(frame, kind) of the first observable stall event of a recorded episode, or None: tuning/stall.StallDetector's
    "empty" / "motion" on the executed actions and the gripper openings (opening[i + 1] follows action i), and for
    RotateValve a blocked twist: a hand closed on the wheel (opening >= 0.3) commanding >= TWIST_MIN_DEG of wrist
    rotation over TWIST_WINDOW frames while executing < 1 / TWIST_RATIO of it (a low grasp jams the twist). Only
    events after frame `after` count (the detector still runs from the start)."""
    from roboeval_solution.tuning.stall import StallDetector

    acts = np.asarray(actions, np.float64).copy()
    acts[:, 12:14] = acts[:, 12:14] > 0.5
    opening = np.asarray(opening, np.float64)
    n = len(acts)
    d = StallDetector(fam)
    found = None
    for i in range(n):
        ev = d.on_step(acts[i], opening[min(i + 1, len(opening) - 1)], i + 1)
        if ev and i + 1 > after:
            found = (i + 1, ev[0])
            break
    if fam == "RotateValve":
        ee = np.asarray(state_ee, np.float64)
        rot = [R.from_rotvec(ee[:, 3:6]), R.from_rotvec(ee[:, 9:12])]
        cmd = np.stack([np.linalg.norm(acts[:, 3:6], axis=1), np.linalg.norm(acts[:, 9:12], axis=1)], 1)
        W = TWIST_WINDOW
        for i in range(max(W, after + 1), min(n, found[0] if found else n)):
            for h in (0, 1):
                if acts[i, 12 + h] < 0.5 or opening[i, h] < 0.3:
                    continue
                c = np.degrees(cmd[i - W: i, h].sum())
                done = np.degrees((rot[h][i] * rot[h][i - W].inv()).magnitude())
                if c >= TWIST_MIN_DEG and done < c / TWIST_RATIO:
                    return i, "twist"
    return found


def reverse_actions(acts: np.ndarray, t: int, k: int) -> np.ndarray:
    """Undo acts[t-k:t] in reverse order: −Δp and r⁻¹ per arm (world-frame deltas), gripper = the command
    in force before the undone step."""
    seg = acts[max(t - k, 0): t]
    out = []
    for i in range(len(seg) - 1, -1, -1):
        a = seg[i].astype(np.float64).copy()
        for p0, r0 in ((0, 3), (6, 9)):
            a[p0: p0 + 3] = -a[p0: p0 + 3]
            a[r0: r0 + 3] = R.from_rotvec(seg[i][r0: r0 + 3]).inv().as_rotvec()
        prev = acts[max(t - k, 0) + i - 1] if max(t - k, 0) + i - 1 >= 0 else seg[0]
        a[12:14] = prev[12:14]
        out.append(a)
    return np.asarray(out, np.float32)


def perturb_actions(rng: np.random.Generator, grip: np.ndarray, steps: int = 10, dist=PERTURB_DIST,
                    max_deg: float = PERTURB_DEG) -> np.ndarray:
    """Random ee shift per arm spread over `steps` env steps (constant per-step delta), grippers held."""
    out = np.zeros((steps, 14), np.float32)
    for p0, r0 in ((0, 3), (6, 9)):
        d = rng.normal(size=3); d *= rng.uniform(*dist) / np.linalg.norm(d)
        ax = rng.normal(size=3); ax *= np.deg2rad(rng.uniform(0, max_deg)) / np.linalg.norm(ax)
        out[:, p0: p0 + 3] = d / steps
        out[:, r0: r0 + 3] = R.from_rotvec(ax).as_rotvec() / steps  # small angle: per-step share of the rotation
    out[:, 12:14] = grip
    return out


def complete(ep: dict) -> bool:
    """Success with every stage reached (task progression 1): the full outcome reward."""
    tp = (ep.get("metrics") or {}).get("tp", ep.get("tp", 1.0))
    return bool(ep.get("success")) and float(tp) >= 1.0 - 1e-6


def counts(ep: dict) -> bool:
    """Does a recovery try count (end its chain, get stored)? Only when complete, at every stage and for every
    target: a stored success short of a stage (e.g. a one-handed LiftPot lift) would reinforce the commonest partial
    outcome."""
    return complete(ep)


def find_failures(collect: str | Path, drop: float = 0.25, min_peak: float = 0.6, min_drop_t: int = 60,
                  min_left: int = 60, slip_window: tuple[int, int] = (30, 10),
                  peak_window: int = PEAK_WINDOW) -> list[dict]:
    """Recovery targets of a collect's PositionAndOrientation rollout episodes, each with origin, actions, t_peak,
    t_drop, dropped and its detector: failures with a P̂(success) drop ("success"); failures without one, and
    successes that missed a stage, with a drop of P̂(stage j reached) for a stage j never reached ("stage<j>", the
    earliest such drop); t_peak is the P̂ argmax within `peak_window` frames before the drop (the peak stage's branch
    point); an earlier stall event (event_frame) moves the branch there."""
    from roboeval_solution.data.store import StoreReader, list_stores

    out = []
    for root in list_stores([Path(collect) / "stores"]):
        r = StoreReader(root)
        for i, e in enumerate(r.episodes):
            if not e.get("recording_ok") or e.get("variation") != "PositionAndOrientation":
                continue
            score = float((e.get("score_ep") or {}).get("score", 1.0))
            st, n = int(e["start"]), int(e["length"])
            budget = int(e["budget_steps"])
            max_t = min(n - 1, budget - min_left)
            found = detector = None
            if complete(e):
                continue
            if not e.get("success"):
                ps = np.asarray(r.arrays["pred_success"][st: st + n], np.float64).ravel()
                found = find_drop(ps, drop, min_drop_t, max_t, min_peak, peak_window)
                detector = "success" if found else None
            if found is None:  # the earliest drop of P̂(stage j) for a stage j never reached
                J = int(e["stage_count"])
                flags = np.asarray(r.arrays["stage_flags"][st: st + n], np.float64)[:, :J]
                pst = np.asarray(r.arrays["pred_stage"][st: st + n], np.float64)[:, :J]
                for j in range(J):
                    d = None if flags[-1, j] > 0.5 else find_drop(pst[:, j], drop, min_drop_t, max_t, min_peak,
                                                                  peak_window)
                    if d is not None and (found is None or d[1] < found[1]):
                        found, detector = d, f"stage{j}"
            ev = event_frame(e["family"], r.arrays["actions"][st: st + n], r.arrays["state_gripper"][st: st + n],
                             r.arrays["state_ee"][st: st + n])
            if ev is not None and min_drop_t // 2 <= ev[0] <= max_t and (found is None or ev[0] < found[1]):
                ps = np.asarray(r.arrays["pred_success"][st: st + n], np.float64).ravel()
                lo = max(ev[0] - int(peak_window), 0)  # the branch moves to the event, the peak to before it
                found, detector = (lo + int(np.argmax(ps[lo: ev[0] + 1])), int(ev[0])), ev[1]
            if found is None:
                continue
            t_peak, t_drop = found
            sc = np.asarray(r.arrays["sc"][st: st + n])
            dropped = t_drop is not None and bool(sc[min(t_drop + slip_window[1], n - 1)]
                                                  > sc[max(t_drop - slip_window[0], 0)])
            out.append({"family": e["family"], "variation": e["variation"], "seed": int(e["seed"]), "budget": budget,
                        "actions": np.asarray(r.arrays["actions"][st: st + n], np.float32),
                        "t_peak": t_peak, "t_drop": t_drop, "dropped": dropped,
                        "detector": detector,
                        "origin": {"collect": Path(collect).name, "store": root.name, "episode": i, "t_peak": t_peak,
                                   "t_drop": t_drop, "length": n, "detector": detector,
                                   "success": bool(e.get("success")), "tp": float(e.get("tp", 0.0)),
                                   "score": score, "dropped": dropped,
                                   "target": "failure" if not e.get("success") else "stages"}})
    return out


def failed_episodes(collect: str | Path) -> list[dict]:
    """Every failed episode of a collect's PR rollouts (recording ok) as a target for the track stage (its branch:
    sim/deviation.find_track): detector "deviation", t_peak / t_drop None."""
    from roboeval_solution.data.store import StoreReader, list_stores

    out = []
    for root in list_stores([Path(collect) / "stores"]):
        r = StoreReader(root)
        for i, e in enumerate(r.episodes):
            if not e.get("recording_ok") or e.get("variation") != "PositionAndOrientation" or e.get("success"):
                continue
            st, n = int(e["start"]), int(e["length"])
            out.append({"family": e["family"], "variation": e["variation"], "seed": int(e["seed"]),
                        "budget": int(e["budget_steps"]),
                        "actions": np.asarray(r.arrays["actions"][st: st + n], np.float32),
                        "t_peak": None, "t_drop": None, "dropped": False, "detector": "deviation",
                        "origin": {"collect": Path(collect).name, "store": root.name, "episode": i, "t_peak": None,
                                   "t_drop": None, "length": n, "detector": "deviation", "success": False,
                                   "tp": float(e.get("tp", 0.0)),
                                   "score": float((e.get("score_ep") or {}).get("score", 0.0)), "dropped": False,
                                   "target": "failure"}})
    return out


def stage_prefix(f: dict, stage: str, rng: np.random.Generator, min_left: int = 30) -> tuple[np.ndarray, int] | None:
    """(non-policy env actions to execute, branch frame) for one target and stage, or None if not applicable."""
    acts, t_drop, t_peak = f["actions"], f["t_drop"], f["t_peak"]
    if stage == "track":  # the deviation onset (sim/deviation.locate)
        if not f.get("track"):
            return None
        t_b = int(f["track"]["t_b"])
        return acts[:t_b], t_b
    if stage == "retry":  # the failed state as it is, slipped or dropped object included
        if t_drop is None or t_drop > f["budget"] - RETRY_LEFT[f["family"]]:
            return None
        return acts[:t_drop], int(t_drop)
    if t_drop is None or (f.get("dropped") and stage in AT_FAILURE):
        return None
    if stage == "drop":
        pre, t_b = acts[:t_drop], t_drop
    elif stage == "back2s":
        t_b = max(t_drop - BACK_STEPS, 20)
        pre = acts[:t_b]
    elif stage == "peak":
        if t_drop - t_peak < 10:
            return None
        t_b = max(t_peak, 1)
        pre = acts[:t_b]
    elif stage == "reverse":
        t_b = t_drop
        pre = np.concatenate([acts[:t_drop], reverse_actions(acts, t_drop, REVERSE_STEPS)])
    elif stage == "perturb":
        t_b = t_drop
        pre = np.concatenate([acts[:t_b], perturb_actions(rng, acts[t_b - 1, 12:14])])
    else:
        raise ValueError(f"unknown recovery stage {stage}")
    if len(pre) > f["budget"] - min_left:
        return None
    return pre, int(t_b)


def best_params(path: str | Path) -> dict:
    """{family: (client params, server overrides)} from a param_search_best.json (per-task posterior-best)."""
    best = json.loads(Path(path).read_text())["best"] if Path(path).exists() else {}
    return {f: ({k: b[k] for k in CLIENT_KEYS if b.get(k) is not None},
                {k: b[k] for k in SERVER_KEYS if b.get(k) is not None}) for f, b in best.items()}


def script_seed(seed: int, chain: int) -> int:
    """Scripted-policy parameter seed of one target's retry. Per target (chain), not per env seed: a collect
    rolls each seed out several times (--repeats-per-seed), and two targets on one seed with one parameter draw are
    the same deterministic episode twice."""
    return int(np.random.SeedSequence([int(seed), int(chain), 0x5C1]).generate_state(1)[0])


def make_job(f: dict, stage: str, prefix: np.ndarray, t_b: int, params: dict, **extra) -> dict:
    client, over = params.get(f["family"], ({}, {}))
    if stage == "retry":  # rollout.py acts with the family's scripted policy, asking the server for its heads
        from roboeval_solution.sim.scripted import for_family

        extra = {"scripted": for_family(f["family"]), "script_seed": script_seed(f["seed"], extra.get("chain", 0)),
                 "retry": True, **extra}
    elif stage == "track":  # the reference run's controller (same parameter draw), advanced to the matched frame
        from roboeval_solution.sim.deviation import ref_seed
        from roboeval_solution.sim.scripted import for_family

        ps, rnd = ref_seed(f["seed"], f["track"]["ref_k"])
        extra = {"scripted": for_family(f["family"]), "script_seed": ps, "script_randomize": rnd,
                 "track_j": int(f["track"]["j"]), **extra}
    return {"family": f["family"], "variation": f["variation"], "seed": f["seed"], "overrides": dict(over),
            "client": dict(client), "prefix": prefix, "source": "recovery", "scenario": stage, "branch_t": t_b,
            "origin": f["origin"], **extra}
