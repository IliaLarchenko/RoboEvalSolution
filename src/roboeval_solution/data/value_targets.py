"""Value-head targets, built from data/score_reward.py.

Observation field -> meaning (head, loss):

    success_target              1[episode succeeds]                                    success_head, BCE
    stage_target / stage_mask   stage j still ahead or reached <= 0.5 s ago [5]        stage_head, masked BCE
                                (sr.stage_frame_targets); mask j < J
    togo_target / togo_mask     normalised behaviour-to-go togo[t] [11] (TOGO_KEYS)    togo_head, MSE
    completion_target           t/T on successful rollouts; 0 on failed ones (the      completion_head, MSE
                                head then predicts completion·P(success));
                                successful demos: demo_values
    completion_weight_scale     per-sample completion-loss scale (demos DEMO_COMPLETION_SCALE)
    *_future_target             success, completion tf/T, togo[t] - togo[tf] at        WM-FAST heads
                                tf = min(t + FUTURE_HORIZON, T)
    wm_vector_target / mask     ψ(t) = [success, stage_target x5, togo[t] x11] (17)    WM-flow Q head (the model
                                                                                       subtracts sg(ψ̂(s_t)))
    is_weight                   1/(N p), clipped (unbiases aux losses under weighted sampling)

The dataset also emits task_id (task family id, CE) for every frame. Targets describe the trained policy, so they
come from rollouts. Demo frames are masked (NaN / mask False) except, for successful demos, the completion target.
Rollout episodes without a usable recording (recording_ok False) keep only the success targets; `error` episodes are
fully masked.
"""
from __future__ import annotations

import numpy as np

from roboeval_solution.data import score_reward as sr

FUTURE_HORIZON = 30  # = action horizon (WM-FAST / WM-flow future targets)
NUM_TOGO = len(sr.TOGO_KEYS)
NUM_STAGES = sr.MAX_STAGES
WM_VECTOR_DIM = sr.WM_VECTOR_DIM
# Per-frame store arrays the targets are built from (written by scripts/rollout.py --record).
REQUIRED_ARRAYS = ("m_run", "ev_acc", "stage_flags", "rb_counts")
DEMO_COMPLETION_SCALE = 0.5  # demo frames weigh half a rollout frame in the completion loss


def episode_value_arrays(ep: dict, arrays: dict, start: int, length: int) -> dict | None:
    """Value targets of one recorded rollout episode, or None when it has no usable recording.

    Returns togo [T+1, 11] (row T = 0; the TD(λ) targets of arrays["togo_td"] when present, else Monte-Carlo),
    stage_first [5] (sr.stage_first_reach: the per-frame targets of the stage head and of the WM vector's stage
    block), stage_mask [5] (bool, j < J), success (0/1), completion_T (T on success, 0 = no completion target) and
    completion_zero (failed episode: target 0)."""
    if ep.get("termination") == "error" or ep.get("recording_ok") is False:
        return None
    fam = ep["family"]
    sl = {k: np.asarray(arrays[k][start: start + length]) for k in REQUIRED_ARRAYS}
    incr = sr.step_increments(sl, ep)
    J = int(ep.get("stage_count") or sr.STAGE_COUNT[fam])
    stage_first = sr.stage_first_reach(sl["stage_flags"], J, ep.get("stage_flags_reset"))
    success = float(bool(ep.get("success")))
    togo = sr.togo_targets(incr, fam)
    if "togo_td" in arrays:  # TD(λ) targets written by compute_advantages.py
        td = np.asarray(arrays["togo_td"][start: start + length], np.float64)
        if np.isfinite(td).all():
            togo = np.vstack([td, np.zeros((1, td.shape[1]))])
    return {
        "success": success,
        "togo": togo.astype(np.float32),
        "stage_first": stage_first,
        "stage_mask": np.arange(NUM_STAGES) < J,
        "completion_T": length if success > 0.5 else 0,
        "completion_zero": bool(success < 0.5),
    }


def success_only_values(ep: dict) -> dict | None:
    """Rollout episode without a usable recording: success targets only (None for `error` episodes)."""
    if ep.get("termination") == "error":
        return None
    return {"success": float(bool(ep.get("success")))}


def demo_values(ep: dict, reward: np.ndarray, length: int, shift: int) -> dict | None:
    """Completion target of a successful demo episode: row r -> (r + shift) / T, T = the eval-aligned
    episode length up to the success step (rows after it are capped at 1; the sampler never anchors on them).

    Released demos have obs shift 1 (row r = the observation after a_r = eval frame r + 1) and the reward at
    row r is the reward after a_{r−1}: a first hit at row h means success after eval step h − 1, so T = h."""
    if not ep.get("success", False):
        return None
    T = length
    hit = np.flatnonzero(np.asarray(reward) > 0)
    if len(hit):
        T = int(hit[0])
    if T <= 0:
        return None
    return {"completion_T": T, "completion_shift": shift, "completion_scale": DEMO_COMPLETION_SCALE}


def frame_targets(values: dict | None, t: int, T: int, *, is_weight: float = 1.0) -> dict:
    """Observation-field targets for frame t (0-based within the episode) of an episode of length T.

    values: None (all masked), or a dict from episode_value_arrays / success_only_values / demo_values."""
    v = values or {}
    nan = np.float32(np.nan)
    tf = min(t + FUTURE_HORIZON, T)
    succ = v.get("success")
    togo, first, smask = v.get("togo"), v.get("stage_first"), v.get("stage_mask")
    out = {
        "success_target": nan, "success_future_target": nan,
        "completion_target": nan, "completion_future_target": nan,
        "completion_weight_scale": np.float32(v.get("completion_scale", 1.0)),
        "stage_target": np.zeros(NUM_STAGES, np.float32), "stage_mask": np.zeros(NUM_STAGES, bool),
        "togo_target": np.zeros(NUM_TOGO, np.float32),
        "togo_mask": np.zeros(NUM_TOGO, bool),
        "togo_future_target": np.zeros(NUM_TOGO, np.float32),
        "togo_future_mask": np.zeros(NUM_TOGO, bool),
        "wm_vector_target": np.zeros(WM_VECTOR_DIM, np.float32), "wm_vector_mask": np.zeros(WM_VECTOR_DIM, bool),
        "is_weight": np.float32(is_weight),
    }
    if succ is not None:
        out["success_target"] = out["success_future_target"] = np.float32(succ)
        out["wm_vector_target"][0] = succ
        out["wm_vector_mask"][0] = True
    cT = int(v.get("completion_T", 0))
    if v.get("completion_zero"):  # failed rollout: completion·success = 0
        out["completion_target"] = out["completion_future_target"] = np.float32(0.0)
    elif cT > 0:
        shift = int(v.get("completion_shift", 0))
        out["completion_target"] = np.float32(min((t + shift) / cT, 1.0))
        if "completion_shift" not in v:  # rollouts only: demos keep their future targets masked
            out["completion_future_target"] = np.float32(min(tf / cT, 1.0))
    if first is not None:
        out["stage_target"] = sr.stage_frame_targets(first, t)
        out["stage_mask"] = np.asarray(smask, bool).copy()
        # the WM-flow Q head predicts psi(t) - sg(psi_hat(s_t)) and psi_hat's stage block is the stage head: same
        # per-frame target, so reached stages leave no spurious residual
        out["wm_vector_target"][1: 1 + NUM_STAGES] = out["stage_target"]
        out["wm_vector_mask"][1: 1 + NUM_STAGES] = smask
    if togo is not None:
        out["togo_target"] = togo[t].astype(np.float32)
        out["togo_mask"] = np.ones(NUM_TOGO, bool)
        # WM-FAST: the chunk's own contribution (behaviour spent over t..tf), not the absolute togo[tf], which is
        # mostly state-dependent (already the V head's job) and gives the action conditioning a weak signal.
        out["togo_future_target"] = (togo[t] - togo[tf]).astype(np.float32)
        out["togo_future_mask"] = np.ones(NUM_TOGO, bool)
        out["wm_vector_target"][1 + NUM_STAGES:] = togo[t]
        out["wm_vector_mask"][1 + NUM_STAGES:] = True
    return out


VALUE_TARGET_KEYS = tuple(frame_targets(None, 0, 1))
