"""Score-exact rewards, value targets, value and advantages for RoboEval (pure numpy).

Single source of truth for the RL signal, shared by the dataset (head targets), compute_advantages.py
(advantages), the policy server (score-weighted candidate ranking) and the rollout workers (per-episode linearised
score). Imported from both the training (py3.11) and the simulator (py3.10) venv: numpy (+ json/pathlib) only,
never `roboeval`.

The objective. The leaderboard averages each metric over a task's episodes
and then applies the RoboEval Diagnostic Score S_f, so the per-episode reward is the linearisation
R_i = g_f · m_i with g_f = ∇S_f at the current policy's behaviour means (successful PositionAndOrientation, "PR",
episodes). Every episode metric is a sum of per-step increments, so r_t = g · Δm_t and the return-to-go is exact.

Conventions: frame t = observation before action t; running arrays hold the value after action t;
Δm_t = run_t − run_{t−1} with run_{−1} = the reset value.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

# ── official scoring constants (vendored from roboeval/challenge/scoring.py; tests check equality) ──
OUTCOME_SPLIT = {"success": 0.7, "tp": 0.3}
WEIGHTS = {"outcome": 0.4, "safety": 0.3, "efficiency": 0.2, "coordination": 0.1}
FAMILIES = {
    "safety": ("scc", "ecc", "sc", "mjj", "mcj"),
    "efficiency": ("cpl", "jpl", "opl", "ct"),
    "coordination": ("bavd", "bgvd"),
}
COUNT_METRICS = ("scc", "ecc", "sc")
RELATIVE_FLOOR = 0.1
CONTROL_HZ = 20

METRIC13 = ("success", "tp", "cpl", "jpl", "opl", "ct", "scc", "ecc", "sc", "mjj", "mcj", "bavd", "bgvd")
MI = {k: i for i, k in enumerate(METRIC13)}
BEHAVIOUR = METRIC13[2:]
TOGO_KEYS = ("time", "cpl", "jpl", "opl", "scc", "ecc", "sc", "mjj", "mcj", "bavd", "bgvd")  # 11 head outputs
TOGO_METRIC = tuple("ct" if k == "time" else k for k in TOGO_KEYS)  # METRIC13 channel of each togo output
MAX_STAGES = 5
TOGO_CAP = 5.0
WM_VECTOR_DIM = 1 + MAX_STAGES + len(TOGO_KEYS)  # [success, stage x5, togo x11] = 17

# Raw `info` keys of the running official metrics (spec.RAW_METRIC_KEYS), METRIC13 order; ct is derived.
RAW_KEYS = {
    "success": "success", "tp": "subtask_progress", "cpl": "total_cartesian_path_length",
    "jpl": "total_joint_path_length", "opl": "total_orientation_path_length", "scc": "self_collision_count",
    "ecc": "env_collision_count", "sc": "slip_count", "mjj": "overall_avg_joint_jerk",
    "mcj": "overall_avg_cartesian_jerk", "bavd": "bimanual_arm_velocity_difference",
    "bgvd": "bimanual_gripper_vertical_difference",
}
BUDGET_SECONDS = {"LiftPot": 10.0, "LiftTray": 10.0, "StackTwoBlocks": 20.0, "StackSingleBookShelf": 40.0,
                  "CubeHandover": 20.0, "PackBox": 20.0, "PickSingleBookFromTable": 15.0, "RotateValve": 20.0}
STAGE_COUNT = {"LiftPot": 4, "LiftTray": 4, "StackTwoBlocks": 3, "StackSingleBookShelf": 4, "CubeHandover": 3,
               "PackBox": 5, "PickSingleBookFromTable": 2, "RotateValve": 4}  # = keys pre-registered at reset

_REFERENCE_PATH = Path(__file__).resolve().parents[3] / "RoboEval" / "roboeval" / "challenge" / "human_reference.json"


@lru_cache(maxsize=1)
def human_reference() -> dict[str, dict[str, float]]:
    """{family: {metric: median over successful human demos}} (roboeval/challenge/human_reference.json)."""
    data = json.loads(_REFERENCE_PATH.read_text())
    return {fam: {k: float(v) for k, v in entry["median"].items()} for fam, entry in data.items()}


def _floor_c(key: str, ref: float) -> float:
    return max(RELATIVE_FLOOR * ref, 1.0 if key in COUNT_METRICS else 0.0, 1e-9)


def _phi(key: str, v, ref: float):
    c = _floor_c(key, ref)
    v = np.asarray(v, np.float64)
    return np.where(v <= ref, 1.0, (ref + c) / (v + c))


def _dphi(key: str, v, ref: float):
    c = _floor_c(key, ref)
    v = np.asarray(v, np.float64)
    return np.where(v <= ref, 0.0, -(ref + c) / (v + c) ** 2)


def _as_vec(m) -> np.ndarray:
    if isinstance(m, dict):
        return np.array([float(m[k]) for k in METRIC13], np.float64)
    return np.asarray(m, np.float64)


# ── score and gradient ──
def task_score_and_grad(family: str, mbar, outcome_override: float | None = 1.0) -> tuple[float, np.ndarray]:
    """S_f(m̄) and ∇S_f [13] (METRIC13 order). Behaviour entries use O := outcome_override (None = true O).
    Vectorised: mbar may be [..., 13]; returns arrays with matching leading dims."""
    ref = human_reference()[family]
    m = _as_vec(mbar)
    O = OUTCOME_SPLIT["success"] * m[..., MI["success"]] + OUTCOME_SPLIT["tp"] * m[..., MI["tp"]]
    B = 0.0
    grad = np.zeros(m.shape, np.float64)
    for fam, keys in FAMILIES.items():
        w = WEIGHTS[fam] / len(keys)
        for k in keys:
            B = B + w * _phi(k, m[..., MI[k]], ref[k])
            grad[..., MI[k]] = w * _dphi(k, m[..., MI[k]], ref[k])
    S = WEIGHTS["outcome"] * O + O * B
    o_b = O if outcome_override is None else outcome_override
    grad *= np.asarray(o_b)[..., None] if np.ndim(o_b) else o_b
    grad[..., MI["success"]] = OUTCOME_SPLIT["success"] * (WEIGHTS["outcome"] + B)
    grad[..., MI["tp"]] = OUTCOME_SPLIT["tp"] * (WEIGHTS["outcome"] + B)
    return (float(S) if np.ndim(S) == 0 else S), grad


def score_weights(eps_by_family: dict[str, list[dict]], *, n_eval: int = 50, n_boot: int = 2000,
                  min_success: int = 10, prev: dict[str, np.ndarray] | None = None, ema: float = 0.5,
                  seed: int = 0, info: dict | None = None) -> dict[str, np.ndarray]:
    """g_f per task: mean over n_boot resamples of n_eval episodes of ∇S_f at their means (O := 1),
    evaluated on successful PositionAndOrientation episodes; then g = ema·g_prev + (1−ema)·g_new.

    Episodes: dicts with "metrics" (official per-episode metrics), "success", "variation" and optional
    "run_order" (0 = newest run). Successful PR episodes are pooled run by run, newest first, until there are
    >= min_success; if still short, all PR episodes of the pool are used; with no PR episode at all the
    gradient is taken at 1.2x the human reference (and an existing prev is kept as is).
    `info` (optional) receives per task: mbar, B, n_eps, source.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for fam, eps in eps_by_family.items():
        pr = [e for e in eps if e.get("variation", "PositionAndOrientation") == "PositionAndOrientation"
              and e.get("termination") != "error"]
        succ = sorted((e for e in pr if e.get("success")), key=lambda e: e.get("run_order", 0))
        chosen, source = [], "success"
        if succ:
            orders = sorted({e.get("run_order", 0) for e in succ})
            for o in orders:
                chosen += [e for e in succ if e.get("run_order", 0) == o]
                if len(chosen) >= min_success:
                    break
        if len(chosen) < min_success:
            chosen, source = pr, "all_pr"
        if not chosen:
            if prev is not None and fam in prev:
                out[fam] = np.asarray(prev[fam], np.float64)
                if info is not None:
                    info[fam] = {"source": "prev", "n_eps": 0}
                continue
            ref = human_reference()[fam]
            mbar = np.array([1.0, 1.0] + [1.2 * ref[k] for k in BEHAVIOUR], np.float64)
            _, g_new = task_score_and_grad(fam, mbar)
            M = mbar[None]
            source = "reference"
        else:
            M = np.stack([_as_vec(e["metrics"]) for e in chosen])
            idx = rng.integers(0, len(M), size=(n_boot, n_eval))
            means = M[idx].mean(1)  # [n_boot, 13]
            _, grads = task_score_and_grad(fam, means)
            g_new = grads.mean(0)
            mbar = M.mean(0)
        if prev is not None and fam in prev:
            g = ema * np.asarray(prev[fam], np.float64) + (1.0 - ema) * g_new
        else:
            g = g_new
        out[fam] = g
        if info is not None:
            ref = human_reference()[fam]
            B = sum(WEIGHTS[f] / len(ks) * float(_phi(k, mbar[MI[k]], ref[k])) for f, ks in FAMILIES.items() for k in ks)
            info[fam] = {"mbar": mbar.tolist(), "B": B, "n_eps": len(M), "source": source}
    return out


# ── per-step decomposition ──
def _diff(run: np.ndarray, reset) -> np.ndarray:
    run = np.asarray(run, np.float64)
    prev = np.concatenate([np.asarray(reset, np.float64)[None], run[:-1]], 0)
    return run - prev


def step_increments(arrays: dict, ep: dict) -> np.ndarray:
    """[T, 13] per-step increments Δm_t whose column sums are the episode's official metrics, with the
    collision / slip channels replaced by their Rao-Blackwellised expectations (when rb_counts is given) and the
    fail fine added to the last frame of fail / unhealthy episodes cut short.

    arrays: one episode's slices of m_run [T,13], ev_acc [T,8], stage_flags [T,5] and (for RB) rb_counts [T,3].
    ep: family, termination, budget_steps, stage_count (J); optional stage_flags_reset [5], and for an episode
    branched mid-episode t0 (env steps replayed before frame 0), m_run_reset [13] and ev_acc_reset [8].
    """
    m_run = np.asarray(arrays["m_run"], np.float64)
    T = len(m_run)
    # episodes branched mid-episode (recovery runs, sim/recovery.py) start from the running values at the branch
    m0 = np.asarray(ep.get("m_run_reset") or np.zeros(13), np.float64)
    a0 = np.asarray(ep.get("ev_acc_reset") or np.zeros(8), np.float64)
    fam = ep["family"]
    J = int(ep.get("stage_count") or STAGE_COUNT[fam])
    incr = np.zeros((T, 13), np.float64)
    incr[:, MI["success"]] = m_run[:, MI["success"]]  # indicator on the success step
    flags = np.asarray(arrays["stage_flags"], np.float64)[:, :J]
    reset_flags = np.asarray(ep.get("stage_flags_reset") or [0] * MAX_STAGES, np.float64)[:J]
    incr[:, MI["tp"]] = _diff(flags, reset_flags).sum(1) / J
    for k in ("cpl", "jpl", "opl"):
        incr[:, MI[k]] = _diff(m_run[:, MI[k]], m0[MI[k]])
    incr[:, MI["ct"]] = 1.0 / CONTROL_HZ
    if "rb_counts" in arrays:
        rb = np.asarray(arrays["rb_counts"], np.float64)
        incr[:, MI["scc"]], incr[:, MI["ecc"]], incr[:, MI["sc"]] = rb[:, 0], rb[:, 1], rb[:, 2]
    else:
        for k in ("scc", "ecc", "sc"):
            incr[:, MI[k]] = _diff(m_run[:, MI[k]], m0[MI[k]])
    acc = np.asarray(arrays["ev_acc"], np.float64)
    n_jerk, n_vel, n_z = acc[-1, 2], acc[-1, 4], acc[-1, 6]
    if n_jerk > 0:
        incr[:, MI["mjj"]] = _diff(acc[:, 0], a0[0]) / n_jerk
        incr[:, MI["mcj"]] = _diff(acc[:, 1], a0[1]) / n_jerk
    if n_vel > 0:
        incr[:, MI["bavd"]] = _diff(acc[:, 3], a0[3]) / n_vel
    if n_z > 0:
        incr[:, MI["bgvd"]] = _diff(acc[:, 5], a0[5]) / n_z
    budget = int(ep.get("budget_steps") or round(BUDGET_SECONDS[fam] * CONTROL_HZ))
    steps = T + int(ep.get("t0") or 0)  # env clock at the end (a branched episode's replayed prefix included)
    if ep.get("termination") in ("fail", "unhealthy") and steps < budget:
        R = budget - steps
        incr[-1, MI["ct"]] += R / CONTROL_HZ
        for k in ("cpl", "jpl", "opl", "scc", "ecc", "sc"):
            incr[-1, MI[k]] += R * incr[:, MI[k]].sum() / T
    return incr


def fined_official(metrics: dict, termination: str, steps: int, budget_steps: int) -> np.ndarray:
    """Episode metric vector [13] from the official per-episode metrics plus the fail fine; no RB counts.
    Used where no per-step recording exists (e.g. un-recorded eval runs, Thompson-sampling rewards)."""
    m = _as_vec(metrics)
    if termination in ("fail", "unhealthy") and 0 < steps < budget_steps:
        R = budget_steps - steps
        m[MI["ct"]] += R / CONTROL_HZ
        for k in ("cpl", "jpl", "opl", "scc", "ecc", "sc"):
            m[MI[k]] += R * m[MI[k]] / steps
    return m


def step_rewards(incr: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.asarray(incr, np.float64) @ np.asarray(g, np.float64)


def returns_to_go(r: np.ndarray) -> np.ndarray:
    """G_t = Σ_{τ≥t} r_τ (γ = 1)."""
    return np.cumsum(np.asarray(r, np.float64)[::-1])[::-1].copy()


# ── head targets ──
def togo_scale(family: str) -> np.ndarray:
    """[11] metric units per head unit: togo_target = metric-to-go / scale."""
    ref = human_reference()[family]
    out = []
    for k in TOGO_KEYS:
        if k == "time":
            out.append(BUDGET_SECONDS[family])
        elif k in COUNT_METRICS:
            out.append(max(ref[k], 1.0))
        else:
            out.append(max(ref[k], 1e-9))
    return np.array(out, np.float64)


def togo_targets(incr: np.ndarray, family: str) -> np.ndarray:
    """[T+1, 11] normalised behaviour-to-go (incl. the fail fine); row T = 0 (terminal); clipped to [0, cap]."""
    cols = [MI[k] for k in TOGO_METRIC]
    T = len(incr)
    out = np.zeros((T + 1, len(TOGO_KEYS)), np.float64)
    out[:T] = np.cumsum(np.asarray(incr, np.float64)[::-1, cols], 0)[::-1]
    out /= togo_scale(family)[None]
    return np.clip(out, 0.0, TOGO_CAP)


STAGE_GRACE = 10  # frames (0.5 s ~ one inference call: 5-13 frames) after a stage is reached with target 1
NOT_AHEAD = -(10 ** 9)  # stage_first_reach of a stage that is never reached (or was already set at reset)


def stage_first_reach(stage_flags: np.ndarray, J: int, reset_flags=None) -> np.ndarray:
    """[5] index of the action that first sets each stage flag; NOT_AHEAD for stages never reached in the
    episode, already set at reset (an episode branched mid-episode) or j >= J."""
    sf = np.asarray(stage_flags, np.float64)
    reset = np.zeros(MAX_STAGES) if reset_flags is None else np.asarray(reset_flags, np.float64)
    out = np.full(MAX_STAGES, NOT_AHEAD, np.int64)
    for j in range(min(J, sf.shape[-1] if sf.ndim == 2 else 0)):
        hit = np.flatnonzero(sf[:, j] > 0.5)
        if len(hit) and not (j < len(reset) and reset[j] > 0.5):
            out[j] = hit[0]
    return out


def stage_frame_targets(first: np.ndarray, t, grace: int = STAGE_GRACE) -> np.ndarray:
    """Stage-head target at frame(s) t ([5], or [len(t), 5]): 1 while the stage is still ahead or was reached
    within the last `grace` frames (t <= first + grace), else 0. The value reads the head only on stages not
    reached before t (value_blocks), where this is P(reached later). After the grace window the target is 0, so a
    frame after a stage was reached and the same-looking frame of an episode that skipped it agree. The grace keeps
    the frames just after the flag sets (which look like the ones just before it) at 1: a 1 -> 0 edge at the flag
    would pull P̂ down on the approach to every stage."""
    first = np.asarray(first, np.int64)
    return (np.asarray(t)[..., None] <= first + grace).astype(np.float32)


# ── value and advantages ──
def ema_hold(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """EMA along frames (axis 0), seeded at x[0]; works for [T] and [T, k]. NaN inputs hold the last value."""
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    e = x[0].copy() if x.ndim > 1 else x[0]
    out[0] = e
    for t in range(1, len(x)):
        xt = x[t]
        e = np.where(np.isfinite(xt), alpha * xt + (1.0 - alpha) * e, e)
        out[t] = e
    return out


def flags_before(stage_flags: np.ndarray, reset_flags=None) -> np.ndarray:
    """[T, 5] stage flags reached before action t (stage_flags[t−1]; the reset flags at t = 0)."""
    sf = np.asarray(stage_flags, np.float64)
    first = np.zeros((1, sf.shape[1])) if reset_flags is None else np.asarray(reset_flags, np.float64)[None, : sf.shape[1]]
    return np.concatenate([first, sf[:-1]], 0)


def value_blocks(preds: dict, flags_before_: np.ndarray, g: np.ndarray, family: str, J: int,
                 completion_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Outcome and behaviour value blocks from the head predictions:
        Vo_t = (g_succ + w_c)·P̂_t + g_tp·(1/J)·Σ_{j<J, not reached before t} P̂stage_{t,j}
        Vb_t = Σ_k g_k·scale_k·ĥ_{t,k}
    w_c = completion_weight when the completion shaping is part of the reward (its terminal potential
    w_c·1[success] is expected return); 0 otherwise.
    preds: "success" [T], "stage" [T,5], "togo" [T,11]."""
    g = np.asarray(g, np.float64)
    T = len(flags_before_)
    Vo = np.zeros(T)
    Vo += (g[MI["success"]] + completion_weight) * np.nan_to_num(np.asarray(preds["success"], np.float64))
    st = np.nan_to_num(np.asarray(preds["stage"], np.float64))[:, :J]
    open_ = 1.0 - np.asarray(flags_before_, np.float64)[:, :J]
    Vo += g[MI["tp"]] / J * (st * open_).sum(1)
    h = np.nan_to_num(np.asarray(preds["togo"], np.float64))
    coef = np.array([g[MI[k]] for k in TOGO_METRIC]) * togo_scale(family)
    Vb = h @ coef
    return Vo, Vb


def _wvar(x, w):
    m = np.sum(w * x) / np.sum(w)
    return np.sum(w * (x - m) ** 2) / np.sum(w)


def _wcov(x, y, w):
    mx, my = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
    return np.sum(w * (x - mx) * (y - my)) / np.sum(w)


def _wmean(x, w):
    return np.sum(w * x) / np.sum(w)


def _alpha_solve(G, V, w, alpha0, n0, n_eps):
    """Ridge on the CENTRED moments toward alpha0, worth n0 episodes: (Cov(G, V) + k·alpha0·Var(V)) / ((1 + k)·Var(V)),
    k = n0 / n_eps."""
    var = _wvar(V, w)
    k = n0 / max(n_eps, 1)
    return float((_wcov(G, V, w) + k * alpha0 * var) / ((1.0 + k) * var))


def fit_alpha(G, Vo, Vb, ep_id, *, prior: dict, prior_episodes: int = 30, clip=(0.0, 1.0), n_boot: int = 500,
              min_episodes: int = 8, seed: int = 0) -> dict:
    """Dampened baseline α·V̂, V̂ = Vo + Vb, fitted on one run's task frames: one α = Cov(G, V̂)/Var(V̂) (the
    control-variate optimum; = ρ² ≤ 1 for a calibrated V̂, so clipped to [0, 1]), episode-weighted (w = 1/T_i so every
    episode counts once), from centred moments with a ridge toward the prior α worth prior_episodes episodes.
    a = mean(G) − α·mean(V̂) only centres the fit (diagnostic, not part of the baseline). Also: the unregularised α
    ("single"), per-block ρ, R² (none / α=1 / fit) and an episode-bootstrap CI."""
    G, Vo, Vb = (np.asarray(x, np.float64) for x in (G, Vo, Vb))
    V = Vo + Vb
    ep_id = np.asarray(ep_id)
    uniq, inv, counts = np.unique(ep_id, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    n_eps = len(uniq)
    alpha0 = float(prior.get("alpha", 0.5))
    out = {"n_eps": int(n_eps), "prior": {"alpha": alpha0}, "guards": []}
    has_var = bool(len(G)) and _wvar(V, w) > 1e-12
    if not has_var:
        out["guards"].append("var_v_zero")
    if n_eps < min_episodes or not has_var:
        if n_eps < min_episodes:
            out["guards"].append("few_episodes")
        alpha, a = alpha0, float(prior.get("a", 0.0))
    else:
        alpha = _alpha_solve(G, V, w, alpha0, prior_episodes, n_eps)
    alpha = float(np.clip(alpha, *clip))
    if "few_episodes" not in out["guards"] and len(G):
        a = float(_wmean(G, w) - alpha * _wmean(V, w))
    out.update(a=a, alpha=alpha)
    if has_var and n_eps >= 2:
        vg = _wvar(G, w)
        out["single"] = float(_wcov(G, V, w) / _wvar(V, w))
        out["rho_o"] = float(_wcov(G, Vo, w) / math.sqrt(max(vg * _wvar(Vo, w), 1e-24)))
        out["rho_b"] = float(_wcov(G, Vb, w) / math.sqrt(max(vg * _wvar(Vb, w), 1e-24)))
        if vg > 1e-12:
            out["r2_none"] = float(1.0 - np.sum(w * G ** 2) / np.sum(w) / vg)
            out["r2_alpha1"] = float(1.0 - np.sum(w * (G - V) ** 2) / np.sum(w) / vg)
            out["r2_fit"] = float(1.0 - np.sum(w * (G - a - alpha * V) ** 2) / np.sum(w) / vg)
        if n_boot and n_eps >= min_episodes:
            rng = np.random.default_rng(seed)
            rows = [np.flatnonzero(inv == i) for i in range(n_eps)]
            bs = []
            for _ in range(n_boot):
                idx = np.concatenate([rows[i] for i in rng.integers(0, n_eps, n_eps)])
                if _wvar(V[idx], w[idx]) > 1e-12:
                    bs.append(np.clip(_alpha_solve(G[idx], V[idx], w[idx], alpha0, prior_episodes, n_eps), *clip))
            if bs:
                out["ci_alpha"] = np.percentile(bs, [2.5, 97.5]).tolist()
    return out


def completion_rewards(C: np.ndarray, success: bool, weight: float = 0.3) -> np.ndarray:
    """Completion shaping as a reward (it enters G and the value fit): F_t = Φ_{t+1} − Φ_t, Φ_t = weight·Ĉ_t, and
    Φ_T = weight·(completion target at T) = weight·1[success] (completion = 1 at success, 0 on failures)."""
    C = np.nan_to_num(np.asarray(C, np.float64))
    phi = weight * C
    return np.r_[phi[1:], weight * float(bool(success))] - phi


def togo_td_targets(incr: np.ndarray, family: str, pred_togo: np.ndarray, lam: float = 0.9) -> np.ndarray:
    """[T, 11] TD(λ) behaviour-to-go targets (γ = 1): y_t = c_t + (1 − λ)·ĥ_{t+1} + λ·y_{t+1}, y_T = 0, ĥ_T = 0,
    with c_t the step's normalised behaviour increment (incl. the fail fine) and ĥ the recorded head predictions.
    λ = 1 is the Monte-Carlo togo_targets[:T]. Clipped to [0, TOGO_CAP]."""
    cols = [MI[k] for k in TOGO_METRIC]
    c = np.asarray(incr, np.float64)[:, cols] / togo_scale(family)[None]
    h = np.nan_to_num(np.asarray(pred_togo, np.float64))
    T = len(c)
    out = np.zeros((T, len(TOGO_KEYS)), np.float64)
    y = np.zeros(len(TOGO_KEYS))
    for t in range(T - 1, -1, -1):
        h_next = h[t + 1] if t + 1 < T else np.zeros(len(TOGO_KEYS))
        y = c[t] + (1.0 - lam) * h_next + lam * y
        out[t] = y
    return np.clip(out, 0.0, TOGO_CAP)


def gae(r: np.ndarray, V: np.ndarray, lam: float = 0.95) -> np.ndarray:
    """GAE(λ) with γ = 1: δ_t = r_t + V_{t+1} − V_t with V_T = 0, A_t = δ_t + λ·A_{t+1}.

    compute_advantages passes the dampened baseline of the shaped return-to-go (completion shaping in r), which is
    0 at the end. The baseline has no constant: a constant would leak into the last frames as −c·λ^(T−t−1)."""
    r, V = np.asarray(r, np.float64), np.asarray(V, np.float64)
    T = len(r)
    Vn = np.r_[V[1:], 0.0]
    delta = r + Vn - V
    A = np.empty(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = delta[t] + lam * acc
        A[t] = acc
    return A
