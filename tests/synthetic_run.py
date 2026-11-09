"""Synthetic rollout runs for the advantage tests (CPU only): a few episodes x ~30 frames x 5 cameras of
random JPEGs, the recorded arrays with held fake predictions, and the run files compute_advantages.py writes
(alpha.json, adv_meta.json)."""
import json
from pathlib import Path

import numpy as np

from roboeval_solution.data import score_reward as sr
from roboeval_solution.data.store import StoreWriter, encode_jpeg, write_array

CAMS = ("head", "front", "external", "left_wrist", "right_wrist")
FAMILY, J, BUDGET = "LiftPot", 4, 200
LAM, EMA, CW, ROUND = 0.95, 0.2, 0.3, 3
G = np.array([0.62, 0.27, -0.024, -0.003, -0.004, -0.006, -0.027, -0.027, -0.027, -3e-4, -1e-3, -0.05, -1.0])
ALPHA = {"a": 0.02, "alpha": 0.8}


def _held(x, calls):
    t = np.arange(len(x))
    return x[np.maximum.accumulate(np.where(calls, t, 0))]


def _episode(rng, T, success, seed, img=32):
    t = np.arange(T)
    n_set = min(J if success else int(rng.integers(0, J)), max(T - 2, 0))
    times = np.sort(rng.choice(np.arange(1, max(T - 1, 1)), n_set, replace=False))
    if success and n_set:
        times[-1] = T - 1
    flags = np.zeros((T, 5), np.uint8)
    for j, tj in enumerate(times):
        flags[tj:, j] = 1
    tp = flags[:, :J].sum(1) / J
    path = np.cumsum(rng.uniform(0, [0.02, 0.1, 0.05], (T, 3)), 0)
    sample = t % 20 == 0
    ecc, scc, sc = (np.cumsum(sample * rng.integers(0, 3, T)) for _ in range(3))
    rb = (rng.uniform(0, 2, (T, 3)) * (rng.random((T, 3)) < 0.15)).round() / 20
    njerk = 2.0 * np.maximum(t - 1, 0)
    jsum = np.cumsum(np.where(t >= 2, rng.uniform(0, 300, T), 0.0))
    csum = np.cumsum(np.where(t >= 2, rng.uniform(0, 60, T), 0.0))
    svel, sz = np.cumsum(rng.uniform(0, 1, T)), np.cumsum(rng.uniform(0, 0.05, T))
    n = t + 1.0
    ev = np.stack([jsum, csum, njerk, svel, n, sz, n, n], 1)
    m_run = np.zeros((T, 13), np.float32)
    m_run[-1, 0] = float(success)
    m_run[:, 1] = tp
    m_run[:, 2:5] = path
    m_run[:, 5] = n / 20
    m_run[:, 6], m_run[:, 7], m_run[:, 8] = scc, ecc, sc
    m_run[:, 9] = jsum / np.maximum(njerk, 1)
    m_run[:, 10] = csum / np.maximum(njerk, 1)
    m_run[:, 11], m_run[:, 12] = svel / n, sz / n
    calls = t % 10 == 0
    arrays = {
        "state": rng.normal(size=(T, 36)).astype(np.float32),
        "actions": rng.normal(size=(T, 14)).astype(np.float32),
        "sc": sc.astype(np.int32),
        "m_run": m_run, "ev_acc": ev, "stage_flags": flags, "rb_counts": rb.astype(np.float32),
        "pred_success": _held(rng.uniform(0, 1, T), calls).astype(np.float32),
        "pred_stage": _held(rng.uniform(0, 1, (T, 5)), calls).astype(np.float32),
        "pred_togo": _held(rng.uniform(0, 2, (T, 11)), calls).astype(np.float32),
        "pred_completion": _held(np.clip(t / T + rng.normal(0, 0.05, T), 0, 1), calls).astype(np.float32),
    }
    images = {c: [encode_jpeg(rng.integers(0, 255, (img, img, 3), dtype=np.uint8)) for _ in range(T)] for c in CAMS}
    metrics = {k: float(m_run[-1, i]) for i, k in enumerate(sr.METRIC13)} | {"tl": float(T)}
    info = {"family": FAMILY, "variation": "PositionAndOrientation" if seed % 3 else "Position", "seed": seed,
            "success": bool(success), "termination": "success" if success else ("fail" if seed % 2 else "budget"),
            "source": "rollout", "tp": float(tp[-1]), "steps": T, "budget_steps": BUDGET, "stage_count": J,
            "policy": {"path": "runs/x/policies/step_4000", "step": 4000}, "metrics": metrics, "recording_ok": True}
    return arrays, images, info


def _true_adv(a, info):
    """A as compute_advantages.py computes it: completion shaping in the reward, the dampened baseline
    α·(Vo − Φ_t + Vb) of the shaped return (no intercept), terminal 0."""
    incr = sr.step_increments(a, info)
    C = sr.ema_hold(a["pred_completion"], EMA)
    r = sr.step_rewards(incr, G) + sr.completion_rewards(C, info["success"], CW)
    preds = {k: sr.ema_hold(a[f"pred_{k}"], EMA) for k in ("success", "stage", "togo")}
    fb = np.concatenate([np.zeros((1, 5)), a["stage_flags"][:-1]], 0)
    Vo, Vb = sr.value_blocks(preds, fb, G, FAMILY, J, CW)
    V = ALPHA["alpha"] * (Vo - CW * np.nan_to_num(C) + Vb)
    return sr.gae(r, V, LAM)


def make_run(root: Path, n_eps=2, T=30, seed=0) -> Path:
    """<root>/state/rollouts/run0/stores/w00 (recorded arrays + advantage) plus alpha.json / adv_meta.json; returns
    the run dir."""
    rng = np.random.default_rng(seed)
    run = root / "state" / "rollouts" / "run0"
    w = StoreWriter(run / "stores" / "w00", cameras=CAMS, info={"source": "rollout"})
    eps = []
    for i in range(n_eps):
        arrays, images, info = _episode(rng, T + 3 * i, success=i % 2 == 0, seed=1_000_000 + i)
        w.add_episode(arrays, images, info)
        eps.append((arrays, info))
    w.close()
    As = np.concatenate([_true_adv(a, info) for a, info in eps])
    mu, sd = float(As.mean()), float(As.std())
    write_array(run / "stores" / "w00", "advantage", np.clip((As - mu) / sd, -2, 2).astype(np.float32))
    (run / "adv_meta.json").write_text(json.dumps({"sigma": sd, "lam": LAM, "round": ROUND, "g_hash": "t", "clip": 2.0,
                                                   "ema": EMA, "completion_weight": CW, "togo_lam": 0.98,
                                                   "norm_by_task": {FAMILY: [mu, sd]}}))
    (run / "alpha.json").write_text(json.dumps({FAMILY: {**ALPHA, "single": 0.7, "n_eps": n_eps, "g": G.tolist()}}))
    return run
