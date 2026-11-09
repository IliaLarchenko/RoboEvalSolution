"""Unit tests for data/score_reward.py. CPU only, pure numpy.

    .venv-sim/bin/python -m pytest tests/test_score_reward.py      # also checks against the official scorer
    .venv/bin/python -m pytest tests/test_score_reward.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.data import score_reward as sr  # noqa: E402

FAMS = list(sr.BUDGET_SECONDS)


def _official():
    try:
        from roboeval.challenge import scoring
        return scoring
    except Exception:
        return None


def _random_metrics(rng, fam, scale=(0.5, 3.0), success=None):
    ref = sr.human_reference()[fam]
    m = {"success": float(rng.random() < 0.5) if success is None else float(success), "tp": float(rng.random())}
    for k in sr.BEHAVIOUR:
        base = ref[k] if ref[k] > 0 else 1.0
        m[k] = float(base * rng.uniform(*scale))
    return m


def test_constants_match_official():
    scoring = _official()
    if scoring is None:
        pytest.skip("roboeval not importable in this venv")
    assert sr.OUTCOME_SPLIT == scoring.OUTCOME_SPLIT
    assert sr.WEIGHTS == scoring.WEIGHTS
    assert {k: tuple(v) for k, v in sr.FAMILIES.items()} == {k: tuple(v) for k, v in scoring.FAMILIES.items()}
    assert tuple(sr.COUNT_METRICS) == tuple(scoring.COUNT_METRICS)
    assert sr.RELATIVE_FLOOR == scoring.RELATIVE_FLOOR
    assert sr.human_reference() == scoring.human_reference()
    from roboeval.challenge import spec
    assert {k: v for k, v in spec.RAW_METRIC_KEYS.items() if k != "ct"} == {k: sr.RAW_KEYS[k] for k in spec.RAW_METRIC_KEYS if k != "ct"}
    assert {f.key: f.budget_seconds for f in spec.TASK_FAMILIES} == sr.BUDGET_SECONDS


def _score(fam, m):
    scoring = _official()
    if scoring is not None:
        return scoring.task_scores(fam, {**m, "tl": 0.0})["score"]
    return sr.task_score_and_grad(fam, m, None)[0]


def test_score_grad():
    rng = np.random.default_rng(0)
    n_checked = 0
    for i in range(200):
        fam = FAMS[i % len(FAMS)]
        ref = sr.human_reference()[fam]
        m = _random_metrics(rng, fam)
        # away from kinks: every behaviour metric at least 2% from its reference
        if any(abs(m[k] - ref[k]) < 0.02 * max(ref[k], 1e-3) for k in sr.BEHAVIOUR):
            continue
        S, g = sr.task_score_and_grad(fam, m, outcome_override=None)
        assert abs(S - _score(fam, m)) < 1e-12
        for j, k in enumerate(sr.METRIC13):
            h = 1e-6 * max(abs(m[k]), 1e-3)
            mp, mm = dict(m), dict(m)
            mp[k] += h
            mm[k] -= h
            fd = (_score(fam, mp) - _score(fam, mm)) / (2 * h)
            assert abs(fd - g[j]) <= 1e-4 * max(abs(fd), 1e-6) + 1e-9, (fam, k, fd, g[j])
        _, g1 = sr.task_score_and_grad(fam, m, outcome_override=1.0)
        O = 0.7 * m["success"] + 0.3 * m["tp"]
        if O > 1e-3:
            np.testing.assert_allclose(g1[2:] * O, g[2:], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(g1[:2], g[:2])
        n_checked += 1
    assert n_checked > 100


def test_bootstrap_smooth():
    fam = "LiftPot"
    ref = sr.human_reference()[fam]
    k = "cpl"
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.15, 60)
    gs = []
    sweep = np.linspace(0.7, 1.3, 61) * ref[k]
    for c in sweep:
        eps = []
        for e in range(60):
            m = {"success": 1.0, "tp": 1.0, **{q: ref[q] * 1.1 + 0.01 for q in sr.BEHAVIOUR}}
            m[k] = c * (1 + noise[e])
            eps.append({"metrics": m, "success": True, "variation": "PositionAndOrientation"})
        g = sr.score_weights({fam: eps}, n_boot=400, seed=0)[fam]
        gs.append(g[sr.MI[k]])
    gs = np.array(gs)
    jumps = np.abs(np.diff(gs))
    # Unsmoothed, ∇S jumps from 0 to φ'(r+) at the reference kink; the bootstrap over 50-episode means
    # spreads that over the sampling width of the mean, so no 1%-of-ref step carries most of the jump.
    kink = abs(sr.task_score_and_grad(fam, {**{q: ref[q] * 1.1 + 0.01 for q in sr.BEHAVIOUR}, k: ref[k] * 1.0001,
                                            "success": 1.0, "tp": 1.0})[1][sr.MI[k]])
    assert jumps.max() < 0.35 * kink, (jumps.max(), kink)


def synthetic_episode(rng, fam="LiftPot", T=120, termination="budget", success=False, budget=None, path=0.05):
    """Consistent per-frame arrays (running metrics, accumulators, flags, RB counts) for one episode."""
    J = sr.STAGE_COUNT[fam]
    m_run = np.zeros((T, 13), np.float32)
    for k in ("cpl", "jpl", "opl"):
        m_run[:, sr.MI[k]] = np.cumsum(rng.uniform(0, path, T))
    for k in ("scc", "ecc", "sc"):
        m_run[:, sr.MI[k]] = np.cumsum(rng.random(T) < 0.01)
    m_run[:, sr.MI["ct"]] = (np.arange(T) + 1) / 20
    flags = np.zeros((T, 5), np.uint8)
    for j in range(J):
        if rng.random() < 0.6:
            flags[rng.integers(0, T):, j] = 1
    if success:
        flags[-1, :J] = 1
        flags[-1:, :J] = 1
        m_run[-1, sr.MI["success"]] = 1.0
    m_run[:, sr.MI["tp"]] = flags[:, :J].sum(1) / J
    acc = np.zeros((T, 8))
    acc[:, 0] = np.cumsum(rng.uniform(0, 50, T))
    acc[:, 1] = np.cumsum(rng.uniform(0, 10, T))
    acc[:, 2] = np.maximum(0, np.arange(T) * 2 - 4)
    acc[:, 3] = np.cumsum(rng.uniform(0, 0.5, T))
    acc[:, 4] = np.arange(T) + 1
    acc[:, 5] = np.cumsum(rng.uniform(0, 0.02, T))
    acc[:, 6] = np.arange(T) + 1
    acc[:, 7] = np.arange(T) + 1
    rb = (rng.random((T, 3)) < 0.05).astype(np.float32) / 20
    arrays = {"m_run": m_run, "ev_acc": acc, "stage_flags": flags, "rb_counts": rb}
    ep = {"family": fam, "termination": "success" if success else termination, "stage_count": J,
          "budget_steps": budget or int(sr.BUDGET_SECONDS[fam] * 20), "success": success}
    return arrays, ep


def test_fail_fine():
    rng = np.random.default_rng(2)
    arrays, ep = synthetic_episode(rng, T=100, termination="fail")
    budget = ep["budget_steps"]
    plain = sr.step_increments(arrays, {**ep, "termination": "budget"})  # the same episode, not fined
    fined = sr.step_increments(arrays, ep)
    R = budget - 100
    d = fined.sum(0) - plain.sum(0)
    assert np.isclose(d[sr.MI["ct"]], R / 20)
    for k in ("cpl", "jpl", "opl", "scc", "ecc", "sc"):
        assert np.isclose(d[sr.MI[k]], R * plain[:, sr.MI[k]].sum() / 100)
    for k in ("success", "tp", "mjj", "mcj", "bavd", "bgvd"):
        assert d[sr.MI[k]] == 0
    assert np.array_equal(fined[:-1], plain[:-1])
    for term, succ in (("budget", False), ("success", True)):
        a2, e2 = synthetic_episode(np.random.default_rng(3), T=100, termination=term, success=succ)
        np.testing.assert_array_equal(sr.step_increments(a2, e2),  # = no budget left (never fined)
                                      sr.step_increments(a2, {**e2, "budget_steps": 100}))
    # a branched episode (t0 env steps replayed before frame 0) is fined only for the budget left after t0 + T
    t0 = budget - 130
    branched = sr.step_increments(arrays, {**ep, "t0": t0})
    d = branched.sum(0) - plain.sum(0)
    assert np.isclose(d[sr.MI["ct"]], (budget - t0 - 100) / 20)
    assert np.isclose(d[sr.MI["cpl"]], 30 * plain[:, sr.MI["cpl"]].sum() / 100)
    np.testing.assert_array_equal(sr.step_increments(arrays, {**ep, "t0": budget - 100}), plain)
    # exactness without fine / RB: column sums = final running metrics
    a3, e3 = synthetic_episode(np.random.default_rng(4), T=90)
    inc = sr.step_increments({k: v for k, v in a3.items() if k != "rb_counts"}, e3)
    tot = inc.sum(0)
    for k in ("cpl", "jpl", "opl", "scc", "ecc", "sc", "tp"):
        assert np.isclose(tot[sr.MI[k]], a3["m_run"][-1, sr.MI[k]], atol=1e-5)
    assert np.isclose(tot[sr.MI["mjj"]], a3["ev_acc"][-1, 0] / a3["ev_acc"][-1, 2])
    assert np.isclose(tot[sr.MI["bgvd"]], a3["ev_acc"][-1, 5] / a3["ev_acc"][-1, 6])
    assert np.isclose(tot[sr.MI["ct"]], 90 / 20)


def test_togo_value_inverse():
    rng = np.random.default_rng(5)
    for fam in FAMS:
        for succ in (False, True):
            arrays, ep = synthetic_episode(rng, fam=fam, T=150, success=succ, path=0.01,
                                           termination="budget" if not succ else "success")
            incr = sr.step_increments(arrays, ep)
            g = sr.task_score_and_grad(fam, _random_metrics(rng, fam, (1.05, 2.0), success=1))[1]
            G = sr.returns_to_go(sr.step_rewards(incr, g))
            togo = sr.togo_targets(incr, fam)
            assert togo.max() < sr.TOGO_CAP  # unclipped for this check
            J = ep["stage_count"]
            final = arrays["stage_flags"][-1] * (np.arange(5) < J)  # final stage flags, j < J
            T = len(incr)
            preds = {"success": np.full(T, float(succ)), "stage": np.tile(final, (T, 1)), "togo": togo[:T]}
            fb = sr.flags_before(arrays["stage_flags"])
            Vo, Vb = sr.value_blocks(preds, fb, g, fam, J)
            np.testing.assert_allclose(Vo + Vb, G, rtol=1e-9, atol=1e-9)


def test_stage_frame_targets():
    """Stage-head targets: 1 while the stage is ahead and for STAGE_GRACE frames after it is reached, then 0;
    0 throughout when never reached or already set at reset; on the frames the value reads (not reached before t)
    they equal the final stage flags."""
    T, J, G = 60, 4, sr.STAGE_GRACE
    flags = np.zeros((T, 5), np.uint8)
    flags[10:, 0] = 1  # reached by action 10
    flags[50:, 2] = 1  # reached near the end (grace runs past T)
    flags[:, 3] = 1  # set at reset (branched episode)
    flags[:, 4] = 1  # j >= J
    first = sr.stage_first_reach(flags, J, reset_flags=[0, 0, 0, 1, 0])
    assert first[0] == 10 and first[2] == 50
    assert first[1] == first[3] == first[4] == sr.NOT_AHEAD
    y = sr.stage_frame_targets(first, np.arange(T))
    assert y.shape == (T, 5)
    np.testing.assert_array_equal(y[:, 0], (np.arange(T) <= 10 + G).astype(np.float32))
    assert y[:, 2].all() and not y[:, 1].any() and not y[:, 3].any() and not y[:, 4].any()
    np.testing.assert_array_equal(sr.stage_frame_targets(first, 31), y[31])
    final = flags[-1] * (np.arange(5) < J)
    open_ = 1 - sr.flags_before(flags, [0, 0, 0, 1, 0])
    np.testing.assert_array_equal((y * open_)[:, :3], (np.tile(final, (T, 1)) * open_)[:, :3])


def test_gae():
    rng = np.random.default_rng(6)
    r, V = rng.normal(size=50), rng.normal(size=50)
    G = sr.returns_to_go(r)
    np.testing.assert_allclose(sr.gae(r, V, lam=1.0), G - V, atol=1e-12)
    np.testing.assert_allclose(sr.gae(r, np.zeros(50), lam=1.0), G, atol=1e-12)
    lam, a = 0.9, 0.7
    A0 = sr.gae(r, np.zeros(50), lam)
    np.testing.assert_allclose(sr.gae(r, np.full(50, a), lam), A0 - a * lam ** (49 - np.arange(50)), atol=1e-12)


def test_fit_alpha():
    rng = np.random.default_rng(7)
    n_eps, T = 200, 30
    ep_id = np.repeat(np.arange(n_eps), T)
    V = np.repeat(rng.normal(0, 1.0, n_eps), T) + rng.normal(0, 0.3, n_eps * T)
    G = V + rng.normal(0, 0.5, n_eps * T)
    sig_e = 0.8
    Vo = V + rng.normal(0, sig_e, n_eps * T)
    Vb = 0.0 * Vo
    out = sr.fit_alpha(G, Vo, Vb, ep_id, prior={"alpha": 0.5}, prior_episodes=0, n_boot=200)
    sv = np.var(V)
    expect = sv / (sv + sig_e ** 2)
    lo, hi = out["ci_alpha"]
    assert lo - 0.05 <= expect <= hi + 0.05, (expect, out)
    # centred: a constant offset of the return moves only the intercept, not α
    shifted = sr.fit_alpha(G + 3.0, Vo, Vb, ep_id, prior={"alpha": 0.5}, prior_episodes=30, n_boot=0)
    ref = sr.fit_alpha(G, Vo, Vb, ep_id, prior={"alpha": 0.5}, prior_episodes=30, n_boot=0)
    assert abs(shifted["alpha"] - ref["alpha"]) < 1e-9 and abs(shifted["a"] - ref["a"] - 3.0) < 1e-9
    # few episodes -> prior
    few = sr.fit_alpha(G[: 5 * T], Vo[: 5 * T], Vo[: 5 * T] * 0.3 + rng.normal(0, .1, 5 * T), ep_id[: 5 * T],
                       prior={"alpha": 0.42}, n_boot=0)
    assert "few_episodes" in few["guards"] and few["alpha"] == 0.42
    # strong prior shrinks toward it
    strong = sr.fit_alpha(G, Vo, Vb, ep_id, prior={"alpha": 0.9}, prior_episodes=10 ** 6, n_boot=0)
    assert abs(strong["alpha"] - 0.9) < 0.02
    # clip at 1 (α* = ρ² ≤ 1)
    hot = sr.fit_alpha(G * 5, Vo, Vb, ep_id, prior={"alpha": 0.5}, prior_episodes=0, n_boot=0)
    assert hot["alpha"] == 1.0 and hot["single"] > 1.0
    assert out["r2_fit"] >= max(0.0, out["r2_alpha1"]) - 1e-9


def test_linearisation():
    rng = np.random.default_rng(8)
    for fam in FAMS:
        n = 50
        eps = [_random_metrics(rng, fam, (0.8, 2.0), success=rng.random() < 0.6) for _ in range(n)]
        M = np.stack([sr._as_vec(m) for m in eps])
        mbar = M.mean(0)
        _, g = sr.task_score_and_grad(fam, mbar, outcome_override=None)
        S = sr.task_score_and_grad(fam, mbar, None)[0]
        dS = np.array([S - sr.task_score_and_grad(fam, np.delete(M, i, 0).mean(0), None)[0] for i in range(n)])
        R = M @ g
        assert np.corrcoef(dS, R - R.mean())[0, 1] > 0.99


def test_store_roundtrip(tmp_path):
    from roboeval_solution.data.store import StoreReader, StoreWriter, write_array

    rng = np.random.default_rng(9)
    w = StoreWriter(tmp_path / "s", cameras=["head"], fps=20)
    lens = [30, 45]
    for i, T in enumerate(lens):
        arrays, ep = synthetic_episode(rng, T=T)
        arrays.update(pred_success=rng.random(T).astype(np.float32),
                      pred_stage=rng.random((T, 5)).astype(np.float32), pred_togo=rng.random((T, 11)).astype(np.float32),
                      pred_completion=rng.random(T).astype(np.float32))
        w.add_episode(arrays, {"head": [b"x"] * T}, {"family": "LiftPot", **{k: v for k, v in ep.items() if k != "family"}})
    w.close()
    r = StoreReader(tmp_path / "s")
    assert r.arrays["m_run"].shape == (sum(lens), 13) and r.arrays["ev_acc"].dtype == np.float64
    assert r.arrays["pred_togo"].shape == (sum(lens), 11) and r.arrays["stage_flags"].dtype == np.uint8
    adv = np.concatenate([np.full(T, i, np.float32) for i, T in enumerate(lens)])
    write_array(tmp_path / "s", "advantage", adv)
    r2 = StoreReader(tmp_path / "s")
    np.testing.assert_array_equal(r2.arrays["advantage"], r2.arrays["episode_index"].astype(np.float32))
