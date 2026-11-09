"""compute_advantages.py end to end on a synthetic recorded run: per-task centring / scaling and λ in adv_meta;
the dataset's per-variation demo advantage."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from roboeval_solution.data.store import StoreReader
from synthetic_run import make_run

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compute_advantages.py"


def test_per_task_normalisation(tmp_path):
    run = make_run(tmp_path, n_eps=12, T=40)  # its advantage.npy is overwritten
    state = tmp_path / "adv_state"
    subprocess.run([sys.executable, str(SCRIPT), "--pool", str(run), "--newest", str(run), "--state", str(state),
                    "--n-boot", "20", "--clip", "100", "--pos-threshold", "0"], check=True, capture_output=True)
    meta = json.loads((run / "adv_meta.json").read_text())
    assert meta["lam"] == 0.95 and meta["norm_by_task"]
    adv = StoreReader(run / "stores" / "w00").arrays["advantage"].astype(np.float64)
    assert abs(adv.mean()) < 1e-5 and abs(adv.std() - 1.0) < 1e-4  # one task: centred, unit scale
    diag = json.loads((state / "advantages_round_0.json").read_text())
    assert np.isfinite(diag["tasks"]["LiftPot"]["succ_last20_mean_adv"])  # how the finishing frames are labelled


def test_tp_weight_and_pos_threshold(tmp_path):
    """--tp-weight scales g_tp in the reward / value only (the logged g stays leaderboard-exact);
    --pos-threshold shifts the normalised advantage down before the clip."""
    run = make_run(tmp_path, n_eps=12, T=40)
    out = {}
    for name, extra in (("ref", ["--tp-weight", "1", "--pos-threshold", "0"]),
                        ("tp3", ["--tp-weight", "3", "--pos-threshold", "0"]),
                        ("tau", ["--tp-weight", "1", "--pos-threshold", "0.2"])):
        state = tmp_path / f"s_{name}"
        subprocess.run([sys.executable, str(SCRIPT), "--pool", str(run), "--newest", str(run), "--state", str(state),
                        "--n-boot", "20", "--clip", "100", *extra], check=True, capture_output=True)
        sw = json.loads((state / "score_weights.jsonl").read_text().splitlines()[-1])
        diag = json.loads((state / "advantages_round_0.json").read_text())
        adv = StoreReader(run / "stores" / "w00").arrays["advantage"].astype(np.float64)
        out[name] = (sw, diag, adv)
    (sw1, d1, a1), (sw3, d3, a3), (swt, dt, at) = out["ref"], out["tp3"], out["tau"]
    tp = 1  # METRIC13 index of tp
    assert sw1["g"] == sw3["g"] and sw3["tp_weight"] == 3.0  # logged g: exact
    assert d3["g"]["LiftPot"][tp] == pytest.approx(3 * d1["g"]["LiftPot"][tp])  # used g: scaled
    assert not np.allclose(a1, a3)
    np.testing.assert_allclose(at, a1 - 0.2, atol=1e-5)
    assert dt["pos_threshold"] == 0.2


def test_score_weights_ema_once_per_newest_collect(tmp_path):
    """Recomputes for the same newest collect (a recovery run arrived, a trainer restart) reuse its g; the EMA steps
    only when a new collect is the newest."""
    a, b = make_run(tmp_path / "a", n_eps=12, T=40), make_run(tmp_path / "b", n_eps=12, T=40, seed=5)
    state = tmp_path / "s"
    run = lambda newest, rnd, *pool: subprocess.run(
        [sys.executable, str(SCRIPT), "--pool", *map(str, pool), "--newest", str(newest), "--state", str(state),
         "--n-boot", "50", "--round", str(rnd)], check=True, capture_output=True)
    lines = lambda: [json.loads(l) for l in (state / "score_weights.jsonl").read_text().splitlines()]
    run(a, 1, a)
    run(a, 2, a)
    l1, l2 = lines()
    assert l2["reused"] == ["LiftPot"] and l2["g"] == l1["g"]  # same newest: no second EMA step
    run(b, 3, a, b)
    l3 = lines()[-1]
    assert l3["reused"] == [] and l3["g"] != l2["g"]  # a new newest collect: one EMA step


def test_value_target_and_terminal(tmp_path):
    """Dampened baseline of the shaped return (LeHome control variate): one α = Cov(G, V̂)/Var(V̂) of the shaped
    return-to-go G and V̂ = (Vo − Φ_t) + Vb (centred, capped at 1), the baseline α·V̂ without the intercept,
    terminal 0; so δ = (1 − α)·r' + α·(r' + ΔV̂) also on the last frame (no −a·λ^(T−t−1) tail)."""
    from roboeval_solution.data import score_reward as sr
    run = make_run(tmp_path, n_eps=12, T=40)
    state = tmp_path / "s"
    subprocess.run([sys.executable, str(SCRIPT), "--pool", str(run), "--newest", str(run), "--state", str(state),
                    "--n-boot", "20", "--clip", "100", "--pos-threshold", "0"], check=True, capture_output=True)
    fit = json.loads((run / "alpha.json").read_text())["LiftPot"]
    mu, sd = json.loads((run / "adv_meta.json").read_text())["norm_by_task"]["LiftPot"]
    r = StoreReader(run / "stores" / "w00")
    g = np.asarray(fit["g"])
    Gv, Vo_all, Vb_all, ids = [], [], [], []
    for i, e in enumerate(r.episodes):
        sl = slice(e["start"], e["start"] + e["length"])
        a = {k: v[sl] for k, v in r.arrays.items()}
        incr = sr.step_increments(a, e)
        C = sr.ema_hold(a["pred_completion"], 0.2)
        rr = sr.step_rewards(incr, g) + sr.completion_rewards(C, e["success"], 0.3)
        preds = {k: sr.ema_hold(a[f"pred_{k}"], 0.2) for k in ("success", "stage", "togo")}
        Vo, Vb = sr.value_blocks(preds, sr.flags_before(a["stage_flags"]), g, "LiftPot", e["stage_count"], 0.3)
        Vo = Vo - 0.3 * np.nan_to_num(C)
        V = fit["alpha"] * (Vo + Vb)
        A = r.arrays["advantage"][sl].astype(np.float64) * sd + mu
        np.testing.assert_allclose(A, sr.gae(rr, V, 0.95), atol=1e-4)
        Gv.append(sr.returns_to_go(rr))
        Vo_all.append(Vo), Vb_all.append(Vb), ids.append(np.full(e["length"], i))
    # the stored fit is the fit of that target (single run: prior = the default)
    ref = sr.fit_alpha(np.concatenate(Gv), np.concatenate(Vo_all), np.concatenate(Vb_all), np.concatenate(ids),
                       prior={"alpha": 0.5}, prior_episodes=30, n_boot=0)
    for k in ("a", "alpha"):
        assert abs(ref[k] - fit[k]) < 1e-6, k


def test_shaping_reward_identity_and_td_togo():
    from roboeval_solution.data import score_reward as sr
    rng = np.random.default_rng(0)
    T = 50
    C = np.sort(rng.uniform(0, 1, T))
    F = sr.completion_rewards(C, True, 0.3)
    assert np.isclose(F.sum(), 0.3 - 0.3 * C[0])  # telescopes to Φ_T − Φ_0
    incr = np.zeros((T, 13)); incr[:, sr.MI["ct"]] = 0.05; incr[::7, sr.MI["sc"]] = 1.0
    h = rng.uniform(0, 1, (T, 11))
    np.testing.assert_allclose(sr.togo_td_targets(incr, "LiftPot", h, lam=1.0),
                               sr.togo_targets(incr, "LiftPot")[:T], atol=1e-12)  # λ = 1: Monte Carlo
    td0 = sr.togo_td_targets(incr, "LiftPot", h, lam=0.0)
    c = incr[:, [sr.MI[k] for k in sr.TOGO_METRIC]] / sr.togo_scale("LiftPot")
    np.testing.assert_allclose(td0[:-1], np.clip(c[:-1] + h[1:], 0, sr.TOGO_CAP), atol=1e-12)  # λ = 0: one step


def test_completion_on_failures_and_td_targets_in_dataset():
    from roboeval_solution.data import value_targets as vt
    from test_value_targets import _values  # noqa: F401  (keeps the helper importable)
    rng = np.random.default_rng(1)
    T = 40
    arrays = {"m_run": np.zeros((T, 13)), "ev_acc": np.ones((T, 8)), "stage_flags": np.zeros((T, 5)),
              "rb_counts": np.zeros((T, 3))}
    ep = {"family": "LiftPot", "termination": "budget", "success": False, "recording_ok": True, "budget_steps": 200}
    ft = vt.frame_targets(vt.episode_value_arrays(ep, arrays, 0, T), 5, T)
    assert ft["completion_target"] == 0.0 and ft["completion_future_target"] == 0.0
    td = rng.uniform(0, 2, (T, 11)).astype(np.float32)
    v = vt.episode_value_arrays(ep, {**arrays, "togo_td": td}, 0, T)
    np.testing.assert_allclose(v["togo"][:T], td, rtol=1e-6)
    assert (v["togo"][T] == 0).all()


def test_compute_advantages_writes_td_and_reward_shaping(tmp_path):
    run = make_run(tmp_path, n_eps=6, T=40)
    subprocess.run([sys.executable, str(SCRIPT), "--pool", str(run), "--newest", str(run), "--state",
                    str(tmp_path / "s"), "--n-boot", "20"], check=True, capture_output=True)
    meta = json.loads((run / "adv_meta.json").read_text())
    assert meta["togo_lam"] == 0.995
    td = StoreReader(run / "stores" / "w00").arrays["togo_td"]
    assert td.shape[1] == 11 and np.isfinite(td).all() and (td >= 0).all()


def test_increments_from_branch_point_and_recovery_source(tmp_path):
    from roboeval_solution.data import score_reward as sr
    from roboeval_solution.training import flywheel
    from roboeval_solution.data.dataset import StoreSource
    T = 10
    m_run = np.zeros((T, 13)); m_run[:, sr.MI["cpl"]] = 2.0 + 0.1 * np.arange(1, T + 1)  # 2.0 run before the branch
    arrays = {"m_run": m_run, "ev_acc": np.ones((T, 8)), "stage_flags": np.zeros((T, 5)), "rb_counts": np.zeros((T, 3))}
    ep = {"family": "LiftPot", "termination": "budget", "budget_steps": 200}
    assert np.isclose(sr.step_increments(arrays, ep)[0, sr.MI["cpl"]], 2.1)  # from 0: the whole pre-branch path
    m0 = np.zeros(13); m0[sr.MI["cpl"]] = 2.0
    assert np.isclose(sr.step_increments(arrays, {**ep, "m_run_reset": m0.tolist()})[0, sr.MI["cpl"]], 0.1)


def test_recovery_runs_are_sources_of_their_own(tmp_path):
    from roboeval_solution.data.dataset import StoreSource
    from roboeval_solution.training import flywheel
    run = tmp_path / "runs" / "x"
    for d in ("rollouts/c0_0001_step_000500", "rollouts/c0_0002_step_001000", "recovery/r0_0001_step_001500",
              "recovery/r0_0002_step_001500", "recovery/again"):
        (run / d / "stores").mkdir(parents=True); (run / d / "DONE").touch()
    # ordered by the collect recovery.json names, whatever the run is called
    for d, c in (("r0_0001_step_001500", "c0_0001_step_000500"), ("r0_0002_step_001500", "c0_0002_step_001000"),
                 ("again", "c0_0002_step_001000")):
        (run / "recovery" / d / "recovery.json").write_text(json.dumps({"collect": c}))
    (run / "recovery" / "c0_0003_step_001500" / "stores").mkdir(parents=True)  # incomplete: no DONE
    assert [Path(r).name for r in flywheel.recovery_runs(run, 2)] == ["again", "r0_0002_step_001500"]  # by collect
    pool = flywheel.current_pool(run, 1)  # collect 1 left the pool: its recovery run still trains (own window)
    cfg = flywheel.DataRefreshConfig(run_dir=str(run))
    src = flywheel.build_sources((StoreSource(paths=("demos",), share=1.0, name="demos"),), cfg, pool, 0)
    by = {s.name: s for s in src}
    assert set(by) == {"demos", "rollouts_c0_0002_step_001000", "recovery_r0_0001_step_001500",
                       "recovery_r0_0002_step_001500", "recovery_again"}
    assert by["recovery_again"].paths == (str(run / "recovery/again/stores"),) and by["recovery_again"].prioritized


def test_extra_recovery_runs_get_pool_advantages(tmp_path):
    run = make_run(tmp_path / "a", n_eps=8, T=40)
    rec = make_run(tmp_path / "b", n_eps=4, T=30, seed=3)
    ej = rec / "stores" / "w00" / "episodes.json"
    eps = json.loads(ej.read_text())
    for e in eps:
        e["source"], e["success"] = "recovery", True
    ej.write_text(json.dumps(eps))
    (rec / "recovery.json").write_text(json.dumps({"collect": run.name}))  # its α: that collect's fit
    before = StoreReader(rec / "stores" / "w00").arrays["advantage"].copy()
    meta_before = (rec / "adv_meta.json").read_text()
    out = lambda *x: subprocess.run([sys.executable, str(SCRIPT), "--pool", str(run), "--newest", str(run), *x,
                                     "--state", str(tmp_path / "s"), "--n-boot", "20"], check=True, capture_output=True)
    out()
    base = json.loads((run / "adv_meta.json").read_text())
    out("--extra", str(rec))
    meta = json.loads((run / "adv_meta.json").read_text())
    assert meta["norm_by_task"] == base["norm_by_task"] and meta["sigma"] == base["sigma"]  # extras never enter fits
    adv = StoreReader(rec / "stores" / "w00").arrays["advantage"]
    assert np.isfinite(adv).all() and not np.allclose(adv, before)  # recomputed from the pool's fits
    assert (rec / "adv_meta.json").read_text() == meta_before  # the recovery run's own files untouched


def test_recovery_cascade_chain():
    sys.path.insert(0, str(SCRIPT.parent))
    import rollout as ro
    from roboeval_solution.sim import recovery as rec
    acts = np.zeros((300, 14), np.float32)
    fails = [{"family": "LiftPot", "variation": "PositionAndOrientation", "seed": 5, "budget": 400, "actions": acts,
              "t_peak": 70, "t_drop": 120, "dropped": d, "origin": {"episode": i, "target": "failure"}}
             for i, d in enumerate((False, True))]
    c = object.__new__(ro.RecoveryCascade)
    c.rec, c.fails, c.rng = rec, fails, np.random.default_rng(0)
    c.stages, c.params = ["drop", "perturb", "reverse", "peak", "back2s"], {}
    c.chains = [(i, list(c.stages)) for i in range(len(fails))]  # one chain per target (no "|" groups)
    c.pos, c.state, c.tries = [0, 0], ["open", "open"], [[], []]
    c.tried = {s: [0, 0] for s in c.stages}
    c.n_job = c.n_done = c.last_log = 0
    jobs = c.initial_jobs()
    assert [j["scenario"] for j in jobs] == ["drop", "peak"]  # the dropped object skips the at-failure stages
    nxt = c.result({**{k: v for k, v in jobs[0].items() if k != "prefix"}, "success": False})
    assert nxt["scenario"] == "perturb" and len(nxt["prefix"]) == 120 + 10
    assert c.result({**{k: v for k, v in nxt.items() if k != "prefix"}, "success": True}) is None
    assert c.state[0] == "recovered" and c.tried["perturb"] == [1, 1]
    last = c.result({**{k: v for k, v in jobs[1].items() if k != "prefix"}, "success": False})
    assert last["scenario"] == "back2s" and last["branch_t"] == 80
    assert c.result({**{k: v for k, v in last.items() if k != "prefix"}, "success": False, "metrics": {"tp": 0.5}}) \
        is None
    assert c.state == ["recovered", "exhausted"] and c.n_done == 2
    left = c.left()
    assert [x["seed"] for x in left] == [5] and [t["stage"] for t in left[0]["tries"]] == ["peak", "back2s"]
    assert left[0]["tries"][1] == {"stage": "back2s", "success": False, "tp": 0.5}
    # a target with no drop to branch at: no stage applies (track targets come from failed_episodes)
    none = {**fails[0], "t_peak": None, "t_drop": None, "detector": "none"}
    stages = ("drop", "perturb", "reverse", "peak", "back2s", "retry", "track")
    assert [st for st in stages if rec.stage_prefix(none, st, np.random.default_rng(0)) is not None] == []
    # retry: the family's scripted policy from the failed state, its own parameter draw per target (a collect rolls
    # each seed out several times: two targets on one seed are not the same deterministic episode twice)
    pre, t_b = rec.stage_prefix(fails[1], "retry", np.random.default_rng(0))
    scr = rec.make_job(fails[1], "retry", pre, t_b, {}, chain=1)
    assert scr["scripted"] == "liftpot" and scr["retry"] and t_b == 120 and len(scr["prefix"]) == 120
    assert scr["script_seed"] == rec.script_seed(5, 1) != rec.script_seed(5, 0)
    # acceptance at any stage: complete only, whether the target failed or missed a stage
    for st in ("drop", "back2s", "retry"):
        assert not rec.counts({"success": True, "metrics": {"tp": 0.5}, "scenario": st,
                               "origin": {"detector": "stage1", "success": False, "target": "failure"}})
        assert not rec.counts({"success": True, "metrics": {"tp": 0.5}, "scenario": st,
                               "origin": {"detector": "none", "success": True, "target": "stages"}})
        assert rec.counts({"success": True, "metrics": {"tp": 1.0}, "scenario": st,
                           "origin": {"success": True, "target": "stages"}})


def test_recovery_complete_only():
    sys.path.insert(0, str(SCRIPT.parent))
    import rollout as ro
    from roboeval_solution.sim import recovery as rec
    assert rec.complete({"success": True, "metrics": {"tp": 1.0}})
    assert not rec.complete({"success": True, "metrics": {"tp": 2 / 3}})  # success that skipped a stage
    assert not rec.complete({"success": False, "metrics": {"tp": 1.0}})
    acts = np.zeros((300, 14), np.float32)
    fails = [{"family": "StackTwoBlocks", "variation": "PositionAndOrientation", "seed": 5, "budget": 400,
              "actions": acts, "t_peak": 70, "t_drop": 120, "dropped": False, "detector": "stage1",
              "origin": {"episode": 0, "success": True, "detector": "stage1", "target": "stages"}}]
    c = object.__new__(ro.RecoveryCascade)
    c.rec, c.fails, c.rng = rec, fails, np.random.default_rng(0)
    c.stages, c.params = ["drop", "perturb", "reverse", "peak", "back2s", "retry", "track"], {}
    c.chains = [(0, list(c.stages))]
    c.pos, c.state, c.tries = [0], ["open"], [[]]
    c.tried = {s: [0, 0] for s in c.stages}
    c.n_job = c.n_done = c.last_log = 0
    job = c.initial_jobs()[0]
    nxt = c.result({**{k: v for k, v in job.items() if k != "prefix"}, "success": True, "metrics": {"tp": 2 / 3}})
    assert nxt is not None and c.state == ["open"]  # incomplete success: the ladder continues
    assert c.result({**{k: v for k, v in nxt.items() if k != "prefix"}, "success": True, "metrics": {"tp": 1.0}}) is None
    assert c.state == ["recovered"] and c.summary()["by_detector"]["stages: stage1"] == {"targets": 1, "recovered": 1}
    # failure targets: complete only
    assert not rec.counts({"success": True, "metrics": {"tp": 0.5},
                           "origin": {"detector": "success", "target": "failure"}})
    assert rec.counts({"success": True, "metrics": {"tp": 1.0}, "origin": {"detector": "success", "target": "failure"}})
    assert not rec.counts({"success": True, "metrics": {"tp": 0.5},
                           "origin": {"detector": "stage0", "success": True, "target": "stages"}})
    assert rec.counts({"success": True, "metrics": {"tp": 1.0}, "origin": {"detector": "stage0", "target": "stages"}})


def _recovery_collect(root, episodes):
    """A collect store with the arrays find_failures reads: (success, tp, score, P̂ success curve) per episode; the
    hands keep moving with open grippers (no stall event)."""
    from roboeval_solution.data.store import StoreWriter
    w = StoreWriter(root / "stores" / "w00", cameras=[], info={"source": "rollout"})
    for i, (success, tp, score, ps) in enumerate(episodes):
        T = len(ps)
        actions = np.zeros((T, 14), np.float32)
        actions[:, 0] = 0.01
        arrays = {"actions": actions, "pred_success": np.asarray(ps, np.float32),
                  "pred_stage": np.full((T, 5), 0.1, np.float32), "stage_flags": np.ones((T, 5), np.uint8),
                  "sc": np.zeros(T, np.int32), "state_gripper": np.ones((T, 2), np.float32),
                  "state_ee": np.zeros((T, 12), np.float32)}
        w.add_episode(arrays, {}, {"family": "LiftPot", "variation": "PositionAndOrientation", "seed": 20_000_000 + i,
                                   "success": success, "tp": tp, "metrics": {"tp": tp}, "score_ep": {"score": score},
                                   "recording_ok": True, "budget_steps": 200, "stage_count": 4})
    w.close()
    return root


def test_recovery_targets(tmp_path):
    from roboeval_solution.sim import recovery as rec
    # t_peak: the P̂ maximum in the 3 s (60 frames) before the drop, not an early-episode maximum
    p = np.r_[np.full(10, 0.95), np.full(100, 0.7), np.full(10, 0.8), np.full(30, 0.7), np.full(40, 0.1)]
    assert rec.find_drop(p, 0.25, 60, 400) == (110, 150)
    assert rec.find_drop(p, 0.25, 60, 400, peak_window=10_000) == (0, 150)  # a window over the whole prefix
    near = np.r_[np.full(100, 0.9), np.full(6, 0.6), np.full(40, 0.1)]  # peak right at the drop: no peak branch
    t_peak, t_drop = rec.find_drop(near, 0.25, 60, 400, peak_window=5)
    f = {"actions": np.zeros((146, 14), np.float32), "t_peak": t_peak, "t_drop": t_drop, "dropped": False,
         "budget": 400}
    assert t_drop - t_peak < 10 and rec.stage_prefix(f, "peak", np.random.default_rng(0)) is None
    drop = [0.2] * 30 + [0.8] * 50 + [0.1] * 80  # a P̂(success) drop at frame 80
    col = _recovery_collect(tmp_path / "c0_0001_step_000500", [
        (False, 0.5, 0.3, drop),  # failure with a drop
        (False, 0.5, 0.3, [0.2] * 160),  # failure without one: no target (no event, no drop, no env ending)
        (True, 1.0, 0.8, [0.9] * 160),  # complete: never a target
        (True, 1.0, 1.0, [0.9] * 160)])
    fails = rec.find_failures(col)
    assert [(f["detector"], f["t_drop"], f["origin"]["target"]) for f in fails] == [("success", 80, "failure")]
