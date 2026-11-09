"""Score-exact advantages for recorded rollout runs.

    python scripts/compute_advantages.py --pool ../runs/<name>/rollouts/c0_0003_x [...] \
        --newest ../runs/<name>/rollouts/c0_0005_y --state ../runs/<name> --round 5

Per pool run (oldest -> newest as given; --newest is added or moved last):
  reward    r_t = g_f · Δm_t: the RoboEval Diagnostic Score linearised at the current policy, with
            Rao-Blackwellised collision / slip counts and the fail fine. g_f is re-estimated from the newest runs'
            successful PR episodes and EMA'd once per newest collect (<state>/score_weights.jsonl); the
            subtask-progress weight g_tp is then multiplied by --tp-weight (> 1: successes that skip stages rank
            clearly below full ones; the logged g stays leaderboard-exact). Plus the completion shaping
            F_t = Φ_{t+1} − Φ_t (Φ = w·Ĉ, Φ_T = w·success).
  value     the dampened baseline (LeHome control variate) b_t = α·V̂_t of the shaped return-to-go G, from the
            collecting policy's own (EMA'd, out-of-sample) head predictions: V̂ = (Vo − Φ_t) + Vb, Vo the outcome block
            (expected success / stage score + terminal potential w·P̂), Vb the behaviour block. One
            α = Cov(G, V̂)/Var(V̂) per run and task (= ρ² for a calibrated V̂, so clipped to [0, 1]) from centred,
            episode-weighted moments with a ridge toward the previous run's α (<run>/alpha.json). No intercept in the
            baseline: a constant cancels in every TD error except the terminal one.
  advantage A = GAE_λ(r, b) with b_T = 0: δ_t = (1 − α)·r_t + α·(r_t + V̂_{t+1} − V̂_t), grounded rewards and the
            predicted value's change mixed by how much the heads are trusted; λ = --lam (0.95 ≈ 1 s: local credit).
            Per task centred and scaled by the newest run's frames, minus --pos-threshold (Ã ≥ 0 marks the frames the
            advantage token labels positive), clipped to ±--clip.
Writes arrays/advantage.npy (NaN for skipped episodes: error, recording not ok) and arrays/togo_td.npy (TD(--togo-lam)
behaviour-to-go targets bootstrapped on the recorded togo head; the dataset prefers them to Monte-Carlo togo) into
every pool store, <run>/adv_meta.json and <state>/advantages_round_<K>.json (diagnostics). --extra runs get
advantage.npy only.
numpy only (+ scipy through sim.recovery): runs in either venv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.data import score_reward as sr  # noqa: E402
from roboeval_solution.data.store import StoreReader, list_stores, write_array  # noqa: E402
from roboeval_solution.sim.recovery import source_collect  # noqa: E402

log = logging.getLogger("advantages")


class Episode:
    __slots__ = ("run", "reader", "meta", "start", "length", "family", "ok", "arrays", "order", "out")

    def __init__(self, run, reader, meta, order):
        self.run, self.reader, self.meta, self.order = run, reader, meta, order
        self.start, self.length = int(meta["start"]), int(meta["length"])
        self.family = meta["family"]
        self.ok = meta.get("termination") != "error" and bool(meta.get("recording_ok")) and self.length > 0
        self.arrays = None
        self.out = {}

    def slice(self):
        if self.arrays is None:
            sl = slice(self.start, self.start + self.length)
            self.arrays = {k: v[sl] for k, v in self.reader.arrays.items()}
        return self.arrays

    @property
    def success(self) -> bool:
        return bool(self.meta.get("success"))

    @property
    def metrics(self) -> dict:
        return self.meta["metrics"]


def load_pool(runs: list[Path]) -> tuple[list[Episode], dict]:
    eps, readers = [], {}
    skipped = defaultdict(int)
    n = len(runs)
    for i, run in enumerate(runs):
        roots = list_stores([run / "stores"])
        if not roots:
            log.warning("%s: no stores", run)
        for root in roots:
            r = StoreReader(root)
            readers[root] = r
            for e in r.episodes:
                ep = Episode(run, r, e, order=n - 1 - i)
                if not ep.ok:
                    skipped["error" if e.get("termination") == "error" else "recording_not_ok"] += 1
                eps.append(ep)
    return eps, {"skipped": dict(skipped), "readers": readers}


def per_episode(ep: Episode, g: np.ndarray, ema: float, cw: float = 0.3, togo_lam: float = 0.995):
    a, m = ep.slice(), ep.meta
    incr = sr.step_increments(a, m)
    r = sr.step_rewards(incr, g)
    C = sr.ema_hold(a["pred_completion"], ema)
    r = r + sr.completion_rewards(C, ep.success, cw)  # completion shaping is reward: in G, the value fit, the GAE
    G = sr.returns_to_go(r)
    J = int(m["stage_count"])
    preds = {"success": sr.ema_hold(a["pred_success"], ema), "stage": sr.ema_hold(a["pred_stage"], ema),
             "togo": sr.ema_hold(a["pred_togo"], ema)}
    fb = sr.flags_before(a["stage_flags"], m.get("stage_flags_reset"))
    Vo, Vb = sr.value_blocks(preds, fb, g, ep.family, J, cw)
    fined = (m.get("termination") in ("fail", "unhealthy")
             and int(m.get("t0") or 0) + ep.length < m["budget_steps"])
    togo_td = sr.togo_td_targets(incr, ep.family, a["pred_togo"], togo_lam)
    phi = cw * np.nan_to_num(np.asarray(C, np.float64))  # completion potential Φ_t (as in completion_rewards)
    # Vo - Φ_t: the outcome block of the shaped return-to-go G (module docstring)
    ep.out = {"incr": incr, "r": r, "G": G, "Vo": Vo - phi, "Vb": Vb, "C": C, "fined": fined, "preds": preds,
              "J": J, "togo_td": togo_td}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.r_[pos, neg])
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def diagnostics(eps_ok: list[Episode], newest_eps: list[Episode], g: dict, alphas: dict, adv_ep: dict,
                sigma: float, skipped: dict) -> dict:
    fams = sorted({e.family for e in eps_ok})
    d: dict = {"sigma": sigma, "skipped": skipped, "tasks": {}}
    for f in fams:
        allf = [e for e in eps_ok if e.family == f]
        new = [e for e in newest_eps if e.family == f]
        t: dict = {"episodes": len(allf), "newest_episodes": len(new),
                   "fined": int(sum(e.out["fined"] for e in allf)),
                   "success": float(np.mean([e.success for e in allf])) if allf else float("nan")}
        # Rao-Blackwellised vs official counts (episode totals, no fine)
        rb = np.array([e.slice()["rb_counts"].astype(np.float64).sum(0) for e in allf])
        off = np.array([[e.metrics.get(k, 0.0) for k in ("scc", "ecc", "sc")] for e in allf], np.float64)
        if len(allf) > 1:
            se = np.sqrt(off.var(0) / len(allf) + rb.var(0) / len(allf))
            t["rb_vs_official"] = {k: {"rb_mean": float(rb[:, i].mean()), "official_mean": float(off[:, i].mean()),
                                       "within_2se": bool(abs(rb[:, i].mean() - off[:, i].mean()) <= 2 * se[i] + 1e-12),
                                       "var_reduction": float(off[:, i].var() / rb[:, i].var()) if rb[:, i].var() > 0 else None}
                                   for i, k in enumerate(("scc", "ecc", "sc"))}
        # linearisation: leave-one-out score changes vs g·m on the newest run's PR episodes (official metrics)
        pr = [e for e in new if e.meta.get("variation") == "PositionAndOrientation"]
        if len(pr) >= 5:
            M = np.stack([sr._as_vec(e.metrics) for e in pr])
            S = sr.task_score_and_grad(f, M.mean(0), None)[0]
            dS = np.array([S - sr.task_score_and_grad(f, np.delete(M, i, 0).mean(0), None)[0] for i in range(len(pr))])
            R = M @ g[f]
            t["linearisation_corr"] = float(np.corrcoef(dS, R)[0, 1]) if np.std(dS) > 0 and np.std(R) > 0 else None
        # head quality on the newest run (out-of-sample)
        if new:
            p = np.concatenate([e.slice()["pred_success"] for e in new]).astype(np.float64)
            y = np.concatenate([np.full(e.length, float(e.success)) for e in new])
            ok = np.isfinite(p)
            if ok.any():
                t["p_success_auc"] = _auc(p[ok], y[ok])
                X = np.c_[np.ones(ok.sum()), p[ok]]
                t["p_success_calib_slope"] = float(np.linalg.lstsq(X, y[ok], rcond=None)[0][1]) if np.std(p[ok]) > 0 else None
            bces = []
            for e in new:  # against the stage head's per-frame targets (sr.stage_frame_targets)
                q = np.clip(e.slice()["pred_stage"].astype(np.float64), 1e-6, 1 - 1e-6)
                first = sr.stage_first_reach(e.slice()["stage_flags"], e.out["J"], e.meta.get("stage_flags_reset"))
                yf = sr.stage_frame_targets(first, np.arange(e.length))
                b = -(yf * np.log(q) + (1 - yf) * np.log(1 - q))
                b = b[:, : e.out["J"]]
                if np.isfinite(b).all():
                    bces.append(b.mean())
            t["stage_bce"] = float(np.mean(bces)) if bces else None
            tg_pred = np.concatenate([e.slice()["pred_togo"] for e in new]).astype(np.float64)
            tg_true = np.concatenate([sr.togo_targets(e.out["incr"], f)[:-1] for e in new])
            tfrac = np.concatenate([np.arange(e.length) / e.length for e in new])
            if np.isfinite(tg_pred).all():
                r2 = {}
                for j, k in enumerate(sr.TOGO_KEYS):
                    bins = {}
                    for b in range(4):
                        sel = (tfrac >= b / 4) & (tfrac < (b + 1) / 4)
                        v = tg_true[sel, j].var()
                        bins[f"q{b}"] = float(1 - ((tg_pred[sel, j] - tg_true[sel, j]) ** 2).mean() / v) if v > 0 else None
                    v = tg_true[:, j].var()
                    r2[k] = {"all": float(1 - ((tg_pred[:, j] - tg_true[:, j]) ** 2).mean() / v) if v > 0 else None, **bins}
                t["togo_r2"] = r2
        # α fits
        t["alpha"] = {run: al[f] for run, al in alphas.items() if f in al}
        t["r2_fit_ok"] = all((a.get("r2_fit") is None or a.get("r2_alpha1") is None
                              or a["r2_fit"] >= max(0.0, a["r2_alpha1"]) - 1e-9) for a in t["alpha"].values())
        # sanity
        A = np.array([adv_ep[id(e)] for e in allf])
        succ = np.array([e.success for e in allf])
        frames = np.concatenate([e.out["A_norm"] for e in allf]) if allf else np.zeros(0)
        t["pct_positive_frames"] = float((frames >= 0).mean()) if len(frames) else None
        # how the frames that finish a success are labelled (a V_T mismatch shows up here first)
        ends = [e.out["A_norm"][-20:] for e in allf if e.success]
        t["succ_last20_mean_adv"] = float(np.concatenate(ends).mean()) if ends else None
        if succ.sum() > 2:
            ct = np.array([e.length for e in allf])[succ]
            col = np.array([sum(e.metrics.get(k, 0.0) for k in ("scc", "ecc", "sc")) for e in allf])[succ]
            t["succ_corr_adv_ct"] = float(np.corrcoef(A[succ], ct)[0, 1]) if np.std(ct) > 0 and np.std(A[succ]) > 0 else None
            t["succ_corr_adv_collisions"] = float(np.corrcoef(A[succ], col)[0, 1]) if np.std(col) > 0 and np.std(A[succ]) > 0 else None
        if (~succ).sum():
            t["fail_frac_negative"] = float((A[~succ] < 0).mean())
        d["tasks"][f] = t
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", nargs="+", required=True, help="rollout run dirs, oldest first")
    p.add_argument("--newest", required=True)
    p.add_argument("--extra", nargs="*", default=[], help="hard-case recovery runs: advantages from the pool's fits "
                   "(the α of the collect they branched, from recovery.json, else the newest's; the newest's per-task "
                   "norm), never part of the score weights, α fits, σ or diagnostics")
    p.add_argument("--state", required=True, help="runs/<name>: score_weights.jsonl + advantages_round_<K>.json")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ema", type=float, default=0.2, help="EMA of the held head predictions")
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--tp-weight", type=float, default=3.0,
                   help="multiplier on the subtask-progress weight g_tp in reward and value (1 = leaderboard-exact)")
    p.add_argument("--pos-threshold", type=float, default=0.2,
                   help="subtracted from the normalised advantage: frames below this many sd above the task's "
                        "average get Ã < 0 (the neutral advantage token)")
    p.add_argument("--completion-weight", type=float, default=0.3,
                   help="completion shaping reward Φ = w·Ĉ (Vo carries its expected terminal w·P̂)")
    p.add_argument("--togo-lam", type=float, default=0.995,
                   help="λ of the TD togo targets (arrays/togo_td.npy); 0.995 ≈ a 10 s horizon at 20 Hz (a shorter "
                        "horizon pulls the targets toward the recorded prediction)")
    p.add_argument("--alpha-prior-episodes", type=int, default=30)
    p.add_argument("--g-ema", type=float, default=0.5)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()

    runs = [Path(r).resolve() for r in a.pool]
    newest = Path(a.newest).resolve()
    if newest not in runs:
        runs.append(newest)
    else:  # newest last
        runs = [r for r in runs if r != newest] + [newest]
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    eps, extra = load_pool(runs)
    ok = [e for e in eps if e.ok]
    if not ok:
        raise SystemExit("no usable episodes in the pool")
    log.info("%d episodes (%d usable) in %d runs; skipped %s", len(eps), len(ok), len(runs), extra["skipped"])

    # score weights
    sw_path = state / "score_weights.jsonl"
    last = None
    if sw_path.exists():
        lines = [json.loads(l) for l in sw_path.read_text().splitlines() if l.strip()]
        last = lines[-1] if lines else None
    prev = {f: np.asarray(v["g"]) for f, v in last["per_task"].items()} if last else None
    by_fam = defaultdict(list)
    for e in ok:
        by_fam[e.family].append({"metrics": e.metrics, "success": e.success, "variation": e.meta.get("variation"),
                                 "termination": e.meta.get("termination"), "run_order": e.order})
    info: dict = {}
    # one EMA step per newest collect: a recompute for the same newest (a recovery run arrived, a restart) reuses its g
    reuse = {f for f in by_fam if last is not None and last.get("newest") == str(newest) and f in prev}
    g = {f: prev[f] for f in reuse}
    info.update({f: {k: v for k, v in last["per_task"][f].items() if k != "g"} for f in reuse})
    todo = {f: L for f, L in by_fam.items() if f not in reuse}
    if todo:
        g.update(sr.score_weights(todo, n_boot=a.n_boot, prev=prev, ema=a.g_ema, seed=a.round, info=info))
    if reuse:
        log.info("score weights: same newest collect as the last round, g reused for %d tasks", len(reuse))
    with open(sw_path, "a") as f:
        f.write(json.dumps({"round": a.round, "newest": str(newest), "time": time.time(), "reused": sorted(reuse),
                            "tp_weight": a.tp_weight,
                            "g": {fam: g[fam].tolist() for fam in g},
                            "per_task": {fam: {"g": g[fam].tolist(), **info.get(fam, {})} for fam in g}}) + "\n")
    g_exact = g  # leaderboard-exact (logged above, EMA'd next round; the linearisation diagnostic)
    g = {f: v.copy() for f, v in g_exact.items()}
    for v in g.values():
        v[sr.MI["tp"]] *= a.tp_weight
    g_hash = hashlib.sha1(json.dumps({k: np.round(v, 12).tolist() for k, v in sorted(g.items())}).encode()).hexdigest()[:12]

    for e in ok:
        per_episode(e, g[e.family], a.ema, a.completion_weight, a.togo_lam)

    # α per run and task, oldest -> newest; prior = the previous run's fit
    prior = defaultdict(lambda: {"alpha": 0.5})
    alphas: dict = {}
    for run in runs:
        al = {}
        for fam in sorted({e.family for e in ok if e.run == run}):
            sel = [e for e in ok if e.run == run and e.family == fam]
            G = np.concatenate([e.out["G"] for e in sel])  # the shaped return-to-go (module docstring)
            Vo = np.concatenate([e.out["Vo"] for e in sel])
            Vb = np.concatenate([e.out["Vb"] for e in sel])
            ids = np.concatenate([np.full(e.length, i) for i, e in enumerate(sel)])
            fit = sr.fit_alpha(G, Vo, Vb, ids, prior=prior[fam], prior_episodes=a.alpha_prior_episodes,
                               n_boot=500, seed=a.round)
            fit["g"] = g[fam].tolist()
            al[fam] = fit
            prior[fam] = {"a": fit["a"], "alpha": fit["alpha"]}
        alphas[str(run)] = al
        (run / "alpha.json").write_text(json.dumps(al, indent=1))

    # advantages
    for e in ok:
        fit = alphas[str(e.run)][e.family]
        o = e.out
        V = fit["alpha"] * (o["Vo"] + o["Vb"])  # the fit's intercept a only centred it
        o["A"] = sr.gae(o["r"], V, a.lam)
        o["V"] = V
    newest_eps = [e for e in ok if e.run == newest]
    var_by_task = [np.var(np.concatenate([e.out["A"] for e in newest_eps if e.family == f]))
                   for f in sorted({e.family for e in newest_eps})]
    sigma = float(np.sqrt(np.mean(var_by_task))) if var_by_task else 1.0
    sigma = sigma if sigma > 1e-12 else 1.0
    norm_by_task = {}  # family -> (mean, std) of A over the newest run's frames of that task (else the pool's)
    for f in sorted({e.family for e in ok}):
        A = np.concatenate([e.out["A"] for e in ([x for x in newest_eps if x.family == f]
                                                 or [x for x in ok if x.family == f])])
        sd = float(A.std())
        norm_by_task[f] = (float(A.mean()), sd if sd > 1e-12 else sigma)
    for e in ok:
        mu, sd = norm_by_task[e.family]
        e.out["A_norm"] = np.clip((e.out["A"] - mu) / sd - a.pos_threshold, -a.clip, a.clip).astype(np.float32)
    adv_ep = {id(e): float(e.out["A_norm"].mean()) for e in ok}

    # recovery runs (--extra): the pool's fits and norm, never fitted on (they are successes branched from failures)
    rec_eps, rec_info = load_pool([Path(r).resolve() for r in a.extra]) if a.extra else ([], {"readers": {}})
    rec_ok = [e for e in rec_eps if e.ok and e.family in g and e.family in norm_by_task]
    by_name = {Path(r).name: str(r) for r in runs}
    src = {r: source_collect(r) for r in {e.run for e in rec_ok}}  # the collect each run branched (recovery.json)
    for e in rec_ok:
        per_episode(e, g[e.family], a.ema, a.completion_weight, a.togo_lam)
        fits = alphas.get(by_name.get(src[e.run], ""), {})
        fit = fits.get(e.family) or alphas[str(newest)].get(e.family) or next(
            al[e.family] for al in alphas.values() if e.family in al)
        o = e.out
        o["A"] = sr.gae(o["r"], fit["alpha"] * (o["Vo"] + o["Vb"]), a.lam)
        mu, sd = norm_by_task[e.family]
        o["A_norm"] = np.clip((o["A"] - mu) / sd - a.pos_threshold, -a.clip, a.clip).astype(np.float32)
    if rec_ok:
        allr = np.concatenate([e.out["A_norm"] for e in rec_ok])
        log.info("recovery runs: %d episodes, mean Ã %+.2f, %.0f%% ≥ 0", len(rec_ok), allr.mean(), 100 * (allr >= 0).mean())

    # write
    for root, r in rec_info["readers"].items():
        adv = np.full(len(r), np.nan, np.float32)
        for e in rec_ok:
            if e.reader is r:
                adv[e.start: e.start + e.length] = e.out["A_norm"]
        write_array(root, "advantage", adv)
    for root, r in extra["readers"].items():
        adv = np.full(len(r), np.nan, np.float32)
        for e in ok:
            if e.reader is r:
                adv[e.start: e.start + e.length] = e.out["A_norm"]
        write_array(root, "advantage", adv)
        td = np.full((len(r), len(sr.TOGO_KEYS)), np.nan, np.float32)
        for e in ok:
            if e.reader is r:
                td[e.start: e.start + e.length] = e.out["togo_td"]
        write_array(root, "togo_td", td)
    for run in runs:
        (run / "adv_meta.json").write_text(json.dumps({"sigma": sigma, "lam": a.lam, "round": a.round, "g_hash": g_hash,
                                                       "clip": a.clip, "ema": a.ema, "tp_weight": a.tp_weight,
                                                       "pos_threshold": a.pos_threshold,
                                                       "completion_weight": a.completion_weight,
                                                       "togo_lam": a.togo_lam, "norm_by_task": norm_by_task},
                                                      indent=1))
    diag = diagnostics(ok, newest_eps, g_exact, alphas, adv_ep, sigma, extra["skipped"])
    diag.update(round=a.round, newest=str(newest), pool=[str(r) for r in runs], g_hash=g_hash,
                g={f: v.tolist() for f, v in g.items()}, tp_weight=a.tp_weight, pos_threshold=a.pos_threshold,
                score_weight_info=info, seconds=time.time() - t0)
    out = state / f"advantages_round_{a.round}.json"
    out.write_text(json.dumps(diag, indent=1, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    print(f"{'task':24s} {'eps':>5s} {'succ':>5s} {'fined':>5s} {'α':>6s} {'%A>=0':>6s} {'lin_r':>6s} {'Ã_end':>6s}")
    nan = float("nan")
    for f, t in diag["tasks"].items():
        al = t["alpha"].get(str(newest), {})
        print(f"{f:24s} {t['episodes']:5d} {t['success']:5.2f} {t['fined']:5d} {al.get('alpha', nan):6.2f} "
              f"{100 * (t.get('pct_positive_frames') or 0):6.1f} "
              f"{(t.get('linearisation_corr') or nan):6.3f} {(t.get('succ_last20_mean_adv') or nan):+6.2f}")
    print(f"σ = {sigma:.4f}; diagnostics -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
