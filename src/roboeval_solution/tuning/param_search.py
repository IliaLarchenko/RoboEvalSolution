"""Thompson sampling over inference / rollout parameters (RoboEval).

Continuous and ordinal parameters
    Each parameter j is mapped to u in [0, 1] over its capped range (log scale if `log`). The episode
    reward is modelled as an additive smooth function

        r = b_task + sum_j f_j(u_j) + sum_j g_task,j(u_j) + noise

    where every curve is a weight vector on K Gaussian bumps (centred so a curve has mean 0 over the
    range and the intercept carries the level). Weights have Normal priors, the per-task deviations g
    a tighter one (hierarchy: tasks share evidence, may differ). That is Bayesian linear regression,
    so the posterior over all curves is Normal, closed form, updated per episode.
    Thompson step: draw all weights once from the posterior, then take each parameter's argmax of
    f_j + g_task,j over its grid (integers for ordinal params). Neighbouring values share evidence,
    so the chosen values move smoothly in any direction within the caps.

Binary parameters
    Beta(alpha, beta) arms per task (+ pooled evidence from the other tasks), reward in [0, 1]
    added fractionally (alpha += r, beta += 1 - r).

Conditional parameters
    `active_if=(("keep", ">", 0), ("inpaint_snap", "==", 0))`: only applied (and only updated) on
    episodes where every condition holds; otherwise the parameter is reported as None and gets no
    evidence. A condition on an inactive parameter fails.

Prior: every curve starts with a mild bump at the parameter's default (`prior_peak`, standardised
reward units), so sampling starts around the current defaults and spreads out as evidence accrues.

Rewards are per-episode official task scores (RoboEval Diagnostic Score formula on one episode),
standardised per task with running moments. Every episode is kept in `history` with the run it came from;
`refit()` retrains a new run's posterior on it with recency weights (the policy changes under the parameters).

    ps = ParamSearch(DEFAULT_SPACE, tasks)
    cfg = ps.sample("LiftPot")            # {"noise_temperature": 0.83, "replan": 7, "inpaint_snap": 0, ...}
    ps.update("LiftPot", cfg, reward=0.41)
    ps.best("LiftPot"); ps.state_dict()
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from roboeval_solution.constants import ACTION_HORIZON


@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    kind: str  # "continuous" | "integer" | "binary"
    lo: float = 0.0
    hi: float = 1.0
    default: float = 0.0
    log: bool = False
    active_if: tuple | None = None  # ((name, "==" | ">", value), ...) that must all hold for this param to apply
    where: str = "server"  # "server" (per-request override) | "client" (rollout worker)

    def to_u(self, x: float) -> float:
        if self.log:
            return (math.log(x) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (x - self.lo) / (self.hi - self.lo)

    def from_u(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        x = np.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo))) if self.log else \
            self.lo + u * (self.hi - self.lo)
        return np.round(x) if self.kind == "integer" else x

    def grid(self, n: int = 41) -> np.ndarray:
        """Candidate values (u-space): every integer for ordinal params, n points otherwise."""
        if self.kind == "integer":
            return np.array([self.to_u(v) for v in range(int(self.lo), int(self.hi) + 1)])
        return np.linspace(0.0, 1.0, n)


_INPAINT = (("keep", ">", 0),)

# Ranges are caps, not a grid. Defaults = the prior's preferred value (a bump at it) and the value of history episodes
# that did not draw the param; they must lie inside the range.
# Server params are per-request overrides (serve_policy.py PER_REQUEST_OVERRIDES); a batch runs
# max(num_steps) flow steps, so num_steps is capped (a 20-step draw slows its whole batch).
DEFAULT_SPACE: tuple[Param, ...] = (
    # inpainting params only act with anchors (keep > 0)
    Param("inpaint_snap", "binary", default=0, active_if=_INPAINT),  # 1 = hard pin of the anchors (alpha 1)
    Param("inpaint_pin_alpha", "continuous", 0.3, 1.0, 0.7, active_if=_INPAINT + (("inpaint_snap", "==", 0),)),
    # ranges: wide enough that no task's best sits at an edge; replan + keep <= the chunk
    Param("inpaint_correction_step_cap", "continuous", 0.01, 0.5, 0.1, log=True, active_if=_INPAINT),
    Param("time_threshold_inpaint", "continuous", 0.0, 0.8, 0.3, active_if=_INPAINT),
    # ranges exclude values where the episode score was lower for every task; every task's best stays inside
    Param("noise_temperature", "continuous", 0.05, 0.7, 0.3),
    # correlated flow-noise shrinkage beta (Σ = beta C + (1 - beta) I; trained with 0.5), server grid 0.0..0.9
    Param("correlation_beta", "continuous", 0.0, 0.9, 0.5),
    Param("num_steps", "integer", 3, 20, 10),
    # below 2.6 the episode score of the trained policy is lower in every task
    Param("cfg_scale", "continuous", 2.6, 10.0, 5.0, log=True),
    # best-of-k chunk among the server's --num-candidates (ranked by the Q head); capped to the server's N
    Param("num_candidates", "integer", 2, 8, 4),
    # 1 = the velocity tokens hidden at inference, as velocity dropout in training
    Param("hide_velocity", "binary", default=0),
    # the front / external camera dropped (camera dropout), the flow time grid t_i = u_i^p (1 = uniform),
    # best-of-N ranked by the score-weighted Q residual instead of Δsuccess
    Param("hide_front", "binary", default=0),
    Param("hide_external", "binary", default=0),
    Param("time_warp", "continuous", 0.15, 1.0, 0.3, log=True),
    Param("rank_full", "binary", default=0, active_if=(("num_candidates", ">", 1),)),
    Param("replan", "integer", 6, 25, 8, where="client"),  # actions executed per policy call
    Param("keep", "integer", 0, 8, 4, where="client"),  # inpainted anchors carried into the next chunk (0 = none)
    Param("gripper_threshold", "continuous", 0.4, 0.95, 0.6, where="client"),  # gripper binarisation
    # replan actions executed over round(replan / exec_speed) env steps
    Param("exec_speed", "continuous", 0.7, 1.4, 1.0, log=True, where="client"),
    # 0 off, 1 boost on the P(success) trigger, 2 on any stall event (tuning/stall.py)
    Param("stall_boost", "integer", 0, 2, 1, where="client"),
)
# Params whose evidence counts only in episodes where the named flag of the episode's cfg is set (stall_boost: a stall
# was detected, whatever the setting: the param only acts there, and other episodes only add noise). Episodes without
# the flag are no evidence for them.
GATED: dict[str, str] = {"stall_boost": "stall_seen"}
class ParamSearch:
    def __init__(self, space: Sequence[Param] = DEFAULT_SPACE, tasks: Sequence[str] = (), n_basis: int = 8,
                 prior_sd: float = 0.5, task_prior_sd: float = 0.25, intercept_sd: float = 1.0,
                 noise_sd: float = 1.0, beta_pool: float = 0.2, seed: int = 0, refresh_every: int = 16,
                 prior_peak: float = 0.3):
        self.space = tuple(space)
        self.by_name = {p.name: p for p in self.space}
        self.tasks = list(tasks)
        self.cont = [p for p in self.space if p.kind != "binary"]
        self.binary = [p for p in self.space if p.kind == "binary"]
        self.K = n_basis
        self.centres = np.linspace(0.0, 1.0, n_basis)
        self.width = 1.0 / (n_basis - 1)
        # centring constants: mean of each bump over u in [0, 1]
        uu = np.linspace(0.0, 1.0, 2001)
        self._bump_mean = self._bumps_raw(uu).mean(0)
        self.noise_var = noise_sd ** 2
        self.beta_pool = beta_pool
        T, J, K = len(self.tasks), len(self.cont), self.K
        self.D = T + J * K + T * J * K
        prior_var = np.concatenate([np.full(T, intercept_sd ** 2), np.full(J * K, prior_sd ** 2),
                                    np.full(T * J * K, task_prior_sd ** 2)])
        self.prior_prec = 1.0 / prior_var
        self.prior_mean = np.zeros(self.D)
        for j, p in enumerate(self.cont):  # global curve j: a bump at the default
            self.prior_mean[T + j * K: T + (j + 1) * K] = prior_peak * np.exp(
                -0.5 * ((self.centres - p.to_u(p.default)) / self.width) ** 2)
        self.prec = np.diag(self.prior_prec)  # posterior precision
        self.h = self.prior_prec * self.prior_mean  # precision-weighted mean
        self.beta = {(t, p.name, v): [1.0, 1.0] for t in self.tasks for p in self.binary for v in (0, 1)}
        self.moments = {t: [0.0, 0.0, 0.0] for t in self.tasks}  # weight, mean, M2 of raw rewards
        self.n_episodes = 0
        # [task, cfg, reward, weight, run, step, collector] per update (weight without recency; run = meta
        # ["run_index"], step = meta["step"] (the policy's training step), collector = meta["collector"]), so a
        # changed space / a new run can be refit with recency weights, and the collectors' evidence merged (rollout.py
        # ParamTuner)
        self.history: list = []
        self.rng = np.random.default_rng(seed)
        self._chol_cache = None
        self.refresh_every = refresh_every  # sample() may use a posterior up to this many updates old
        self._stale = 0
        self.meta: dict = {}  # free-form (e.g. the policy the evidence was collected with)

    # ── features ──
    def _bumps_raw(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u)[:, None]
        return np.exp(-0.5 * ((u - self.centres[None]) / self.width) ** 2)

    def _bumps(self, u) -> np.ndarray:
        return self._bumps_raw(u) - self._bump_mean[None]

    def _active(self, p: Param, cfg: dict) -> bool:
        for name, op, val in p.active_if or ():
            x = cfg.get(name)
            if x is None or not (x == val if op == "==" else x > val):
                return False
        return True

    def _apply_conditions(self, cfg: dict) -> dict:
        """keep <= the chunk length - replan (the server has no actions past the chunk to carry over: it would cut keep
        short and the evidence would name a keep that never ran), then inactive params -> None (repeated: a condition on
        a now-inactive param fails too)."""
        if cfg.get("replan") is not None and cfg.get("keep") is not None:
            cfg["keep"] = min(int(cfg["keep"]), max(ACTION_HORIZON - int(cfg["replan"]), 0))
        for _ in range(3):
            for p in self.space:
                if cfg.get(p.name) is not None and not self._active(p, cfg):
                    cfg[p.name] = None
        return cfg

    def in_space(self, cfg: dict) -> bool:
        """The configuration can be drawn from this space: every range param inside its range (None = not drawn /
        inactive: it ran the default)."""
        for p in self.cont:
            x = cfg.get(p.name)
            if x is not None and not (p.lo - 1e-6 * (p.hi - p.lo) <= float(x) <= p.hi + 1e-6 * (p.hi - p.lo)):
                return False
        return True

    @staticmethod
    def _evidence(p: Param, cfg: dict) -> bool:
        """Does this episode carry evidence about p (GATED: only with its flag set)?"""
        return p.name not in GATED or bool(cfg.get(GATED[p.name]))

    def _features(self, task: str, cfg: dict) -> np.ndarray:
        T, K = len(self.tasks), self.K
        ti = self.tasks.index(task)
        phi = np.zeros(self.D)
        phi[ti] = 1.0
        for j, p in enumerate(self.cont):
            if not self._active(p, cfg) or cfg.get(p.name) is None or not self._evidence(p, cfg):
                continue
            b = self._bumps(min(max(p.to_u(float(cfg[p.name])), 0.0), 1.0))[0]
            phi[T + j * K: T + (j + 1) * K] = b
            off = T + len(self.cont) * K + (ti * len(self.cont) + j) * K
            phi[off: off + K] = b
        return phi

    def _standardise(self, task: str, r: float) -> float:
        w, mean, m2 = self.moments[task]
        sd = math.sqrt(m2 / w) if w > 1.0 else 0.0
        return (r - mean) / sd if sd > 1e-6 else r - mean

    # ── posterior ──
    def _posterior(self, exact: bool = True):
        if self._chol_cache is None or (self._stale and (exact or self._stale >= self.refresh_every)):
            self._stale = 0
            L = np.linalg.cholesky(self.prec)
            mean = cho_solve((L, True), self.h)
            self._chol_cache = (L, mean)
        return self._chol_cache

    def _draw_weights(self) -> np.ndarray:
        L, mean = self._posterior(exact=False)
        z = self.rng.standard_normal(self.D)
        # prec = L L^T -> cov = L^-T L^-1 -> sample = mean + L^-T z
        return mean + solve_triangular(L, z, lower=True, trans="T")

    def _curve(self, w: np.ndarray, task: str, j: int, u: np.ndarray) -> np.ndarray:
        T, K, J = len(self.tasks), self.K, len(self.cont)
        ti = self.tasks.index(task)
        wg = w[T + j * K: T + (j + 1) * K]
        off = T + J * K + (ti * J + j) * K
        return self._bumps(u) @ (wg + w[off: off + K])

    # ── API ──
    def sample(self, task: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for p in self.binary:
            draws = {}
            for v in (0, 1):
                a, b = self.beta[(task, p.name, v)]
                pa = sum(self.beta[(t, p.name, v)][0] - 1.0 for t in self.tasks if t != task)
                pb = sum(self.beta[(t, p.name, v)][1] - 1.0 for t in self.tasks if t != task)
                draws[v] = self.rng.beta(a + self.beta_pool * pa, b + self.beta_pool * pb)
            cfg[p.name] = int(max(draws, key=draws.get))
        w = self._draw_weights()
        for j, p in enumerate(self.cont):
            g = p.grid()
            u = g[int(np.argmax(self._curve(w, task, j, g)))]
            x = float(p.from_u(np.array([u]))[0])
            cfg[p.name] = int(x) if p.kind == "integer" else x
        return self._apply_conditions(cfg)

    def update(self, task: str, cfg: dict, reward: float, weight: float = 1.0) -> None:
        reward = float(reward)
        mo = self.moments[task]  # weighted Welford
        mo[0] += weight
        d = reward - mo[1]
        mo[1] += weight * d / mo[0]
        mo[2] += weight * d * (reward - mo[1])
        self._add(task, cfg, reward, weight)

    def _add(self, task: str, cfg: dict, reward: float, weight: float) -> None:
        """Posterior update of one episode (reward standardised by the current moments)."""
        r = self._standardise(task, reward)
        phi = self._features(task, cfg)
        self.prec += weight * np.outer(phi, phi) / self.noise_var
        self.h += weight * phi * r / self.noise_var
        self._stale += 1
        r01 = min(max(reward, 0.0), 1.0)
        for p in self.binary:
            if cfg.get(p.name) is not None and self._active(p, cfg) and self._evidence(p, cfg):
                ab = self.beta[(task, p.name, int(cfg[p.name]))]
                ab[0] += weight * r01
                ab[1] += weight * (1.0 - r01)
        self.n_episodes += 1
        # every drawn value, also of params not in this space
        self.history.append([task, dict(cfg), reward, weight, int(self.meta.get("run_index", 0)), self.meta.get("step"),
                             self.meta.get("collector")])

    def best(self, task: str) -> dict[str, Any]:
        """Posterior-mean argmax per parameter (binary: higher Beta mean)."""
        cfg: dict[str, Any] = {}
        for p in self.binary:
            means = {v: (lambda ab: ab[0] / (ab[0] + ab[1]))(self.beta[(task, p.name, v)]) for v in (0, 1)}
            cfg[p.name] = int(max(means, key=means.get))
        _, mean = self._posterior()
        for j, p in enumerate(self.cont):
            g = p.grid()
            x = float(p.from_u(np.array([g[int(np.argmax(self._curve(mean, task, j, g)))]]))[0])
            cfg[p.name] = int(x) if p.kind == "integer" else x
        return self._apply_conditions(cfg)

    @classmethod
    def refit(cls, old: "ParamSearch", space: Sequence[Param], tasks: Sequence[str], seed: int = 0,
              decay: float = 1.0, run_index: int | None = None, min_weight: float = 1e-3, *, step: int,
              step_per_run: float = 500.0) -> "ParamSearch":
        """Retrain from the history with recency weights: an episode counts its weight x decay^age, age = |step -
        its policy step| / step_per_run (comparable across collectors; by distance, so after a rollback the evidence
        of the dropped newer policies counts less too). Below `min_weight` an episode is dropped. Rewards are
        standardised by the weighted moments of the whole kept history and the posterior is refit under `space`
        (changed ranges, added / removed params). A param the old episodes did not draw takes its default (what those
        episodes ran with); values of params the new space drops stay in the history, so they come back if the param
        is re-added."""
        ps = cls(space, tasks, seed=seed)
        ps.meta = dict(old.meta)
        cur = int(old.meta.get("run_index", 0) if run_index is None else run_index)
        ps.meta["run_index"] = cur
        hist = []  # (task, cfg, reward, weight, run, effective weight)
        dropped = 0
        for h in old.history:
            task, cfg, reward, weight = h[:4]
            run, h_step, coll = int(h[4]), h[5], h[6]
            age = (step - h_step) / step_per_run
            eff = weight * decay ** abs(age)
            if eff >= min_weight and task in ps.moments:
                if not ps.in_space(cfg):  # drawn outside a narrowed range
                    dropped += 1
                    continue
                hist.append((task, cfg, reward, weight, run, eff, h_step, coll))
        for task, _, reward, _, _, eff, _, _ in hist:  # moments first: one standardisation for all kept episodes
            mo = ps.moments[task]
            mo[0] += eff
            d = reward - mo[1]
            mo[1] += eff * d / mo[0]
            mo[2] += eff * d * (reward - mo[1])
        rows = []
        for task, cfg, reward, weight, run, eff, h_step, coll in hist:
            full = {p.name: cfg.get(p.name, p.default if p.kind != "binary" else int(p.default)) for p in ps.space}
            for p in ps.space:  # a param the old space had as inactive (None) stays None
                if p.name in cfg and cfg[p.name] is None:
                    full[p.name] = None
            cfg = {**cfg, **ps._apply_conditions(full)}
            rows.append(ps._features(task, cfg))
            r01 = min(max(reward, 0.0), 1.0)
            for p in ps.binary:
                if cfg.get(p.name) is not None and ps._active(p, cfg) and ps._evidence(p, cfg):
                    ab = ps.beta[(task, p.name, int(cfg[p.name]))]
                    ab[0] += eff * r01
                    ab[1] += eff * (1.0 - r01)
            ps.history.append([task, cfg, reward, weight, run, h_step, coll])
        if rows:  # the Gaussian part in one product (same sums as _add per episode)
            phi = np.stack(rows)
            w = np.array([h[5] for h in hist])
            r = np.array([ps._standardise(h[0], h[2]) for h in hist])
            ps.prec += (phi * (w / ps.noise_var)[:, None]).T @ phi
            ps.h += phi.T @ (w * r) / ps.noise_var
        ps.n_episodes = len(hist)
        ps.meta["refit_dropped"] = dropped
        ps.effective = float(sum(h[5] for h in hist))
        ps._chol_cache = None
        return ps

    # ── persistence ──
    def state_dict(self) -> dict:
        return {"space": [dataclasses.asdict(p) for p in self.space], "tasks": self.tasks, "K": self.K,
                "prec": self.prec.tolist(), "h": self.h.tolist(), "prior_prec": self.prior_prec.tolist(),
                "prior_mean": self.prior_mean.tolist(), "meta": self.meta,
                "beta": [[t, n, v, ab] for (t, n, v), ab in self.beta.items()],
                "moments": self.moments, "n_episodes": self.n_episodes, "noise_var": self.noise_var,
                "history": self.history,
                "beta_pool": self.beta_pool}

    @classmethod
    def from_state_dict(cls, d: dict, seed: int = 0) -> "ParamSearch":
        tup = lambda x: tuple(tup(i) for i in x) if isinstance(x, (list, tuple)) else x  # noqa: E731
        space = [Param(**{**p, "active_if": tup(p["active_if"]) if p["active_if"] else None}) for p in d["space"]]
        ps = cls(space, d["tasks"], n_basis=d["K"], noise_sd=math.sqrt(d["noise_var"]), beta_pool=d["beta_pool"],
                 seed=seed)
        ps.prec, ps.h, ps.prior_prec = np.array(d["prec"]), np.array(d["h"]), np.array(d["prior_prec"])
        ps.prior_mean = np.array(d["prior_mean"])
        ps.meta = d["meta"]
        ps.beta = {(t, n, int(v)): list(ab) for t, n, v, ab in d["beta"]}
        ps.moments = {t: list(m) for t, m in d["moments"].items()}
        ps.n_episodes = d["n_episodes"]
        ps.history = d["history"]
        return ps

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.state_dict()))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "ParamSearch":
        return cls.from_state_dict(json.loads(Path(path).read_text()), seed=seed)
