"""RL flywheel data mix: the rollout pool, LeHome share decay, score-exact advantages, and the in-process
data refresh that swaps the training data loader while the model keeps training.

Sources (build_sources); each source's sampling mass is frames x sample_share, normalised over all sources:
  - demos: the base config's sources, sample_share demo_weight, every demo frame at one fixed advantage;
  - one advantage-prioritised source per collect in the pool (the newest `pool_window`);
  - one advantage-prioritised source per hard-case recovery run (the newest `recovery_window`);
  - one source per scripted generation g<GGGG> (the newest `scripted_window`): full-reward successes at advantage 1.0.
The sample_share of a collect, recovery run or generation is the LeHome age decay: 1.0 when it first joins training,
x rollout_decay per `steps_per_decay` training steps after that. Frames inside a source keep their relative weights
(task balance, variation weight, advantage priority exp(clip(A)): data/dataset.py).

Used by scripts/rl_loop.py (the initial pool / config) and scripts/train.py (DataRefresher, TrainConfig.data_refresh):
a background thread notices new collects, recovery runs and scripted chunks, recomputes the advantages over the pool
(CPU subprocess), builds and warms up a new data loader, and train.py swaps to it at a step boundary.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("flywheel")
REPO = Path(__file__).resolve().parents[3]


@dataclasses.dataclass(frozen=True)
class DataRefreshConfig:
    """In-process pool refresh (TrainConfig.data_refresh; set by rl_loop.py --role trainer)."""
    run_dir: str                       # runs/<name>: rollouts/<collect>/DONE, flywheel.json, score_weights.json
    pool_window: int = 6               # newest N collects
    interval_s: float = 120.0          # how often to look for new collects
    steps_per_decay: int = 1000        # one LeHome "round" of share decay
    # The demos' sample_share (per-frame weight relative to a fresh collect's frames; 0 = not trained on)
    demo_weight: float = 0.5
    # Advantage of every demo frame. -1 keeps the demos as a low-advantage behaviour the model learns but
    # classifier-free guidance steers away from.
    demo_advantage: float = -1.0
    rollout_decay: float = 0.8
    python: str = sys.executable
    # Hard-case recovery runs (runs/<name>/recovery/<run>/ with DONE): the newest N, one source each (0 = off).
    recovery_window: int = 6
    # Scripted runs (runs/<name>/scripted/g<GGGG>_<part>_<policy>/ with DONE, scripts/scripted_round.sh): successes
    # only, advantage 1.0, actions only (no aux targets, dataset.py). One source per generation (age decay from its
    # first chunk's arrival); the newest `scripted_window`.
    scripted_window: int = 8
    scripted_refresh_s: float = 600.0  # a rebuild for new scripted chunks alone at most this often


def collect_order(path: str | Path) -> tuple:
    """Sort key for a collect dir: (policy step, collect number) from its name `<x>_<NNNN>_step_<SSSSSS>`,
    then DONE mtime. Not mtime alone: a re-pulled old collect gets a fresh DONE and would rank newest."""
    p = Path(path)
    m = re.search(r"_(\d+)_step_(\d+)$", p.name)
    done = p / "DONE"
    mtime = done.stat().st_mtime if done.exists() else 0.0
    return (int(m.group(2)), int(m.group(1)), mtime) if m else (float("inf"), 0, mtime)


def current_pool(run_dir: str | Path, pool_window: int) -> list[str]:
    """Complete collects (DONE marker) in <run_dir>/rollouts, oldest first (collect_order), newest `pool_window`."""
    pool = sorted((str(f.parent) for f in (Path(run_dir) / "rollouts").glob("*/DONE")), key=collect_order)
    return pool[-pool_window:] if pool_window > 0 else []


def _state(run_dir: Path) -> dict:
    """Flywheel state (first_step, refreshes) in its own file (rl_loop's State rewrites state.json from memory)."""
    p = run_dir / "flywheel.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(run_dir: Path, d: dict) -> None:
    tmp = run_dir / "flywheel.json.tmp"
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, run_dir / "flywheel.json")


def age_decay(cfg: DataRefreshConfig, names: list[str], step: int) -> dict[str, float]:
    """{name: rollout_decay ** ((step - first) / steps_per_decay)}: the sample_share of a collect, recovery run or
    scripted generation by its age in training steps; records each one's first-seen step in <run_dir>/flywheel.json
    (key first_step)."""
    run_dir = Path(cfg.run_dir)
    d = _state(run_dir)
    first = d.setdefault("first_step", {})
    for n in names:
        first.setdefault(n, int(step))
    _save_state(run_dir, d)
    k = max(cfg.steps_per_decay, 1)
    return {n: cfg.rollout_decay ** (max(step - first[n], 0) / k) for n in names}


def store_frames(paths, success_only: bool = False) -> int:
    """Frames of the episodes a source trains on (its stores' episodes.json; successes only if success_only)."""
    from roboeval_solution.data.store import list_stores

    n = 0
    for root in list_stores(paths):
        for e in json.loads((root / "episodes.json").read_text()):
            if not success_only or e.get("success", False):
                n += int(e["length"])
    return n


def compute_advantages(cfg: DataRefreshConfig, pool: list[str], tag: int, extra: list[str] = ()) -> None:
    """Score-exact advantages over the pool (newest = last) -> arrays/advantage.npy in every store, and the
    current score weights -> <run_dir>/score_weights.json (shipped with exports). `extra` (recovery runs) get
    advantages from the pool's fits without entering them."""
    run_dir = Path(cfg.run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "logs" / f"advantages_r{tag:03d}.log", "w") as f:
        r = subprocess.run([cfg.python, "scripts/compute_advantages.py", "--pool", *pool, "--newest", pool[-1],
                            *(["--extra", *extra] if extra else []),
                            "--state", str(run_dir), "--round", str(tag)],
                           cwd=REPO, env=dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES=""),
                           stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"advantages {tag} failed (see {run_dir}/logs/advantages_r{tag:03d}.log)")
    lines = [l for l in (run_dir / "score_weights.jsonl").read_text().splitlines() if l.strip()]
    g = {fam: v["g"] for fam, v in json.loads(lines[-1])["per_task"].items()}
    tmp = run_dir / "score_weights.json.tmp"
    tmp.write_text(json.dumps({"round": tag, "g": g}, indent=1))
    os.replace(tmp, run_dir / "score_weights.json")


def recovery_runs(run_dir: str | Path, window: int = 10) -> list[str]:
    """Complete hard-case recovery runs (DONE marker, stores) under <run_dir>/recovery, ordered by the collect they
    branched (recovery.json, sim/recovery.py; collect_order), newest `window`."""
    from roboeval_solution.sim.recovery import source_collect

    runs = {}
    for f in (Path(run_dir) / "recovery").glob("*/DONE"):
        if (f.parent / "stores").is_dir():
            runs[str(f.parent)] = source_collect(f.parent)
    order = sorted(runs, key=lambda r: (collect_order(Path(run_dir) / "rollouts" / runs[r]), Path(r).name))
    return order[-window:] if window > 0 else []


def scripted_runs(run_dir: str | Path) -> list[str]:
    """Complete scripted runs (DONE marker, stores) under <run_dir>/scripted, by name."""
    return sorted(str(f.parent) for f in (Path(run_dir) / "scripted").glob("*/DONE") if (f.parent / "stores").is_dir())


GENERATION = re.compile(r"^(g\d{4})_")


def scripted_generations(scr: list[str], window: int) -> dict[str, list[str]]:
    """{generation: its runs} for the newest `window` generations (g<GGGG>_<part>_<policy> names; others ignored)."""
    gens: dict[str, list[str]] = {}
    for r in scr:
        m = GENERATION.match(Path(r).name)
        if m:
            gens.setdefault(m.group(1), []).append(r)
    keep = sorted(gens)[-window:] if window > 0 else []
    return {g: gens[g] for g in keep}


def base_sources_of(sources) -> list:
    """The base config's own sources (the demos) out of a list build_sources made: the collects (rollouts_*),
    recovery runs (recovery_*) and scripted data (scripted*) it appends are rebuilt on every refresh, so carrying
    them as base sources would add them twice (and keep retired or deleted stores)."""
    return [s for s in sources if not (s.name or "").startswith(("rollouts", "recovery", "scripted"))]


def source_mass(s) -> float:
    """A source's sampling mass before the global normalisation (dataset: share x weight_scale)."""
    return s.share * s.weight_scale


def build_sources(base_sources, cfg: DataRefreshConfig, pool: list[str], step: int, rec: list[str] | None = None,
                  scr: list[str] | None = None) -> tuple:
    """Data sources, each with share = its frames x its sample_share (module docstring): the base config's demo
    sources, one advantage-prioritised source per collect and per recovery run, one per scripted generation in the
    window. Recovery episodes are successes and carry no value targets (dataset.py), so they only train the policy."""
    from roboeval_solution.data.dataset import StoreSource

    rec = recovery_runs(cfg.run_dir, cfg.recovery_window) if rec is None else rec
    scr = scripted_runs(cfg.run_dir) if scr is None else scr
    gens = scripted_generations(scr, cfg.scripted_window)
    decay = age_decay(cfg, [Path(r).name for r in pool] + [Path(r).name for r in rec] + list(gens), step)
    demo_share = cfg.demo_weight
    # the base config's share stays (base_sources_of hands the built sources back to every refresh: a share set
    # here would compound); frames x sample_share goes into weight_scale (dataset mass = share x weight_scale)
    sources = [dataclasses.replace(s, weight_scale=store_frames(s.paths, s.success_only) * demo_share,
                                   fixed_advantage=cfg.demo_advantage)  # every demo frame at one advantage
               for s in base_sources]
    for r in pool:
        sources.append(StoreSource(paths=(str(Path(r) / "stores"),), prioritized=True,
                                   share=store_frames((str(Path(r) / "stores"),)) * decay[Path(r).name],
                                   name=f"rollouts_{Path(r).name}"))
    for r in rec:
        sources.append(StoreSource(paths=(str(Path(r) / "stores"),), prioritized=True,
                                   share=store_frames((str(Path(r) / "stores"),)) * decay[Path(r).name],
                                   name=f"recovery_{Path(r).name}"))
    for g in sorted(gens):  # full-reward successes, advantage 1.0, actions only
        paths = tuple(str(Path(x) / "stores") for x in gens[g])
        sources.append(StoreSource(paths=paths, share=store_frames(paths, True) * decay[g], success_only=True,
                                   default_advantage=1.0, name=f"scripted_{g}"))
    total = sum(source_mass(s) for s in sources) or 1.0
    kind, parts = {}, []
    for s in sources:
        k, _, key = (s.name or "demos").partition("_")
        kind[k] = kind.get(k, 0.0) + source_mass(s) / total
        parts.append(f"{key or k} {source_mass(s) / total:.3f}" + (f" (x{decay[key]:.2f})" if key in decay else ""))
    log.info("step %d data mix (frames x sample_share): %s | %s", step,
             ", ".join(f"{k} {v:.3f}" for k, v in kind.items()), ", ".join(parts))
    return tuple(sources)


class DataRefresher:
    """Background pool refresh for train.py. `poll()` returns a new (data_loader, iterator, first batch)
    once a refreshed loader is ready, else None; the swap happens in the training loop between steps."""

    def __init__(self, train_config, base_sources, sharding, make_loader, start_step: int):
        self.cfg: DataRefreshConfig = train_config.data_refresh
        self.train_config, self.base_sources = train_config, tuple(base_sources)
        self.sharding, self.make_loader = sharding, make_loader
        self.step = int(start_step)
        # the pool the config was built (and its advantages computed) for, not whatever is DONE by now:
        # a collect finished in between must count as new, or it trains without advantages
        self.pool = [str(Path(src.paths[0]).parent) for src in train_config.data.sources
                     if (src.name or "").startswith("rollouts_")]
        self.rec = sorted({str(Path(p).parent) for src in train_config.data.sources for p in src.paths
                           if Path(p).parent.parent.name == "recovery"}, key=collect_order)
        self.scr = sorted({str(Path(p).parent) for src in train_config.data.sources
                           if (src.name or "").startswith("scripted") for p in src.paths})
        self.tag = int(_state(Path(self.cfg.run_dir)).get("refreshes", 0))
        self.built_step = int(start_step)
        self.built_time = time.time()
        self._ready = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="data_refresh", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def set_step(self, step: int):
        self.step = int(step)

    def poll(self):
        with self._lock:
            ready, self._ready = self._ready, None
        return ready

    def _run(self):
        while True:
            time.sleep(self.cfg.interval_s)
            try:
                pool = current_pool(self.cfg.run_dir, self.cfg.pool_window)
                rec = sorted(recovery_runs(self.cfg.run_dir, self.cfg.recovery_window), key=collect_order)
                scr = scripted_runs(self.cfg.run_dir)
                gens = scripted_generations(scr, self.cfg.scripted_window)
                scr = sorted(r for runs in gens.values() for r in runs)  # the ones that train
                changed = pool != self.pool or rec != self.rec
                # rebuild for new collects or scripted runs (those at most every scripted_refresh_s: a generation
                # arrives chunk by chunk), or once per decay period so the shares keep decaying
                scr_due = scr != self.scr and time.time() - self.built_time >= self.cfg.scripted_refresh_s
                if not pool or (not changed and not scr_due
                                and self.step - self.built_step < self.cfg.steps_per_decay):
                    continue
                t0 = time.time()
                self.tag += 1
                new = [Path(p).name for p in pool if p not in self.pool]
                if changed:
                    log.info("data refresh %d: new collects %s, recovery runs %s; recomputing advantages over %d "
                             "collects + %d recovery runs", self.tag, new,
                             [Path(r).name for r in rec if r not in self.rec], len(pool), len(rec))
                    compute_advantages(self.cfg, pool, self.tag, rec)
                else:
                    log.info("data refresh %d: share decay update at step %d", self.tag, self.step)
                build_step = self.step
                data = dataclasses.replace(self.train_config.data,
                                           sources=build_sources(self.base_sources, self.cfg, pool, self.step, rec,
                                                                 scr))
                loader = self.make_loader(dataclasses.replace(self.train_config, data=data), self.sharding)
                it = iter(loader)
                first = next(it)  # workers up and the first batch ready before the swap
                with self._lock:
                    self._ready = (loader, it, first)
                self.pool, self.rec, self.scr, self.built_step = pool, rec, scr, build_step
                self.built_time = time.time()
                d = _state(Path(self.cfg.run_dir)); d["refreshes"] = self.tag; _save_state(Path(self.cfg.run_dir), d)
                log.info("data refresh %d ready in %.0fs", self.tag, time.time() - t0)
            except Exception:  # noqa: BLE001 — never kill training; retry on the next tick
                log.exception("data refresh failed; keeping the current data loader")
