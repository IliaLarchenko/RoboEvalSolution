"""Background bucket sync for the trainer <-> collector flywheel.

Runs next to `rl_loop.py --role trainer|collector` and never blocks it:

  trainer:   push any complete local export runs/<run>/policies/step_S that train.py's own upload
             (scripts/bucket_sync.py export-policy) did not push -> bucket policies/<run>/step_S, LATEST if newer;
             pull every bucket rollouts/<run>/<id>/ that has a DONE marker -> runs/<run>/rollouts/<id>/
             (local DONE written last, so the trainer only ever sees complete runs); same for bucket
             recovery/<run>/<id>/ -> runs/<run>/recovery/<id>/ (hard-case recovery runs, rollout.py --recover),
             and scripted/<run>/<chunk>/ -> runs/<run>/scripted/<chunk>/ (scripted generations,
             scripts/scripted_round.sh); chunks named in scripted/<run>/RETIRED.txt leave training and the disk;
             --keep-collects / --keep-recovery prune the oldest pulled runs.
  collector: pull bucket policies/<run>/LATEST when it changes -> runs/<run>/policies/step_S (atomic);
             push every local rollout run with READY (and no PUSHED) -> bucket rollouts/<run>/<id>/ + DONE;
             same for runs/<run>/recovery/<id>/ -> bucket recovery/<run>/<id>/ (videos excluded; local stores dropped
             once newer runs are pushed, --keep-recovery-stores) and runs/<run>/scripted/<chunk>/ -> bucket
             scripted/<run>/<chunk>/ (local stores deleted once pushed).

    .venv/bin/python scripts/bucket_daemon.py --run <name> --role trainer --keep-collects 8 --keep-recovery 8
    .venv/bin/python scripts/bucket_daemon.py --run <name> --role collector --keep-rollouts 1
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.distributed import bucket  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("bucket_daemon")
RETIRE_GRACE_S = 3600  # well past the flywheel's next refresh, which drops the retired chunk
PRUNE_GRACE_S = 3600
GENERATION = re.compile(r"^(g\d{4})_")  # = flywheel.GENERATION


def _order(path: Path) -> tuple:
    """(policy step, number) from a run name <x>_<NNNN>_step_<SSSSSS> (flywheel.collect_order without the import)."""
    m = re.search(r"_(\d+)_step_(\d+)$", path.name)
    return (int(m.group(2)), int(m.group(1))) if m else (float("inf"), 0)


def complete_exports(pdir: Path) -> list[Path]:
    return sorted(p for p in pdir.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")
                  and (p / "params").exists() and (p / "assets").exists())


def _pull_complete(remote: str, local: Path, rdir: Path) -> None:
    """Pull into <run>/staging_pull/ and move into place only when complete. The bucket copy carries its own
    DONE marker, which the sync can bring down early: a direct pull could expose a half-downloaded run."""
    stage = rdir / "staging_pull" / local.name
    shutil.rmtree(stage, ignore_errors=True)
    stage.parent.mkdir(parents=True, exist_ok=True)
    bucket.pull_dir(remote, stage)
    (stage / "DONE").touch()
    if local.exists():  # leftover partial pull (no local DONE, or this would have been skipped)
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, local)


def _section(name: str, fn, *args) -> None:
    """One part of a tick; a failure (e.g. an unreadable bucket file) skips only this part until the next tick."""
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("%s failed this tick (will retry)", name)


def prune_oldest(kind_dir: Path, keep: int) -> None:
    """Disk: all but the newest `keep` complete runs (DONE) under runs/<run>/<rollouts|recovery>, oldest first by policy
    step, leave training (DONE -> PRUNED: the flywheel drops them at its next refresh; the name goes to
    <kind_dir>/.pruned so the pull never fetches it again) and are deleted PRUNE_GRACE_S later, once the refreshed
    loader no longer reads them."""
    if keep <= 0:
        return
    runs = sorted((f.parent for f in kind_dir.glob("*/DONE")), key=_order)
    pruned = kind_dir / ".pruned"
    for r in runs[:-keep]:
        names = set(pruned.read_text().split()) if pruned.exists() else set()
        tmp = kind_dir / ".pruned.tmp"
        tmp.write_text("\n".join(sorted(names | {r.name})) + "\n")
        os.replace(tmp, pruned)
        os.replace(r / "DONE", r / "PRUNED")
        (r / "PRUNED").touch()  # the grace clock starts now
        log.info("pruned %s/%s from training (keeping the newest %d); files go in %.0f h", kind_dir.name, r.name, keep,
                 PRUNE_GRACE_S / 3600)
    for f in kind_dir.glob("*/PRUNED"):
        if time.time() - f.stat().st_mtime > PRUNE_GRACE_S:
            shutil.rmtree(f.parent, ignore_errors=True)
            log.info("deleted pruned %s/%s", kind_dir.name, f.parent.name)


def push_exports(run: str, rdir: Path) -> None:
    """Exports train.py's upload missed (it marks the ones it pushed in policies/.pushed). LATEST only moves forward:
    a late push of an older export must not send the collectors back to it."""
    pushed = rdir / "policies" / ".pushed"
    pushed.mkdir(parents=True, exist_ok=True)
    for exp in complete_exports(rdir / "policies"):
        if (pushed / exp.name).exists():
            continue
        t0 = time.time()
        bucket.push_dir(exp, f"policies/{run}/{exp.name}")
        latest = (bucket.get_text(f"policies/{run}/LATEST") or "").strip()
        if exp.name > latest:
            bucket.put_text(f"policies/{run}/LATEST", exp.name)
        (pushed / exp.name).touch()
        log.info("pushed %s in %.0fs; LATEST=%s", exp.name, time.time() - t0, max(exp.name, latest))


def pull_runs(run: str, rdir: Path, kind: str) -> None:
    """Complete bucket <kind>/<run>/<id>/ (DONE) -> runs/<run>/<kind>/<id>/, except the ones pruned here (.pruned)."""
    pruned = rdir / kind / ".pruned"  # dropped from training on purpose (disk): never pulled again
    skip = set(pruned.read_text().split()) if pruned.exists() else set()
    for name in bucket.list_dir(f"{kind}/{run}"):
        local = rdir / kind / name
        if name in skip or (local / "DONE").exists() or bucket.get_text(f"{kind}/{run}/{name}/DONE") is None:
            continue  # pruned, here already, or still uploading
        t0 = time.time()
        _pull_complete(f"{kind}/{run}/{name}", local, rdir)
        log.info("pulled %s %s in %.0fs", kind, name, time.time() - t0)


def sync_scripted(run: str, rdir: Path) -> None:
    """Scripted generation chunks. The ones named in bucket scripted/<run>/RETIRED.txt (scripts/scripted_data.py retire)
    leave training: local DONE -> RETIRED (the flywheel drops them at its next refresh), files deleted RETIRE_GRACE_S
    later (delete_retired). Retirement is decided on the local dirs, not the bucket listing (a bucket clean-up may
    already have removed the chunk), and a chunk with a local RETIRED marker is never pulled again. RETIRED.txt that
    cannot be read raises (bucket.get_text): this tick pulls nothing rather than taking it as empty."""
    sdir = rdir / "scripted"
    retired = set((bucket.get_text(f"scripted/{run}/RETIRED.txt") or "").split())
    for done in sorted(sdir.glob("*/DONE")):
        if done.parent.name in retired:
            os.replace(done, done.parent / "RETIRED")
            (done.parent / "RETIRED").touch()  # the rename keeps DONE's (pull-time) mtime: the grace clock starts now
            log.info("retired scripted %s", done.parent.name)
    for name in bucket.list_dir(f"scripted/{run}"):
        local = sdir / name
        if not GENERATION.match(name) or name in retired or (local / "RETIRED").exists() or (local / "DONE").exists():
            continue
        if bucket.get_text(f"scripted/{run}/{name}/DONE") is None:
            continue  # still uploading
        t0 = time.time()
        _pull_complete(f"scripted/{run}/{name}", local, rdir)
        log.info("pulled scripted %s in %.0fs", name, time.time() - t0)


def delete_retired(rdir: Path) -> None:
    """Retired chunks' stores go RETIRE_GRACE_S after the retirement, by the local marker (works offline)."""
    for marker in (rdir / "scripted").glob("*/RETIRED"):
        if (marker.parent / "stores").exists() and time.time() - marker.stat().st_mtime > RETIRE_GRACE_S:
            shutil.rmtree(marker.parent / "stores", ignore_errors=True)
            log.info("deleted the stores of retired scripted %s", marker.parent.name)


def trainer_tick(run: str, rdir: Path, keep_collects: int = 0, keep_recovery: int = 0) -> None:
    _section("pushing exports", push_exports, run, rdir)
    _section("pulling rollouts", pull_runs, run, rdir, "rollouts")
    _section("pulling recovery runs", pull_runs, run, rdir, "recovery")
    _section("syncing scripted chunks", sync_scripted, run, rdir)
    _section("deleting retired chunks", delete_retired, rdir)
    _section("pruning rollouts", prune_oldest, rdir / "rollouts", keep_collects)
    _section("pruning recovery runs", prune_oldest, rdir / "recovery", keep_recovery)


def pull_policy(run: str, rdir: Path, keep_local: int) -> None:
    latest = (bucket.get_text(f"policies/{run}/LATEST") or "").strip()
    pdir = rdir / "policies"
    if latest and not (pdir / latest / "params").exists():
        t0 = time.time()
        tmp = pdir / (latest + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        bucket.pull_dir(f"policies/{run}/{latest}", tmp)
        os.replace(tmp, pdir / latest)
        log.info("pulled policy %s in %.0fs", latest, time.time() - t0)
        for old in complete_exports(pdir)[:-keep_local]:
            shutil.rmtree(old, ignore_errors=True)


def push_ready(run: str, rdir: Path, kind: str, exclude: list[str], drop_stores: bool = False) -> None:
    """Local runs/<run>/<kind>/<id>/ with READY and no PUSHED -> bucket <kind>/<run>/<id>/ + DONE, then PUSHED."""
    for r in sorted((rdir / kind).glob("*")):
        if (r / "READY").exists() and not (r / "PUSHED").exists():
            t0 = time.time()
            bucket.push_dir(r, f"{kind}/{run}/{r.name}", exclude=["READY", "PUSHED", *exclude])
            bucket.put_text(f"{kind}/{run}/{r.name}/DONE", "1")
            (r / "PUSHED").touch()
            if drop_stores:
                shutil.rmtree(r / "stores", ignore_errors=True)  # GBs per chunk; the bucket has them
            log.info("pushed %s %s in %.0fs", kind, r.name, time.time() - t0)


def drop_recovery_stores(run: str, rdir: Path, keep: int) -> None:
    """Stores of pushed recovery runs (GBs each; nothing on the collector reads them again) but the newest `keep`.
    Only when the bucket still has the run: a run whose bucket copy is gone (a clean-up) is the only copy left and gets
    a LOCAL_ONLY marker instead (checked once)."""
    pushed = sorted((r for r in (rdir / "recovery").glob("*") if (r / "PUSHED").exists() and (r / "stores").exists()
                     and not (r / "LOCAL_ONLY").exists()), key=lambda r: (r / "PUSHED").stat().st_mtime)
    for r in pushed[:max(0, len(pushed) - keep)]:
        if bucket.get_text(f"recovery/{run}/{r.name}/DONE") is None:
            (r / "LOCAL_ONLY").touch()
            log.info("recovery %s is not in the bucket any more: its local stores are kept (LOCAL_ONLY)", r.name)
            continue
        shutil.rmtree(r / "stores", ignore_errors=True)
        log.info("removed the local stores of pushed recovery %s", r.name)


def drop_pushed_rollouts(rdir: Path, keep: int) -> None:
    """Local copies of uploaded collects (the bucket has them), the newest `keep` kept."""
    pushed = sorted((r for r in (rdir / "rollouts").glob("*") if (r / "PUSHED").exists()),
                    key=lambda r: (r / "PUSHED").stat().st_mtime)
    for r in pushed[:max(0, len(pushed) - keep)]:
        shutil.rmtree(r, ignore_errors=True)
        log.info("removed local copy of pushed rollouts %s", r.name)


def collector_tick(run: str, rdir: Path, keep_local: int, keep_rollouts: int = -1, keep_recovery_stores: int = -1
                   ) -> None:
    _section("pulling the policy", pull_policy, run, rdir, keep_local)
    _section("pushing rollouts", push_ready, run, rdir, "rollouts", [])
    _section("pushing recovery runs", push_ready, run, rdir, "recovery", ["videos*", "videos*/*"])
    _section("pushing scripted chunks", push_ready, run, rdir, "scripted", [], True)
    if keep_rollouts >= 0:
        _section("removing pushed rollouts", drop_pushed_rollouts, rdir, keep_rollouts)
    if keep_recovery_stores >= 0:
        _section("removing pushed recovery stores", drop_recovery_stores, run, rdir, keep_recovery_stores)


def log_handlers(path: Path) -> list:
    """stderr and the log file, or only stderr when stderr already appends to that file (otherwise every line
    would be written twice)."""
    try:
        same = path.exists() and os.path.samestat(os.fstat(2), path.stat())
    except OSError:
        same = False
    return [logging.StreamHandler()] if same else [logging.StreamHandler(), logging.FileHandler(path)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--role", choices=["trainer", "collector"], required=True)
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--keep-local", type=int, default=2, help="collector: local policy exports to keep")
    p.add_argument("--keep-rollouts", type=int, default=-1,
                   help="collector: local copies of pushed rollout runs to keep (-1 = all)")
    p.add_argument("--keep-recovery-stores", type=int, default=1,
                   help="collector: pushed recovery runs that keep their local stores, newest first (-1 = all)")
    p.add_argument("--keep-collects", type=int, default=0,
                   help="trainer: all but the newest N pulled collects leave training and are deleted an hour later "
                        "(disk; N = the flywheel's pool_window; 0 = keep all)")
    p.add_argument("--keep-recovery", type=int, default=0,
                   help="trainer: the same for recovery runs (N = the recovery window; 0 = keep all)")
    a = p.parse_args()
    rdir = Path(a.runs_dir) / a.run
    (rdir / "logs").mkdir(parents=True, exist_ok=True)
    (rdir / "rollouts").mkdir(parents=True, exist_ok=True)
    (rdir / "policies").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=log_handlers(rdir / "logs" / f"bucket_{a.role}.log"))
    log.info("bucket %s, run %s, role %s", bucket.BUCKET, a.run, a.role)
    while True:
        if a.role == "trainer":
            trainer_tick(a.run, rdir, a.keep_collects, a.keep_recovery)
        else:
            collector_tick(a.run, rdir, a.keep_local, a.keep_rollouts, a.keep_recovery_stores)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
