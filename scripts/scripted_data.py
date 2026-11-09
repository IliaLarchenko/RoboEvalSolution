"""Scripted generations in the bucket (scripts/scripted_round.sh, rollout.py --scripted): retirement.

Chunks live at bucket scripted/<run>/g<GGGG>_<part>_<policy>/ (DONE when complete). Retiring appends chunk names to
scripted/<run>/RETIRED.txt: the trainer's bucket_daemon then takes them out of training (local DONE -> RETIRED, the
flywheel drops them at its next refresh) and deletes their local files after a grace period. The bucket copies stay.

    .venv/bin/python scripts/scripted_data.py retire --run <run> --generation 5   # every chunk of generation 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.distributed import bucket  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHUNK = re.compile(r"^g\d{4}_\d+_\w+$")


def retired_names(run: str) -> set[str]:
    """RETIRED.txt's names (empty if it does not exist yet); an unreadable file raises (bucket.BucketError)."""
    return set((bucket.get_text(f"scripted/{run}/RETIRED.txt") or "").split())


def retire(run: str, generations: list[int]) -> list[str]:
    """Add every chunk of `generations` (in the bucket listing or in the local scripted/ dir) to RETIRED.txt; returns
    the names added."""
    listed = [n for n in bucket.list_dir(f"scripted/{run}") if CHUNK.match(n)]
    local = [d.name for d in (ROOT / "runs" / run / "scripted").glob("g*_*") if d.is_dir() and CHUNK.match(d.name)]
    new = set()
    for g in generations:
        new |= {n for n in listed + local if n.startswith(f"g{g:04d}_")}
    retired = retired_names(run)  # read right before the write; raises rather than overwrite an unread list
    added = sorted(new - retired)
    if added:
        bucket.put_text(f"scripted/{run}/RETIRED.txt", "\n".join(sorted(retired | new)) + "\n")
    return added


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["retire"])
    p.add_argument("--run", required=True)
    p.add_argument("--generation", type=int, action="append", default=[],
                   help="every chunk of this generation (repeatable)")
    a = p.parse_args()
    added = retire(a.run, a.generation)
    print("retired:", " ".join(added) or "(nothing new)")


if __name__ == "__main__":
    main()
