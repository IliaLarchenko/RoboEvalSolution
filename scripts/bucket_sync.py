"""One-shot bucket transfer of a policy export.

    python scripts/bucket_sync.py export-policy STEP_DIR --run NAME   # bf16 export + push + LATEST (train.py)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.distributed import bucket  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("export-policy")
    ep.add_argument("step_dir")
    ep.add_argument("--run", required=True)
    a = p.parse_args()

    if a.cmd == "export-policy":
        from roboeval_solution.distributed.export import export_policy

        step = Path(a.step_dir)
        out = export_policy(step, ROOT / "runs" / a.run / "policies" / f"step_{int(step.name):06d}")
        weights = ROOT / "runs" / a.run / "score_weights.json"
        if weights.exists():  # the flywheel's current score weights ship with every export
            shutil.copy(weights, out / "score_weights.json")
        bucket.push_dir(out, f"policies/{a.run}/{out.name}")
        bucket.put_text(f"policies/{a.run}/LATEST", out.name)
        (out.parent / ".pushed").mkdir(exist_ok=True)  # bucket_daemon.py --role trainer: nothing left to push
        (out.parent / ".pushed" / out.name).touch()
        print("exported + pushed", out)


if __name__ == "__main__":
    main()
