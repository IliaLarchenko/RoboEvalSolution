"""rl_loop.py --config: the YAML keys become flag defaults, the command line overrides them."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rl_loop  # noqa: E402


def test_trainer_config():
    a = rl_loop.parse_args(["--config", str(ROOT / "configs/trainer.yaml"), "--name", "r1"])
    assert (a.role, a.batch_size, a.peak_lr, a.decay_lr, a.cooldown_start, a.cooldown_lr, a.stop_step) == \
        ("trainer", 2816, 1.36e-4, 1.36e-5, 56000, 3e-6, 58000)


def test_collector_config_and_overrides():
    a = rl_loop.parse_args(["--config", str(ROOT / "configs/collector.yaml"), "--name", "r1", "--collector-id", "1",
                            "--workers", "8"])
    assert (a.role, a.collector_id, a.workers, a.episodes_per_task, a.render_gpus) == ("collector", 1, 8, 250, "0")
    assert a.between_cmd == "scripts/scripted_round.sh r1 12 0 2-11,14-23"


def test_unknown_key(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("role: trainer\nbatchsize: 8\n")
    with pytest.raises(SystemExit):
        rl_loop.parse_args(["--config", str(cfg), "--name", "r1"])


def test_without_config_role_is_required():
    with pytest.raises(SystemExit):
        rl_loop.parse_args(["--name", "r1"])
