"""CosineDecayWithCooldown (training/config.py): the cosine schedule up to the cooldown, then a cosine from its LR
there down to the cooldown LR, then constant."""
import math

import pytest

pytest.importorskip("optax")
from roboeval_solution.training.config import CosineDecayWithCooldown  # noqa: E402


def test_off_is_the_cosine_schedule():
    s = CosineDecayWithCooldown(warmup_steps=1000, peak_lr=1.36e-4, decay_steps=100_000, decay_lr=1.36e-5)
    base = s.__class__.__mro__[1].create(s)  # openpi's CosineDecaySchedule
    for t in (0, 500, 1000, 50_000, 99_999, 120_000):
        assert float(s.create()(t)) == pytest.approx(float(base(t)), rel=1e-6)


def test_cooldown():
    kw = dict(warmup_steps=1000, peak_lr=1.36e-4, decay_steps=100_000, decay_lr=1.36e-5)
    base = CosineDecayWithCooldown(**kw).create()
    lr = CosineDecayWithCooldown(**kw, cooldown_start=56_000, cooldown_steps=2500, cooldown_lr=3e-6).create()
    lr_s = float(base(56_000))
    assert lr_s == pytest.approx(6.42e-5, rel=2e-3)  # LR at the cooldown start of configs/trainer.yaml
    for t in (0, 1000, 48_250, 55_999):
        assert float(lr(t)) == pytest.approx(float(base(t)), rel=1e-6)
    for t in (56_000, 56_625, 57_250, 58_000, 58_500, 70_000):
        frac = min(1.0, (t - 56_000) / 2500)
        want = 3e-6 + 0.5 * (lr_s - 3e-6) * (1 + math.cos(math.pi * frac))
        assert float(lr(t)) == pytest.approx(want, rel=1e-5)
    assert float(lr(57_250)) == pytest.approx((lr_s + 3e-6) / 2, rel=1e-5)
    assert float(lr(58_500)) == pytest.approx(3e-6, rel=1e-6) and float(lr(90_000)) == pytest.approx(3e-6, rel=1e-6)
