"""Scripted policies without a simulator: the registry's parameter draws, the ShelfBook hull corners and the
continuity of the PickBook pinch point over the book's yaw."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim import scripted
from roboeval_solution.sim import scripted_pickbook as pb
from roboeval_solution.sim import scripted_shelfbook as sb


@pytest.mark.parametrize("name", sorted(scripted.POLICIES))
def test_make_draws_known_params_in_range(name):
    ctl, params = scripted.make(name, seed=43_000_000 * 16)
    mod = __import__(scripted.POLICIES[name][1], fromlist=["RANDOMIZE"])
    assert set(params) == set(mod.RANDOMIZE) <= set(mod.DEFAULTS)
    for k, (lo, hi) in mod.RANDOMIZE.items():
        assert lo <= params[k] <= hi and ctl.p[k] == params[k]
    assert scripted.make(name, seed=43_000_000 * 16)[1] == params  # deterministic per seed
    assert scripted.make(name, seed=43_000_000 * 16, randomize=False)[1] == {}


def test_shelf_corners_are_the_true_hull():
    lo, hi = np.array([-0.0134, -0.0530, -0.0694]), np.array([0.0131, 0.0411, 0.0949])  # hull around the mesh origin
    rng = np.random.default_rng(0)
    for _ in range(20):
        M, o = R.random(random_state=rng).as_matrix(), rng.normal(size=3)
        true = np.array([o + M @ np.array([x, y, z])
                         for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        got = sb.book_corners(o + M @ pb.BOX_OFF, R.from_matrix(M))  # book_frame returns the box centre
        assert np.abs(np.sort(got, 0) - np.sort(true, 0)).max() < 3e-4


def _flat_book(yaw):
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.array([0.0, 0.0, 1.0])
    return R.from_matrix(np.stack([x, y, np.cross(x, y)], 1))


def test_pickbook_pinch_point_continuous_in_yaw():
    ctl = pb.ScriptedPickBook()
    ctl.arm = 0
    c = np.array([0.40, 0.15, 0.70])
    yaws = np.radians(np.arange(1.0, 44.0, 0.05))  # one arm and one edge over this range
    pts = np.array([ctl._pinch_plan(c, _flat_book(a))[0][:2] for a in yaws])
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() < 2e-3  # no jump
