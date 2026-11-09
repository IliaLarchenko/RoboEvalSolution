"""flywheel.current_pool orders collects by (policy step, collect number), not DONE mtime: a re-pulled old
collect (fresh DONE) must not rank newest."""
import os

from roboeval_solution.training.flywheel import current_pool


def test_repulled_collect_keeps_its_place(tmp_path):
    names = ["c0_0005_step_003500", "c0_0006_step_004000", "c0_0010_step_007000", "c0_0011_step_008000"]
    for i, n in enumerate(names):
        (tmp_path / "rollouts" / n).mkdir(parents=True)
        (tmp_path / "rollouts" / n / "DONE").touch()
        os.utime(tmp_path / "rollouts" / n / "DONE", (1000 + i, 1000 + i))
    os.utime(tmp_path / "rollouts" / names[0] / "DONE", (9999, 9999))  # re-pulled: newest DONE
    pool = [os.path.basename(p) for p in current_pool(tmp_path, 3)]
    assert pool == names[1:]
    assert [os.path.basename(p) for p in current_pool(tmp_path, 10)] == names
