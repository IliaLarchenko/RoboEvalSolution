"""The flywheel data mix: every dataset (demos, collect, recovery run, scripted generation) sampled in proportion to
its frames x its sample_share (the age decay; the demos' demo_weight), nothing else; one source per scripted
generation, the newest `scripted_window`."""
import json
from pathlib import Path

from roboeval_solution.data.dataset import StoreSource
from roboeval_solution.training.flywheel import (DataRefreshConfig, base_sources_of, build_sources, current_pool,
                                                 scripted_generations, source_mass, store_frames)


def _run(path, eps=((100, True),)):
    """A run dir with DONE and one store whose episodes have these (length, success)."""
    store = path / "stores" / "w00"
    store.mkdir(parents=True)
    (store / "meta.json").write_text("{}")
    (store / "episodes.json").write_text(json.dumps([{"length": n, "success": ok} for n, ok in eps]))
    (path / "DONE").write_text("1")
    return str(path)


def _demos(tmp_path, frames=1000):
    _run(tmp_path / "demos", ((frames, True),))
    return (StoreSource(paths=(str(tmp_path / "demos" / "stores"),), share=1.0, default_advantage=1.0,
                        name="demos"),)


def test_mix_is_frames_times_sample_share(tmp_path):
    demos = _demos(tmp_path, 1000)
    cfg = DataRefreshConfig(run_dir=str(tmp_path), steps_per_decay=1000, rollout_decay=0.5, demo_weight=0.3)
    old = _run(tmp_path / "rollouts" / "c0_0001_step_000500", ((300, True), (500, False)))
    build_sources(demos, cfg, [old], step=0, rec=[], scr=[])  # the first collect joins at step 0
    new = _run(tmp_path / "rollouts" / "c0_0002_step_001000", ((200, True),))
    rec = _run(tmp_path / "recovery" / "r0_0001_step_000500", ((50, True),))
    scr = [_run(tmp_path / "scripted" / "g0001_1_packbox", ((40, True), (60, False))),
           _run(tmp_path / "scripted" / "g0001_2_stack", ((30, True),))]
    by = {s.name: s for s in build_sources(demos, cfg, [old, new], step=2000, rec=[rec], scr=scr)}
    assert source_mass(by["demos"]) == 1000 * 0.3  # frames x demo_weight, no decay, no floor
    assert by["rollouts_c0_0001_step_000500"].share == 800 * 0.25  # failures count; 2000 steps old: 0.5^2, no floor
    assert by["rollouts_c0_0002_step_001000"].share == 200 * 1.0
    assert by["recovery_r0_0001_step_000500"].share == 50 * 1.0
    s = by["scripted_g0001"]  # successes only
    assert s.share == 70 * 1.0 and s.success_only and s.default_advantage == 1.0 and not s.prioritized
    assert all(x.weight_scale == 1.0 and source_mass(x) == x.share for n, x in by.items() if n != "demos")
    assert store_frames(s.paths) == 130 and store_frames(s.paths, success_only=True) == 70


def test_generation_sources(tmp_path):
    demos = _demos(tmp_path)
    pool = [_run(tmp_path / "rollouts" / f"c0_{i:04d}_step_{i * 500:06d}") for i in (1, 2)]
    gens = {g: [_run(tmp_path / "scripted" / f"g{g:04d}_{p}_{pol}") for p, pol in ((1, "packbox"), (2, "stack"))]
            for g in range(1, 7)}
    _run(tmp_path / "scripted" / "s0_0005_stack")  # not a generation: ignored
    cfg = DataRefreshConfig(run_dir=str(tmp_path), scripted_window=4, steps_per_decay=1000, rollout_decay=0.5)
    assert current_pool(tmp_path, 6) == pool
    by = {s.name: s for s in build_sources(demos, cfg, pool, step=1000, rec=[])}
    assert sorted(n for n in by if n.startswith("scripted")) == [f"scripted_g{g:04d}" for g in (3, 4, 5, 6)]
    assert set(by["scripted_g0006"].paths) == {f"{r}/stores" for r in gens[6]}
    assert by["scripted_g0006"].share == 200  # fresh: frames x 1
    first = json.loads((tmp_path / "flywheel.json").read_text())["first_step"]
    assert first["g0006"] == 1000 and "g0002" not in first
    # a generation decays from its first step like a collect: 2000 steps later at 0.5^2, a new one at 1
    g7 = [_run(tmp_path / "scripted" / "g0007_1_packbox")]
    by = {s.name: s for s in build_sources(demos, cfg, pool, step=3000, rec=[])}
    assert "scripted_g0003" not in by and by["scripted_g0007"].paths == (f"{g7[0]}/stores",)
    assert by["scripted_g0006"].share == 200 * 0.25 and by["scripted_g0007"].share == 100


def test_scripted_generations_window():
    gens = scripted_generations(["/r/scripted/g0002_1_a", "/r/scripted/s0_0001_a", "/r/scripted/g0001_1_a",
                                 "/r/scripted/g0003_2_b", "/r/scripted/g0003_1_a"], window=2)
    assert gens == {"g0002": ["/r/scripted/g0002_1_a"], "g0003": ["/r/scripted/g0003_2_b", "/r/scripted/g0003_1_a"]}


def test_demo_weight_and_fixed_advantage(tmp_path):
    demos = _demos(tmp_path)
    pool = [_run(tmp_path / "rollouts" / "c0_0001_step_000500")]
    _run(tmp_path / "scripted" / "g0001_1_packbox")
    cfg = DataRefreshConfig(run_dir=str(tmp_path), demo_weight=0.0)
    by = {s.name: s for s in build_sources(demos, cfg, pool, step=0, rec=[])}
    # demos not sampled; the scripted generation keeps its own mass
    assert source_mass(by["demos"]) == 0.0 and source_mass(by["scripted_g0001"]) == 100
    # the defaults: demos at sample share 0.5 as the low-advantage side, every demo frame at advantage -1
    cfg = DataRefreshConfig(run_dir=str(tmp_path))
    by = {s.name: s for s in build_sources(demos, cfg, pool, step=0, rec=[])}
    assert abs(source_mass(by["demos"]) - 500.0) < 1e-9 and by["demos"].fixed_advantage == -1.0
    assert by["scripted_g0001"].fixed_advantage is None and by["scripted_g0001"].default_advantage == 1.0


def test_refresh_base_sources_are_the_demos_only(tmp_path):
    demos = _demos(tmp_path)
    pool = [_run(tmp_path / "rollouts" / f"c0_{i:04d}_step_{i * 500:06d}") for i in (2, 3)]
    rec = _run(tmp_path / "recovery" / "r0_0002_step_001500")
    (Path(rec) / "recovery.json").write_text(json.dumps({"collect": "c0_0002_step_001000"}))
    _run(tmp_path / "scripted" / "g0001_1_packbox")
    cfg = DataRefreshConfig(run_dir=str(tmp_path))
    built = build_sources(demos, cfg, pool, step=0, rec=[rec])
    # the launch config's collects, recovery runs and scripted data are rebuilt by every refresh: not base sources
    assert [s.name for s in base_sources_of(built)] == ["demos"]
    rebuilt = build_sources(base_sources_of(built), cfg, pool, step=0, rec=[rec])
    names = [s.name for s in rebuilt]
    assert len(names) == len(set(names)) == len(built)
    # a refresh rebuilds from the built demos: the same masses
    assert [source_mass(s) for s in rebuilt] == [source_mass(s) for s in built]
    again = build_sources(base_sources_of(rebuilt), cfg, pool, step=0, rec=[rec])
    assert [source_mass(s) for s in again] == [source_mass(s) for s in built]
