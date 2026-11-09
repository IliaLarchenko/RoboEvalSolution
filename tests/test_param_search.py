"""ParamSearch.refit: the Thompson-sampling history across search-space changes, recency weights and the
collectors' shared evidence."""
import numpy as np

from roboeval_solution.tuning.param_search import ParamSearch, Param


def test_refit_keeps_values_of_dropped_params():
    a, b = Param("a", "continuous", 0.0, 1.0, 0.5), Param("b", "continuous", 0.0, 1.0, 0.5)
    ps = ParamSearch([a, b], ["T"], seed=0)
    ps.meta["step"] = 1000
    ps.update("T", {"a": 0.2, "b": 0.9}, 1.0)
    dropped = ParamSearch.refit(ps, [a], ["T"], step=1000)
    back = ParamSearch.refit(dropped, [a, b], ["T"], step=1000)
    assert back.history[0][1]["b"] == 0.9  # the drawn value, not the default
    narrow = ParamSearch.refit(ps, [a, Param("b", "continuous", 0.0, 0.5, 0.25)], ["T"], step=1000)
    assert np.isfinite(narrow._features("T", {"a": 0.2, "b": 0.9})).all()  # beyond the shrunk range: edge


def test_refit_weights_runs_by_recency():
    a = Param("a", "continuous", 0.0, 1.0, 0.5)
    ps = ParamSearch([a], ["T"], seed=0)
    for run, r in enumerate((0.0, 1.0, 1.0)):  # one episode per run 0, 1, 2 (policy steps 0, 500, 1000)
        ps.meta.update(run_index=run, step=500 * run)
        ps.update("T", {"a": 0.3}, r)
    assert [h[4] for h in ps.history] == [0, 1, 2]
    new = ParamSearch.refit(ps, [a], ["T"], decay=0.5, run_index=3, step=1500)
    assert [h[3] for h in new.history] == [1.0, 1.0, 1.0]  # stored weights stay; recency is applied at refit
    w, mean, _ = new.moments["T"]
    assert np.isclose(w, 0.125 + 0.25 + 0.5) and np.isclose(mean, 0.75 / 0.875)  # newer runs weigh more
    ps.history[0][3] = 1e-3
    assert len(ParamSearch.refit(ps, [a], ["T"], decay=0.5, run_index=3, step=1500).history) == 2  # below min_weight
    again = ParamSearch.refit(new, [a], ["T"], decay=0.5, step=1500)  # same policy step: the same posterior
    assert np.allclose(again.prec, new.prec) and np.allclose(again.h, new.h)


def test_refit_recency_by_policy_step_and_merged_collectors(tmp_path, monkeypatch):
    """Recency by policy step (comparable across collectors), and the collectors' evidence merged through the bucket:
    each publishes its own entries, a refit takes its own + the others'."""
    import json
    import sys
    from pathlib import Path

    from roboeval_solution.tuning.param_search import DEFAULT_SPACE, ParamSearch

    space = [p for p in DEFAULT_SPACE if p.name in ("noise_temperature", "hide_velocity")]
    ps = ParamSearch(space, ["LiftPot"], seed=0)
    ps.meta.update(run_index=3, step=1000, collector=0)
    ps.update("LiftPot", {"noise_temperature": 0.3, "hide_velocity": 0}, reward=1.0)
    ps.meta.update(run_index=4, step=1500)
    ps.update("LiftPot", {"noise_temperature": 0.3, "hide_velocity": 0}, reward=0.5)
    r = ParamSearch.refit(ps, space, ["LiftPot"], decay=0.5, run_index=5, step=2000, step_per_run=500.0)
    # ages 2 runs (step 1000), 1 run (step 1500): weights 0.25 + 0.5
    assert abs(r.effective - 0.75) < 1e-9 and all(len(h) == 7 for h in r.history)
    assert r.history[0][5:] == [1000, 0]
    # after a rollback (current step below theirs) the newer-step evidence counts by distance: |1000 - 1000| = 0 runs,
    # |1500 - 1000| = 1 run
    rb = ParamSearch.refit(ps, space, ["LiftPot"], decay=0.5, run_index=5, step=1000, step_per_run=500.0)
    assert abs(rb.effective - (1.0 + 0.5)) < 1e-9

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import rollout

    store = {}
    from roboeval_solution.distributed import bucket
    monkeypatch.setattr(bucket, "put_text", lambda path, text: store.__setitem__(path, text))
    monkeypatch.setattr(bucket, "get_text", lambda path: store.get(path))
    monkeypatch.setattr(bucket, "list_dir", lambda prefix: sorted(k.rsplit("/", 1)[1] for k in store
                                                                  if k.startswith(prefix + "/")))
    r.meta.update(step=2000, run_index=5)
    rollout.publish_history(r, ("R", 0), 500.0)  # its two entries with a policy step
    assert json.loads(store["state/R/param_search/c0.json"])["collector"] == 0
    # collector 1 merges collector 0's evidence (tagged 0) with its own; its own c1 entries are not read back
    own1 = [["LiftPot", {"noise_temperature": 0.5, "hide_velocity": 1}, 0.9, 1.0, 0, 2000, 1]]
    merged = rollout.shared_history(own1 + [["LiftPot", {}, 0.1, 1.0, 0, 1900, 0]], ("R", 1))
    assert merged[0] == own1[0] and len(merged) == 1 + 2 and all(h[6] == 0 for h in merged[1:])
    args = type("A", (), {"out": "/x/runs/R/rollouts/c1_0003_step_002000",
                          "param_search": str(tmp_path / "R" / "param_search.json")})()
    assert rollout.ps_share(args) == ("R", 1)
    assert rollout.policy_step({"policy_id": "step_053000"}) == 53000
    assert rollout.policy_step({"policy_id": "/a/checkpoints/pi05/run/53000"}) == 53000
