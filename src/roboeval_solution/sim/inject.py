"""Programmatic hard cases for scripted recovery data (scripts/rollout.py --inject): common failure states, generated
so that the scripted recovery behaviour is learned from them.

The family's scripted controller runs with ONE injected execution error, unrecorded. When the error shows (the first
observable stall event after it, sim/recovery.event_frame: a gripper closed on nothing or lost hold, the hands
hovering, RotateValve's twist jammed; slip / release errors: a fixed delay after them, they are errors by
construction), its actions become the job's prefix and sim/scripted.Retry (open, rise, settle, a fresh scripted
attempt) takes over from that state, recorded like a retry branch of a policy failure. No error state within MAX_WAIT
steps (the error was tolerated), or the episode ending first: the job gives no episode.

Errors (modelled on the policy's failures; arm, size and direction drawn per job from its seed; offsets in the world
frame: x forward, y left, z up; m):
  close    at the arm's k-th close command its gripper stays open STEPS more steps while the arm moves by the offset,
           then closes: too high (valve / book pinch at the fingertips: slips), too low (valve: the twist jams)
  release  at the arm's k-th release its gripper stays closed STEPS more steps while the arm moves by the offset, then
           opens: the top cube set beside / half over the base
  slip     DELAY steps after the arm's k-th close (lifting / carrying / twisting) its gripper opens for STEPS steps,
           then the controller's close command closes it again on nothing: the object slipped out, the hand closed
"""
from __future__ import annotations

import numpy as np

STEPS = 8  # injected motion / held gripper (steps)
MAX_WAIT = 80  # steps after the injection for the error to show
RELEASE_DELAY = 20  # release errors: the branch this many steps after the late release
SLIP_DELAY = 10  # slip errors: the branch this many steps after the hand closed again (on nothing)
UP, DOWN = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
HORIZ = "horizontal"  # a random horizontal direction

# family -> [(kind, arm, k-th close / release of that arm, direction or slip delay (steps lo, hi), (size lo, hi) m)];
# arm: None = either (drawn), "first" = the arm that reaches its k-th event first (PickBook acts with one arm;
# CubeHandover: the giver closes first)
ERRORS = {
    "LiftPot": [("slip", None, 1, (8, 25), None)],
    "LiftTray": [("slip", None, 1, (8, 25), None)],
    "RotateValve": [("close", None, 1, DOWN, (0.015, 0.025)), ("close", None, 1, UP, (0.025, 0.035))],
    "CubeHandover": [("slip", "first", 1, (5, 25), None)],
    "StackTwoBlocks": [("slip", None, 1, (8, 25), None), ("release", None, 1, HORIZ, (0.035, 0.06))],
    "PickSingleBookFromTable": [("slip", "first", 2, (3, 8), None), ("close", "first", 2, UP, (0.02, 0.03))],
    "StackSingleBookShelf": [("slip", "first", 2, (8, 25), None), ("close", "first", 2, UP, (0.02, 0.03))],
}


def _stack_off(env) -> bool:
    """StackTwoBlocks: no cube rests on the other (their heights differ by less than 3 cm of the 4 cm stack)."""
    from roboeval_solution.sim.scripted_common import pose

    z = [pose(b.body)[0][2] for b in env.blocks]
    return abs(z[0] - z[1]) < 0.03


# release errors: the branch only once the error shows in the state (else the job gives no episode)
ERROR_SHOWN = {"StackTwoBlocks": _stack_off}


def draw(family: str, seed: int) -> dict | None:
    """A job's error spec (JSON-able), or None for a family without injected errors."""
    errs = ERRORS.get(family)
    if not errs:
        return None
    rng = np.random.default_rng([int(seed), 0x1A1])
    kind, arm, nth, d, size = errs[int(rng.integers(len(errs)))]
    arm = int(rng.integers(2)) if arm is None else arm  # ("first": resolved by the Injector)
    if kind == "slip":
        return {"kind": kind, "arm": arm, "nth": int(nth), "delay": int(rng.integers(d[0], d[1] + 1))}
    lo, hi = size
    if d == HORIZ:
        a = rng.uniform(0, 2 * np.pi)
        d = (float(np.cos(a)), float(np.sin(a)), 0.0)
    off = np.asarray(d, np.float64) * rng.uniform(lo, hi)
    return {"kind": kind, "arm": arm, "nth": int(nth), "offset": [float(x) for x in off]}


class Injector:
    """Wraps a scripted controller's actions: the spec's error at the arm's k-th close / release command."""

    def __init__(self, spec: dict):
        self.kind, self.nth = spec["kind"], int(spec["nth"])
        self.mode = spec["arm"] if spec["arm"] == "first" else None
        self.arm = 0 if self.mode else int(spec["arm"])  # resolved when the mode's arm reaches its k-th event
        self.counts = np.zeros(2, int)
        self.offset = np.asarray(spec.get("offset", (0.0, 0.0, 0.0)), np.float32)
        self.delay = int(spec.get("delay", 0))
        self.countdown = -1  # slip: steps until the gripper opens
        self.prev = np.zeros(2, bool)  # the controller's previous gripper commands (closed)
        self.window = -1  # steps of the injection left (-1: not reached yet)
        self.start = self.done = None  # env steps the injection began / ended

    def __call__(self, a: np.ndarray, t: int) -> np.ndarray:
        a = np.asarray(a, np.float32).copy()
        closed = a[12:14] > 0.5
        if self.window < 0 and self.start is None and self.countdown < 0:
            fire = None
            for g in (0, 1):
                hit = ((closed[g] and not self.prev[g]) if self.kind in ("close", "slip")
                       else (self.prev[g] and not closed[g]))
                if not hit:
                    continue
                self.counts[g] += 1
                if self.counts[g] != self.nth:
                    continue
                reached = int((self.counts >= self.nth).sum())  # arms at their k-th event now (this one included)
                if (self.mode is None and g == self.arm) or (self.mode == "first" and reached == 1):
                    fire = g
            if fire is not None:
                self.arm = fire
                if self.kind == "slip":
                    self.countdown = self.delay
                else:
                    self.window, self.start = STEPS, t
        h = self.arm
        self.prev = closed
        if self.countdown > 0:
            self.countdown -= 1
            if self.countdown == 0:
                self.window, self.start = STEPS, t
        if self.window > 0 and self.kind == "slip":
            a[12 + h] = 0.0  # the hand lets go; the controller's close command closes it again afterwards
            self.window -= 1
            if self.window == 0:
                self.done = t + 1
        elif self.window > 0:
            a[6 * h: 6 * h + 3] = self.offset / STEPS  # the arm moves by the offset instead of the controller's plan
            a[6 * h + 3: 6 * h + 6] = 0.0
            a[12 + h] = 0.0 if self.kind == "close" else 1.0  # the close comes late / the release comes late
            self.window -= 1
            if self.window == 0:
                self.done = t + 1
        return a


def make_prefix(env, job: dict, policy: str, budget: int, min_left: int, ee_state):
    """(prefix actions up to the error state, info) for an inject job, or (None, reason). Runs on `env` (reset here
    with the job's seed; the caller resets again before replaying the prefix). ee_state(env) -> [12] hand poses."""
    from roboeval_solution.sim import scripted
    from roboeval_solution.sim.recovery import event_frame

    job.setdefault("script_seed", int(job["seed"]) * 16)
    ctl, _ = scripted.make(policy, seed=int(job["script_seed"]), randomize=True)
    inj = Injector(job["inject"])
    obs, _ = env.reset(seed=job["seed"])
    acts, opens, ees = [], [np.asarray(obs["proprioception_grippers"], np.float32)], [ee_state(env)]
    fam = job["family"]
    for t in range(budget - min_left):
        a = np.asarray(ctl(env, t), np.float32)
        a[12:14] = (a[12:14] > 0.5).astype(np.float32)
        a = inj(a, t)
        obs, _, term, trunc, _ = env.step(a)
        if env.success or term or trunc:
            return None, "episode ended before an error state"
        acts.append(a)
        opens.append(np.asarray(obs["proprioception_grippers"], np.float32))
        ees.append(ee_state(env))
        if inj.done is None or t + 1 < inj.done:
            continue
        if inj.kind in ("release", "slip"):
            if t + 1 - inj.done >= (SLIP_DELAY if inj.kind == "slip" else RELEASE_DELAY):
                if inj.kind != "slip" and fam in ERROR_SHOWN and not ERROR_SHOWN[fam](env):
                    return None, "error tolerated (the object still in place)"
                return np.stack(acts), {"event": inj.kind, "branch_t": t + 1, "inject_t": inj.start}
            continue
        if (t + 1 - inj.done) % 3 == 0:
            ev = event_frame(fam, np.stack(acts), np.stack(opens), np.stack(ees[:-1]), after=inj.start)
            if ev is not None:
                return np.stack(acts[: ev[0]]), {"event": ev[1], "branch_t": int(ev[0]), "inject_t": inj.start}
        if t + 1 - inj.done > MAX_WAIT:
            return None, "error tolerated (no stall event)"
    return None, "no injection point / no time left"
