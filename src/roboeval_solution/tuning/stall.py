"""Stall events of one episode, from what the policy itself observes. Numpy only (used by the py3.10 rollout client).

Events
  old      the served P(success) fell STALL_DROP below its high since the last boost, or stayed flat (range <
           STALL_FLAT over STALL_WINDOW calls) below 0.5
  empty    a gripper closed on nothing: the close command on and the opening <= EMPTY_OPENING for EMPTY_STEPS steps
           after the gripper held something (opening >= HELD_OPENING), or MISS_STEPS steps after the close command
           (not in HOOK_FAMILIES, whose hook / press phase closes on nothing by design); not in the first EMPTY_START
           steps (the grippers settle after the reset and PackBox's press closes from frame 0)
  motion   the hands hover: commanded net displacement over MOTION_STEPS steps < MOTION_EPS m (both hands summed),
           no gripper change (dithering in place: hovers / regrasp loops)
The stage head is not used: its target is "stage j still ahead or reached <= 0.5 s ago", so a low value also
follows every reached stage.

The stall_boost parameter (rollout.py, param search; integer 0..2): 0 off, 1 boost on "old" events, 2 boost on any
event; a boost = the next STALL_BOOST_CALLS policy calls at 1.5x noise temperature (cap 1.2), best of all candidates.
Every event is detected whatever the setting, so the param search learns stall_boost only from episodes with a
stall (stall_seen): the setting can only matter there, and other episodes would only add noise.
"""
from __future__ import annotations

import numpy as np

STALL_DROP, STALL_FLAT, STALL_WINDOW, STALL_BOOST_CALLS = 0.2, 0.02, 6, 3
EMPTY_OPENING, HELD_OPENING, EMPTY_STEPS, MISS_STEPS = 0.05, 0.15, 3, 8
EMPTY_START = 25
MOTION_EPS, MOTION_STEPS = 0.02, 40
HOOK_FAMILIES = ("PickSingleBookFromTable", "StackSingleBookShelf", "PackBox")
NEW_KINDS = ("empty", "motion")


class StallDetector:
    def __init__(self, family: str):
        self.family = family
        self.sp_hist: list[float] = []
        self.closed_prev = np.zeros(2, bool)
        self.since_close = np.zeros(2, int)
        self.held = np.zeros(2, int)  # steps the opening was >= HELD_OPENING under the current close command
        self.empty_n = np.zeros(2, int)
        self.empty_fired = np.zeros(2, bool)
        self.motion: list[float] = []
        self.motion_fired = False
        self.first: dict[str, int] = {}  # event kind -> env step it was first detected

    def _event(self, kind: str, t: int, out: list) -> None:
        self.first.setdefault(kind, int(t))
        out.append(kind)

    def on_call(self, values: dict, t: int, boosting: bool) -> list[str]:
        """Events at a policy call from its served head outputs. `boosting`: a boost is running (the "old" trigger
        does not fire during one)."""
        out: list[str] = []
        sp = values.get("success_pred")
        if sp is not None:
            self.sp_hist.append(float(np.ravel(sp)[0]))
            recent = self.sp_hist[-STALL_WINDOW:]
            if not boosting and (self.sp_hist[-1] < max(self.sp_hist) - STALL_DROP or (
                    len(recent) == STALL_WINDOW and recent[-1] < 0.5 and max(recent) - min(recent) < STALL_FLAT)):
                self.sp_hist = self.sp_hist[-1:]
                self._event("old", t, out)
        return out

    def on_step(self, action: np.ndarray, opening: np.ndarray, t: int) -> list[str]:
        """Events after an env step: the executed action (grippers binarised, 1 = close) and the gripper openings
        observed after it (proprioception_grippers, 1 = open)."""
        out: list[str] = []
        closed = np.asarray(action[12:14]) > 0.5
        opening = np.asarray(opening, np.float64).ravel()[:2]
        for h in (0, 1):
            if not closed[h]:
                self.since_close[h] = self.held[h] = self.empty_n[h] = 0
                self.empty_fired[h] = False
                continue
            self.since_close[h] += 1
            if opening[h] >= HELD_OPENING:
                self.held[h] += 1
                self.empty_n[h] = 0
                continue
            self.empty_n[h] = self.empty_n[h] + 1 if opening[h] <= EMPTY_OPENING else 0
            lost = self.held[h] >= 3 and self.empty_n[h] >= EMPTY_STEPS
            missed = (self.family not in HOOK_FAMILIES and self.held[h] == 0 and self.since_close[h] >= MISS_STEPS
                      and self.empty_n[h] >= EMPTY_STEPS)
            if (lost or missed) and t < EMPTY_START:  # the settling / frame-0 press: consumed until it reopens
                self.empty_fired[h] = True
            if (lost or missed) and not self.empty_fired[h]:
                self.empty_fired[h] = True
                self._event("empty", t, out)
        changed = bool((closed != self.closed_prev).any())
        self.closed_prev = closed
        self.motion.append((np.asarray(action[0:3], np.float64), np.asarray(action[6:9], np.float64), changed))
        win = self.motion[-MOTION_STEPS:]
        if len(win) == MOTION_STEPS:
            net = np.linalg.norm(sum(w[0] for w in win)) + np.linalg.norm(sum(w[1] for w in win))
            if net < MOTION_EPS and not any(w[2] for w in win):
                if not self.motion_fired:
                    self.motion_fired = True
                    self._event("motion", t, out)
            elif net >= 2 * MOTION_EPS:
                self.motion_fired = False  # moving again: a later stall counts again
        return out

    @staticmethod
    def boosts(mode: int, events: list[str]) -> bool:
        """Does this setting (stall_boost 0 / 1 / 2) boost on these events?"""
        return bool(events) and (mode >= 2 or (mode == 1 and "old" in events))

    def summary(self) -> dict:
        new = [self.first[k] for k in NEW_KINDS if k in self.first]
        return {"old": self.first.get("old"), "new": min(new) if new else None, "first": dict(self.first)}
