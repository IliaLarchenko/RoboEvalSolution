"""Scripted StackSingleBookShelf policy (privileged poses): the PickSingleBookFromTable grasp (scripted_pickbook.py:
push if needed, square edge pinch, pitched approach; the same code AND parameters, pb.GRASP), then carry the book into
the lower shelf compartment and let go on the board.

The shelf's books (yaw within +-30 deg of a long edge toward the robot) are pinched at the middle of that edge by the
shared rule (it moves toward the corner only for books turned > mid_hi): a pinch near a corner lets the book pivot off
its centre of mass in the carry.

  lift   up until the book's lowest point clears the shelf board by `clear`, forward as far as keeps the book's
         far end in front of the board (targets from the commanded pose: the hand sags under the book)
  into   level, toward the compartment: pinch point to x_place, y clipped inside the side walls
  lower  down until the book is 4-8 mm above the board (or touches it), then let go
  open   open the gripper (released on the shelf: stage 4, success), then back off toward the robot
From the lift on, the free arm makes way as the carrying hand comes within avoid_d of it (outward and back, smoothly
with the distance): a carry across the body otherwise drives the working forearm into the free arm's forearm.
"""
from __future__ import annotations

import numpy as np

from roboeval_solution.sim import scripted_pickbook as pb
from roboeval_solution.sim.scripted_common import cmd_pose, ee_sites, holding
from roboeval_solution.sim.scripted_pickbook import PAD, ScriptedPickBook, book_frame

BOARD_Z = 1.026  # lower shelf board top
SHELF_X0 = 0.516  # compartment front
SHELF_Y = (-0.441, 0.572)  # side walls

# the shelf's own parameters: the carry, the placement and the free hand; never a grasp parameter (pb.GRASP: the book
# grasp is the pick task's, one strategy for both book tasks). Shelf-driven settings of the shared grasp: no second
# pull when the centre-of-mass limit leaves it short of the margin anyway (re-closing for it within RoboEval's 20-step
# slip sample counts a slip), the middle of the edge for the shelf's yaws (a corner pinch lets the book pivot in the
# carry), the 1.0 cm overhang clearance, and the push arm rule's book-side preference arm_wy_push 3.3 (with 3 a book
# right of the middle turned ~25 deg goes to the left arm, whose carry crosses the body into the free forearm)
SHELF = dict(carry_pitch=10.0, clear=0.025, x_place=0.56, y_margin=0.16, open_steps=6, vcarry=0.018,
             lower_v=0.014, lower_lead=0.06, touch_n=1, drop_h=0.006, z_max=1.15, min_clear=0.006, far_gap=0.015,
             lift_fwd=0.07, t_lift=120, idle_drop=0.15, stall_n=3, stall_dz=0.0015, avoid_d=0.45, avoid_out=0.30,
             avoid_back=0.15)
SHELF_RANDOMIZE = dict(drop_h=(0.004, 0.008), idle_drop=(0.12, 0.20), carry_pitch=(8.0, 12.0), clear=(0.02, 0.03),
                       x_place=(0.55, 0.575), vcarry=(0.016, 0.020))
assert not set(SHELF) & set(pb.GRASP) and not set(SHELF_RANDOMIZE) & set(pb.GRASP), "the book grasp is shared"
# the pick's grasp + its idle hand (idle_v, idle_out; idle_drop the shelf's) + the shelf's own; the pick's lift (lift,
# pull) is never used here
DEFAULTS = dict(pb.DEFAULTS, **SHELF)
RANDOMIZE = dict(pb.GRASP_RANDOMIZE, **SHELF_RANDOMIZE)


def book_corners(c, rb):
    """The 8 corners of the book's collision hull box: book_frame gives its centre and axes (x thickness, y width,
    z length), pb.HALF its half extents (the hull is off-centre in the mesh frame; book_frame accounts for that)."""
    M = rb.as_matrix()
    return np.array([c + M @ (np.array([sx, sy, sz]) * pb.HALF) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def book_lowest_z(c, rb):
    return float(book_corners(c, rb)[:, 2].min())


def _book_on_board(env):
    return env.books[0].is_colliding(env.book_shelf.lower_shelf_body)


class ScriptedShelfBook(ScriptedPickBook):
    def __init__(self, **params):
        super().__init__(**{**DEFAULTS, **params})

    def __call__(self, env, t):
        if self.phase not in ("lift", "into", "lower", "open", "back"):
            return super().__call__(env, t)  # push / pinch / close (and retry), up to its "lift"
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        cmd[:, :3] += self.offset
        c, rb = book_frame(env)
        arm = self.arm
        centre, ap, f, _, _ = self.plan
        h3 = np.array([ap[0], ap[1], 0.0]) / np.linalg.norm(ap[:2])
        # carrying, the approach flattens to carry_pitch (as the demos): the Panda hand is ~20 cm along its finger
        # axis and only fits the 23 cm compartment with that axis well off vertical; the book turns with it
        phi = np.radians(p["carry_pitch"])
        a_c = np.cos(phi) * h3 + np.sin(phi) * np.array([0, 0, -1.0])
        f_c = np.cos(phi) * np.array([0, 0, 1.0]) + np.sin(phi) * h3
        rt = pb._rot(a_c, f_c)
        ap = a_c
        self.k += 1
        ph, grip, nxt, vmax = self.phase, 1.0, None, p["vcarry"]
        low = book_lowest_z(c, rb)
        target = cur[arm, :3].copy()
        if ph in ("lift", "into"):  # grip lost: the pick-book retry (open, back off, start over)
            self.lost = getattr(self, "lost", 0) + 1 if not holding(env, env.books[0], arm) else 0
            if self.lost >= 4 and self.retries < p["max_retries"]:
                self.retries += 1
                self.retry_from = cur[arm, :3].copy()
                self.phase, self.k = "retry", 0
                return super().__call__(env, t)
        if ph == "lift":
            # from the commanded pose (the arm sags under the book; a target from where it is creeps), at most z_max
            target = cmd[arm, :3].copy()
            target[2] = min(cmd[arm, 2] + (BOARD_Z + p["clear"] - low), p["z_max"])
            # forward as far as keeps the book's far end in front of the board (the arm, folded at the pinch close
            # to its base, reaches high only further out)
            if not hasattr(self, "lift_x0") or self.k <= 1:
                self.lift_x0 = cmd[arm, :2].copy()
            room = SHELF_X0 - p["far_gap"] - book_corners(c, rb)[:, 0].max()  # x the far end may still go
            fwd = float(np.clip(room / max(h3[0], 0.5), -0.01, 0.02))
            if (cmd[arm, :2] - self.lift_x0) @ h3[:2] + fwd > p["lift_fwd"]:
                fwd = p["lift_fwd"] - (cmd[arm, :2] - self.lift_x0) @ h3[:2]
            target[:2] = cmd[arm, :2] + max(fwd, -0.01) * h3[:2]
            self.lows = (getattr(self, "lows", []) if self.k > 1 else []) + [low]
            stalled = len(self.lows) > 10 and self.lows[-1] - self.lows[-11] < 0.002
            if low > BOARD_Z + p["clear"] - 0.015 or (stalled and low > BOARD_Z + p["min_clear"]) or self.k > p["t_lift"]:
                nxt = "into"
                self.carry_z = cmd[arm, 2]
        elif ph == "into":
            pinch = cur[arm, :3] + PAD * ap
            y_goal = float(np.clip(pinch[1], SHELF_Y[0] + p["y_margin"], SHELF_Y[1] - p["y_margin"]))
            target = cur[arm, :3] + np.array([p["x_place"] - pinch[0], y_goal - pinch[1], 0.0])
            target[2] = max(self.carry_z, cur[arm, 2] + (BOARD_Z + p["clear"] - low))
            sag = low < BOARD_Z - 0.005
            if sag:  # the book sags below the board: no further in (back out a little) and lift
                target[:2] = cur[arm, :2] - 0.01 * h3[:2]
                target[2] = cur[arm, 2] + 0.02
            # never lower while sagging (it would let go of the book in front of the shelf)
            elif np.linalg.norm((target - cur[arm, :3])[:2]) < 0.01 or self.k > 90:
                nxt = "lower"
        elif ph == "lower":
            target = cur[arm, :3] - np.array([0, 0, p["lower_lead"]])
            vmax = p["lower_v"]
            touching = _book_on_board(env)
            self.touch_run = getattr(self, "touch_run", 0) + 1 if touching else 0
            # the finger under the book meets the board first (the book stays ~1 cm up): down until the hand stops
            self.zs = (getattr(self, "zs", []) if self.k > 1 else []) + [float(cur[arm, 2])]
            stalled = len(self.zs) > p["stall_n"] and self.zs[-1 - p["stall_n"]] - self.zs[-1] < p["stall_dz"]
            # let go just above the board (the book drops a few mm; the finger under it stays off the board)
            if self.touch_run >= p["touch_n"] or low < BOARD_Z + p["drop_h"] or stalled or self.k > 40:
                nxt = "open"
        elif ph == "open":
            target, grip = cmd[arm, :3], 0.0
            if self.k >= p["open_steps"]:
                nxt = "back"
                self.back_from = cur[arm, :3].copy()
        elif ph == "back":
            target, grip = self.back_from - 0.10 * h3 + [0, 0, 0.02], 0.0
        dp, dr = self._servo(arm, cmd, np.asarray(target, float), rt, vmax)
        a = np.zeros(14, np.float32)
        a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
        a[12 + arm] = grip
        # the free hand makes way as the carrying hand comes near it (a carry across the body otherwise runs into
        # it): outward and back toward the robot, smoothly with the hands' distance
        o = 1 - arm
        side = 1.0 if o == 0 else -1.0
        near = float(np.clip((p["avoid_d"] - np.linalg.norm(cur[arm, :2] - cur[o, :2])) / p["avoid_d"], 0.0, 1.0))
        w = near * near * (3 - 2 * near)
        rest = self.home[o, :3] + [-p["avoid_back"] * w, side * (p["idle_out"] + p["avoid_out"] * w), -p["idle_drop"]]
        do, _ = self._servo(o, cmd, rest, pb.R.from_rotvec(self.home[o, 3:]), max(p["idle_v"], p["vcarry"] * w))
        a[6 * o: 6 * o + 3] = do
        if nxt:
            self.phase, self.k = nxt, 0
        return a
