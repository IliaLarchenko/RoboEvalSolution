"""Scripted PackBox policy with privileged state, for kick-start data (scripts/rollout.py --scripted packbox).

It reaches the full reward: success + every stage, including both flap grasps that the demos skip. The grippers
point down with the fingers lined up along the hinge axis.

  1. Both hands press their open flap with closed fingers. Both pads touch it (= holding a flap, stages 1-2) and
     both hands press at the same time for `hold_steps`. Then each lifts off while opening the fingers, rel_l /
     rel_r (0-6) frames after that. The slip metric counts a hold lost while the gripper is not opening, so a
     release with the gripper already open would count as a slip; this one does not. Nothing here senses the
     contact (the policy cannot): a hand is pressing once it rests on the flap (no longer moving down). Instead the
     press itself gets both pads down: the finger axis is kept along the hinge, level with the flap (the far-reach
     lean of the approach would tilt the fingers by up to ~14 deg, one pad short of the flap), and the press point
     keeps the closed fingers well inside the flap: along the hinge at most press_y_far (14 cm) toward its
     robot-side end, 7 cm in from the flap's side edge.
  2. Each hand goes over its flap's tip, comes in under the flap and swings it about the hinge on the back of
     the hand. Far pads lean the hand away from the shoulder (fingertips out), one smooth rule over the reach, so
     far, turned boxes stay inside the arm's workspace. Along the hinge the hand works toward the robot-side end
     for flaps pointing away (reach) and toward the far end for flaps pointing at the robot (clear of its own
     shoulder), one smooth blend over the flap direction.
  3. The right flap goes first: the hand follows it down along the arc and holds it shut near its hinge, clear
     of the left flap and of the left arm. The left hand swings its flap up to `left_wait` deg at the same time,
     waits until the right flap is nearly shut, then follows its own flap down on top -> success. Past upright
     the flap comes down onto the open fingers: the hand is rolled, and slid in from the flap's edge, so it rests
     on one finger well inside the flap (a two-finger contact counts as a hold, its loss without opening as a slip).

Waypoints are 'pad' targets in the box frame: site = pad - PAD along the approach axis (the flap rides on the palm);
the open fingertips are TIP below the site. Side s = +1 for the left arm / flap_2 (box +x), -1 for the right arm /
flap_1 (box -x). The commanded deltas are eased (each step moves `ease` of the way to the wanted delta) for low jerk;
a hand stalled next to its waypoint goes on. About 20% of PackBox seeds start with an unavoidable env collision (the
box spawns in the cabinet door, which the first physics step pushes into the robot torso).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, ee_sites, pose, smoothstep

PAD = 0.058
TIP = 0.112
WRIST = 0.107  # IK site -> wrist (joint 7) along the approach axis
SHOULDER_Z = 0.333  # arm root (link0) -> shoulder (joint 2)
HINGE_X, HINGE_Z = 0.21, 0.144
FLAP_LEN = 0.25
OPEN_DEG = 20.0  # open flaps rest 20 deg above the outward horizontal
SIDE = {0: +1.0, 1: -1.0}  # arm 0 = LEFT -> flap_2, arm 1 = RIGHT -> flap_1
FLAP_IDX = {0: 1, 1: 0}  # packing_box.get_state(): [flap_1 (right), flap_2 (left)], 0 = shut
RIDE_TAGS = ("swing", "wait_right", "follow", "hold_shut")

DEFAULTS = dict(y=0.12, press_r=0.17, press_depth=0.01, press_grip=1.0, pre_tol=0.02, press_y_far=0.14,
                press_still=2, press_level=1.0, hold_steps=10, hold_timeout=55, lift_r=0.20, lift_under=0.03,
                tip_clear=0.02, over_z=0.31, via_tol=0.03, deg_per_step=4.0, top_deg=150.0, left_wait=95.0,
                gate_state=0.35, end_deg=176.0, end_r=0.065, end_under=0.03, end_deg_per_step=6.0, max_dp=0.02,
                max_dr=0.08, swing_tol=0.05, swing_wait=4, y_far=0.20, y_near=-0.12, y_ride_max=0.14, away_lo=0.1,
                away_hi=0.5,
                lean_max=30.0, lean_lo=0.68, lean_hi=0.82, ride_roll=12.0, roll_lo=60.0, roll_hi=100.0, ease=0.5,
                stall_eps=0.002, stall_tol=2.5, stall_n=4, lag=0, rel_l=0, rel_r=0)
CLOSED_TAGS = ("pre_press", "press")  # fingers closed: pressing the open flap (opened as the hand lifts off)
PRESS_TAGS = ("pre_press", "press", "lift_off")  # the finger axis kept level with the flap (press_level)
# per-episode uniform draws (diverse data); ints are drawn as ints
RANDOMIZE = dict(y=(0.09, 0.14), press_r=(0.15, 0.19), press_depth=(0.008, 0.014), hold_steps=(8, 14),
                 lift_r=(0.19, 0.205), lift_under=(0.025, 0.035), tip_clear=(0.015, 0.03), over_z=(0.30, 0.33),
                 deg_per_step=(3.5, 4.5), left_wait=(85.0, 100.0), gate_state=(0.25, 0.45), top_deg=(140.0, 155.0),
                 end_r=(0.05, 0.075), max_dp=(0.017, 0.022),
                 # the policy tends to release both presses on one frame, a hand whose pads missed the flap
                 # included; here one hand starts up to 12 frames late (+ = the right: the early one
                 # keeps pressing until both hold) and each hand releases its own 0-6 frames after both held hold_steps
                 lag=(-12, 12), rel_l=(0, 6), rel_r=(0, 6))


# ── privileged state ──────────────────────────────────────────────────────────

def box_pose(env):
    return pose(env.packing_box.body)


def flap_states(env) -> np.ndarray:
    return np.asarray(env.packing_box.get_state(), float)


def flap_angles(env) -> np.ndarray:
    """[right flap_1, left flap_2] in deg above the outward horizontal (open ~20, upright 90, shut 180)."""
    out = []
    for j in env.packing_box._joints:
        out.append(180.0 - np.degrees(float(j.get_joint_position()) - float(j.mjcf.range[0])))
    return np.asarray(out)


# ── plan ──────────────────────────────────────────────────────────────────────

def _arc(s, p, deg, r=None, under=None):
    """Pad target on the hinge arc: `deg` above the outward horizontal, `under` below the flap's top face."""
    r = p["lift_r"] if r is None else r
    under = p["lift_under"] if under is None else under
    a = np.radians(deg)
    u = np.array([s * np.cos(a), 0.0, np.sin(a)])
    n = np.array([-s * np.sin(a), 0.0, np.cos(a)])
    pt = np.array([s * HINGE_X, 0.0, HINGE_Z]) + r * u - under * n
    pt[1] = p["y"]
    return pt


def _open_point(s, p, depth):
    """Fingertip target `depth` into the open flap's top face at radius press_r and y_press along the hinge (as a
    pad target)."""
    a = np.radians(OPEN_DEG)
    pt = np.array([s * (HINGE_X + p["press_r"] * np.cos(a)), p.get("y_press", p["y"]),
                   HINGE_Z + p["press_r"] * np.sin(a) + 0.004])
    return pt - depth * np.array([-s * np.sin(a), 0, np.cos(a)]) + np.array([0, 0, TIP - PAD])


def _plan(arm, p):
    """[(kind, pad target | event, max steps, tolerance, tag)]"""
    s, y = SIDE[arm], p["y"]
    st = [("go", _open_point(s, p, -0.03), 60, p["pre_tol"], "pre_press"),
          ("hold", _open_point(s, p, p["press_depth"]), p["hold_steps"], p["hold_timeout"], "press"),
          ("go", _open_point(s, p, -0.04) + np.array([0, 0, 0.02]), 15, p["via_tol"], "lift_off")]
    tip_x = HINGE_X + FLAP_LEN * np.cos(np.radians(OPEN_DEG)) + p["tip_clear"]
    start = _arc(s, p, OPEN_DEG)
    st += [("go", np.array([s * tip_x, y, p["over_z"]]), 40, p["via_tol"], "over_tip"),
           ("go", np.array([s * tip_x, y, start[2] - 0.02]), 35, 0.015, "outside_low"),
           ("go", start, 12, 0.01, "under")]
    top = p["top_deg"] if arm == 1 else p["left_wait"]
    # self-paced arc: the next waypoint once the hand is within swing_tol of this one (at most swing_wait steps):
    # full speed while the arm keeps up, slower where it lags (flaps rotated toward the robot)
    sw, stol = p["swing_wait"], p["swing_tol"]
    st += [("go", _arc(s, p, d), sw, stol, "swing") for d in np.arange(OPEN_DEG + p["deg_per_step"], top + 1e-6,
                                                                         p["deg_per_step"])]
    if arm == 0:
        st += [("wait", "R_DOWN", None, None, "wait_right")]
        st += [("go", _arc(s, p, d), sw, stol, "swing")
               for d in np.arange(p["left_wait"] + p["deg_per_step"], p["top_deg"] + 1e-6, p["deg_per_step"])]
    # follow the falling flap down to the shut pose, moving in to radius end_r (clear of the other flap)
    for d in np.arange(p["top_deg"] + p["end_deg_per_step"], p["end_deg"] + 1e-6, p["end_deg_per_step"]):
        f = (d - p["top_deg"]) / (p["end_deg"] - p["top_deg"])
        st.append(("go", _arc(s, p, d, r=p["lift_r"] + f * (p["end_r"] - p["lift_r"]),
                              under=p["lift_under"] + f * (p["end_under"] - p["lift_under"])), sw, stol, "follow"))
    # hold the flap shut to the end, resting on one finger (see RIDE_TAGS)
    st += [("go", _arc(s, p, p["end_deg"], r=p["end_r"], under=p["end_under"]), 10_000, 0.0, "hold_shut")]
    return st


class ScriptedPackBox:
    """controller(env, t) -> 14-d env action (ee deltas + grippers); one instance per episode."""

    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.plans = {a: _plan(a, self.p) for a in (0, 1)}
        self.i = {0: 0, 1: 0}
        self.k = {0: 0, 1: 0}
        self.events = set()
        self.yaw_sign = {}
        self.press_still = {0: 0, 1: 0}  # steps at rest in the press, per hand
        self.both_press, self.t_both = 0, None  # steps both hands pressed at rest; when that reached hold_steps
        self.roll = {0: 0.0, 1: 0.0}  # press roll about the site x axis (fixed: no contact feedback)
        self.offset = None
        self.v = {0: np.zeros(6), 1: np.zeros(6)}  # last commanded ee delta per arm
        self.last_site = {0: None, 1: None}
        self.stall = {0: 0, 1: 0}
        self.y_ride = {0: (self.p["y"], self.p["y"]), 1: (self.p["y"], self.p["y"])}  # hinge position: reach, ride

    def phase(self, arm: int) -> str:
        return self.plans[arm][self.i[arm]][4]

    def _roll_sign(self, env, arm, rt):
        """Sign of a roll about the site x axis that lowers the fingertip on the box +y side."""
        _, rb = box_pose(env)
        dz = (rt * R.from_rotvec([0.1, 0.0, 0.0])).apply([0.0, 1.0, 0.0])[2]  # site +y finger direction after +roll
        return -1.0 if dz * np.dot(rt.apply([0.0, 1.0, 0.0]), rb.apply([0.0, 1.0, 0.0])) > 0 else 1.0

    def _shoulder(self, env, arm):
        if not hasattr(self, "_shoulders"):  # arm root (link0) + the shoulder height above it
            ph = env._mojo.physics
            self._shoulders = [np.asarray(ph.data.xpos[ph.model.name2id(n.replace("\\", "/"), "body")], float)
                               + np.array([0.0, 0.0, SHOULDER_Z]) for n in env._robot.ik_config.arm_roots]
        return self._shoulders[arm]

    def _target(self, env, arm, pad_box, cur_rot, cmd, level=0.0):
        """World (site position, rotation): approach down, fingers along the box y (hinge) axis. Far pads
        lean the approach away from the shoulder (fingertips out, wrist back): one smooth rule over the reach
        a straight-down hand would need at the pad's next commanded position (continuous along the path), so
        far flaps stay inside the arm's workspace. level (0-1): that much of the lean's component along the hinge is
        dropped, so the finger axis stays along the hinge, level with the flap (pressing: both pads on it)."""
        p, rb = box_pose(env)
        pw = rb.apply(pad_box) + p
        sh = self._shoulder(env, arm)
        pad = cmd[:3] + PAD * R.from_rotvec(cmd[3:]).apply([0.0, 0.0, 1.0])
        nxt = pad + np.clip(pw - pad, -self.p["max_dp"], self.p["max_dp"])
        reach = float(np.linalg.norm(nxt + np.array([0.0, 0.0, PAD + WRIST]) - sh))
        lean = np.radians(self.p["lean_max"]) * smoothstep(reach, self.p["lean_lo"], self.p["lean_hi"])
        h = np.array([nxt[0] - sh[0], nxt[1] - sh[1], 0.0]); h /= np.linalg.norm(h)
        z = np.array([0.0, 0.0, -np.cos(lean)]) + np.sin(lean) * h
        yb = rb.apply([0.0, 1.0, 0.0])
        if level:  # the fingers tilt out of level by ~ lean x (h . hinge) otherwise (14 deg at 30 deg, 0.5)
            z = z - level * yb * float(np.dot(z, yb)); z /= np.linalg.norm(z)
        y = yb - z * np.dot(yb, z); y /= np.linalg.norm(y)
        if arm not in self.yaw_sign:  # the fingers are symmetric: the yaw nearest the current one
            c = [R.from_matrix(np.stack([np.cross(s * y, z), s * y, z], 1)) for s in (1.0, -1.0)]
            self.yaw_sign[arm] = 1.0 if (c[0] * cur_rot.inv()).magnitude() <= (c[1] * cur_rot.inv()).magnitude() else -1.0
        y = self.yaw_sign[arm] * y
        return pw - PAD * z, R.from_matrix(np.stack([np.cross(y, z), y, z], 1))

    def __call__(self, env, t: int) -> np.ndarray:
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        if self.offset is None:  # FK (robot base) -> world, measured at rest
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0)
            # the more a flap points away from the robot, the nearer the robot-side end of its hinge the hand works
            # (reach); the more it points toward the robot, the nearer the far end (the hand passes its tip clear of
            # the arm's own shoulder): one smooth blend over the flap's outward direction (x component), no switch
            _, rb = box_pose(env)
            for arm in (0, 1):
                c = float(rb.apply([SIDE[arm], 0.0, 0.0])[0])
                w_far = smoothstep(c, self.p["away_lo"], self.p["away_hi"])
                w_near = smoothstep(-c, self.p["away_lo"], self.p["away_hi"])
                y = self.p["y"] + w_far * (self.p["y_far"] - self.p["y"]) + w_near * (self.p["y_near"] - self.p["y"])
                # the press stays clear of the flap's side edge (the closed fingers 1-2 cm off the hinge position
                # put a pad over the edge at y_far: 1.4 cm in from it): press_y_far for a flap pointing away
                y_press = y + w_far * (self.p["press_y_far"] - self.p["y_far"])
                self.plans[arm] = _plan(arm, {**self.p, "y": y, "y_press": y_press})
                # once the flap is past upright the far reach is no longer needed: the hand slides in from the
                # flap's edge so the finger it rides on stays well inside the flap (see RIDE_TAGS)
                self.y_ride[arm] = (y, y + w_far * (min(self.p["y_far"], self.p["y_ride_max"]) - self.p["y_far"]))
        cmd[:, :3] += self.offset
        a = np.zeros(14, np.float32)
        # pressing: in the press with the hand at rest on the flap (it moves < stall_eps per step: joint state, no
        # contact sensing); once both hands have pressed together for hold_steps, each lets go rel_l / rel_r frames
        # later (its own delay: neither waits for the other's release)
        for arm in (0, 1):
            moved = np.linalg.norm(cur[arm, :3] - self.last_site[arm]) if self.last_site[arm] is not None else 1.0
            press = self.phase(arm) == "press" and self.k[arm] >= 2
            self.press_still[arm] = self.press_still[arm] + 1 if press and moved < self.p["stall_eps"] else 0
        if min(self.press_still.values()) >= self.p["press_still"]:
            self.both_press += 1
            if self.both_press >= self.p["hold_steps"] and self.t_both is None:
                self.t_both = t
        if abs(flap_states(env)[FLAP_IDX[1]]) < self.p["gate_state"]:
            self.events.add("R_DOWN")
        for arm in (0, 1):
            if t < (self.p["lag"] if arm == 1 else -self.p["lag"]):  # the late hand holds still until its start,
                # fingers as its first step has them (closed for the press: the data never shows a press with open
                # fingers, which never holds the flap)
                a[12 + arm] = self.p["press_grip"] if self.plans[arm][self.i[arm]][4] in CLOSED_TAGS else 0.0
                continue
            kind, arg, n, tol, tag = self.plans[arm][self.i[arm]]
            if kind == "wait":
                if arg not in self.events:  # zero delta (eased): keep the commanded pose
                    self.v[arm] *= 1.0 - self.p["ease"]
                    a[6 * arm: 6 * arm + 6] = self.v[arm]
                    continue
                self.i[arm] += 1; self.k[arm] = 0
                kind, arg, n, tol, tag = self.plans[arm][self.i[arm]]
            w_ride = smoothstep(float(flap_angles(env)[FLAP_IDX[arm]]), self.p["roll_lo"], self.p["roll_hi"])
            if tag in RIDE_TAGS:
                y0, y1 = self.y_ride[arm]
                arg = np.array([arg[0], y0 + w_ride * (y1 - y0), arg[2]])
            pos, rt = self._target(env, arm, arg, R.from_rotvec(cur[arm, 3:]), cmd[arm],
                                   level=self.p["press_level"] if tag in PRESS_TAGS else 0.0)
            if kind == "hold" or tag == "lift_off":
                rt = rt * R.from_rotvec([self.roll[arm], 0.0, 0.0])
            elif tag in RIDE_TAGS:
                # the flap comes down onto the open fingers once past upright: roll the hand about its radial axis
                # (smoothly with the flap angle) so it rests on one finger, never on both (a hold whose loss would
                # count as a slip); the robot-side finger (box +y) is the lower one, which also keeps the forearm
                # nearer the robot, away from the other arm
                rt = rt * R.from_rotvec([self._roll_sign(env, arm, rt) * np.radians(self.p["ride_roll"]) * w_ride, 0.0, 0.0])
            dp = np.clip(pos - cmd[arm, :3], -self.p["max_dp"], self.p["max_dp"])
            dr = (rt * R.from_rotvec(cmd[arm, 3:]).inv()).as_rotvec()
            nr = np.linalg.norm(dr)
            if nr > self.p["max_dr"]:
                dr *= self.p["max_dr"] / nr
            # eased velocity: each step moves `ease` of the way from the last delta to the wanted one (no steps in
            # the commanded velocity at waypoint switches -> low jerk)
            self.v[arm] += self.p["ease"] * (np.concatenate([dp, dr]) - self.v[arm])
            a[6 * arm: 6 * arm + 6] = self.v[arm]
            # fingers closed only while pressing the open flap (then opened as the hand lifts off: a release);
            # open from there on (the flap rides on one open finger, see RIDE_TAGS)
            a[12 + arm] = self.p["press_grip"] if tag in CLOSED_TAGS else 0.0
            self.k[arm] += 1
            if kind == "go":
                d = np.linalg.norm(pos - cur[arm, :3])
                # stalled (blocked at the edge of the workspace or resting on the flap) close to the waypoint: go on
                moved = np.linalg.norm(cur[arm, :3] - self.last_site[arm]) if self.last_site[arm] is not None else 1.0
                self.stall[arm] = self.stall[arm] + 1 if moved < self.p["stall_eps"] else 0
                adv = (self.k[arm] >= n or (tol > 0 and d < tol)
                       or (tol > 0 and d < self.p["stall_tol"] * tol and self.stall[arm] >= self.p["stall_n"]))
            else:  # hold: pressing until both hands have pressed together n steps, + this hand's own delay
                adv = (self.t_both is not None and t >= self.t_both + self.p["rel_l" if arm == 0 else "rel_r"]) \
                    or self.k[arm] >= tol
            if adv and self.i[arm] < len(self.plans[arm]) - 1:
                self.i[arm] += 1; self.k[arm] = 0
        self.last_site = {arm: cur[arm, :3].copy() for arm in (0, 1)}
        return a
