"""Scripted PickSingleBookFromTable policy (privileged book pose), the demos' strategy made reliable.

The book lies flat on the counter (collision hull 16.4 x 9.4 x 2.65 cm): no top-down grasp fits (8 cm opening). Like
the demos, one hand pinches the part of the book that overhangs the counter's front edge, pitched down, one finger over
the top face and one under the bottom face:

  arm    the one whose outward heading (head_t off +x, mirrored per arm) matches the pinched edge: the sign of its
         inward normal's y; for a push pulled toward the book's own side (arm_wy_push: a far cross-body reach behind
         the book is out of the arm's comfortable range)
  pinch  of the two edges at the corner nearest the robot, the one facing it most, squarely (pads flat along it):
         corner_off along it from that corner, moving smoothly to its middle as it turns parallel to the counter
         edge (its normal's x component from mid_lo to mid_hi); the heading turned toward the arm's outward
         direction (<= max_skew); the deepest pinch (inset, pitch) the overhang allows, one continuous depth, its
         height with it (pinch_dz_min -> pinch_dz below the book's mid-plane); the last approach horizontal at the
         pinch height, the hand height following the measured pads (pad_center); close, lift up and toward the
         robot -> success
  push   if even the shallowest pinch lacks overhang: the same hand, its closed fingers side by side across the
         heading (a thin bar), pitched pp_pitch, reaches over the book, goes down behind its far edge and draws it
         toward the robot until the overhang suffices (keeping its centre of mass com_margin behind the counter
         edge); up off the edge (opening once clear), back over the book no lower than it rose, and the overhang is
         checked again (at most max_pushes pulls)
  idle   the other hand settles idle_drop lower and idle_out outward (closer in height to the working hand: the
         two-arm height / speed differences are scored)

A grip lost while lifting: open, back off, return to the start pose and plan again (max_retries). Servo:
scripted_common.servo with the stopping-distance cap. StackSingleBookShelf (scripted_shelfbook.py) grasps the book
with this code and these parameters (GRASP) as they are.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, ee_sites, holding, servo, smoothstep

EDGE_X = 0.346  # counter front edge (world x)
COUNTER_Z = 0.690  # counter top
# the book's collision hull is not centred on its geom origin: its box spans x (thickness) -0.0134..0.0131,
# y (width) -0.053..0.0411, z (length) -0.0694..0.0949 in the geom frame (9.4 x 16.4 cm, not 10.6 x 19)
HALF = np.array([0.0132, 0.0471, 0.0821])  # half extents of the hull's box: thickness, width, length
BOX_OFF = np.array([-0.0001, -0.0059, 0.0127])  # the box centre in the geom frame
PAD = 0.1034  # site -> pad centres along the approach axis

# The book grasp (arm, push, pinch, close; everything up to the lift), shared AS IS by StackSingleBookShelf
# (scripted_shelfbook.py): one strategy for a given book pose in both book tasks; where it differs between poses, it
# is a smooth function of the pose:
#   where   corner_off from the near corner of the edge facing the robot, smoothly to the edge's middle as the edge
#           turns parallel to the counter edge (its normal's x mid_lo 0.75 -> mid_hi 0.85): the middle for books
#           within ~32 deg of it (all the shelf's books: its carry needs the middle, a corner pinch lets the book
#           pivot), the corner past ~41 deg (the long / short edge switch at 45 deg stays between two corner pinches)
#   depth   the deepest the overhang allows: lam 0..1, inset inset_min 2.0 -> inset 3.2 cm, pitch 48 -> 45 deg
#   height  with the depth: at the book's mid-plane for the shallowest pinch (pinch_dz_min; with the middle pinch at
#           the centre-of-mass push limit a lower finger meets the counter edge), 8 mm below it at the full depth
#           (pinch_dz: margin for the learner, below)
# edge_clear: 1.0 cm of overhang clearance (with less, a second pull at the centre-of-mass limit re-closes the hook
# within RoboEval's slip sample window); repush_min 1.0 cm: no second pull for a few mm. arm_wy_push (the push arm:
# book side vs pinch heading, |hn_y| / |y|) lies between the shelf books' ratio (~3.06: they need the book-side arm,
# the heading arm's carry crosses the body) and the pick books' (>= 3.48: they need the heading arm).
# No draw of where / how deep / how high.
GRASP = dict(com_margin=0.016, inset_min=0.020, pitch_min=48.0, edge_clear=0.010, push_margin=0.016, inset=0.032,
             pitch=45.0, pre_back=0.054, close_steps=14, kp=0.35, vmax=0.018, vpush=0.012, amax=0.004,
             max_dr=0.10, via_tol=0.02, vtransit=0.022, grasp_tol=0.008, settle=4, max_retries=2,
             corner_off=0.03, mid_lo=0.75, mid_hi=0.85, skew_w=1.0, max_skew=30.0, head_t=30.0,
             t_push_pre=60, t_push_over=50, t_push_down=30, t_push=45, ov_margin=0.006, depth_steps=15,
             pad_center=0.3, pad_up=0.025, pad_bias_max=0.025, pad_tol=0.004,
             pp_over=0.03, pp_h=0.026, pp_up=0.05, pp_behind=0.025, pp_pitch=62.0, pp_release=0.015,
             pp_rel_steps=4, pp_vdown=0.03, pp_vend=0.03, pp_kend=1.0, pp_lead=0.004, pp_xcap=0.40, pp_xslope=1.5,
             repush_min=0.010, pp_zcap=1.0, push_min=0.012, arm_wy_push=3.3,
             max_pushes=2, pre_up=0.05, pre_front=0.03, pinch_dz=-0.008, pinch_dz_min=0.0)
# per-episode draws of the grasp: speeds and approach nuisances only (no draw of where / how deep / how high to pinch)
GRASP_RANDOMIZE = dict(push_margin=(0.012, 0.02), pitch=(40.0, 52.0), pre_back=(0.05, 0.058), kp=(0.4, 0.5),
                       vmax=(0.020, 0.024), vtransit=(0.026, 0.030), amax=(0.0045, 0.0055))
# the pick task's own: its lift (up and toward the robot) and the idle free hand
DEFAULTS = dict(GRASP, lift=0.16, pull=0.10, idle_drop=0.08, idle_v=0.008, idle_out=0.05)
RANDOMIZE = dict(GRASP_RANDOMIZE, lift=(0.17, 0.19), pull=(0.03, 0.05))
# Margin for the learner (the policy pinches high and shallow; thin-edge pinches slip): the pinch is planned deep
# (inset 3.2 cm) and 8 mm below the book's mid-plane (pinch_dz), both FIXED (a random pinch depth / height would be
# a hidden variable the policy cannot see). With the deep pinch a long level approach (pre_back > ~6 cm) ends with
# the fingers off the book's faces, hence the narrow pre_back range.


def book_frame(env):
    """(centre, R) of the box of the book's collision hull: columns x = thickness, y = width, z = length."""
    ph = env._mojo.physics
    g = [g for g in env.books[0].body.geoms if g.is_collidable()][0]
    bd = ph.bind(g._mjcf if hasattr(g, "_mjcf") else g.mjcf)
    M = np.asarray(bd.xmat, float).reshape(3, 3)
    return np.asarray(bd.xpos, float) + M @ BOX_OFF, R.from_matrix(M)


def book_com(env):
    """The book's centre of mass (world): that of the body carrying its collision mesh (its parent has no mass)."""
    ph = env._mojo.physics
    g = [g for g in env.books[0].body.geoms if g.is_collidable()][0]
    gid = int(ph.bind(g._mjcf if hasattr(g, "_mjcf") else g.mjcf).element_id)
    return np.asarray(ph.data.xipos[ph.model.geom_bodyid[gid]], float).copy()


def footprint(c, rb):
    """Book corners (xy) and the in-plane axes."""
    M = rb.as_matrix()
    return [c[:2] + sy * HALF[1] * M[:2, 1] + sz * HALF[2] * M[:2, 2] for sy in (-1, 1) for sz in (-1, 1)]


def far_x_at(c, rb, yb):
    """(x, y) of the footprint's far boundary (largest x) on the line y = yb, yb kept inside the footprint."""
    cs = footprint(c, rb)
    cs = [cs[0], cs[1], cs[3], cs[2]]  # around the rectangle
    ys = [q[1] for q in cs]
    yb = float(np.clip(yb, min(ys), max(ys)))
    xs = []
    for q0, q1 in zip(cs, cs[1:] + cs[:1]):
        if (q0[1] - yb) * (q1[1] - yb) <= 0 and abs(q1[1] - q0[1]) > 1e-9:
            xs.append(q0[0] + (yb - q0[1]) / (q1[1] - q0[1]) * (q1[0] - q0[0]))
    return (max(xs) if xs else float(c[0])), yb


def _rot(z, y):
    z = z / np.linalg.norm(z)
    y = y - z * (y @ z)
    y = y / np.linalg.norm(y)
    return R.from_matrix(np.stack([np.cross(y, z), y, z], 1))


class ScriptedPickBook:
    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.phase, self.k = "init", 0
        self.retries = 0
        self.offset = None

    def _pad_clearance(self, env, arm, c, rb):
        """Clearance of the upper finger pad over the book's top face, along the book's normal."""
        from roboeval.const import HandSide

        g = env.robot.grippers[HandSide.LEFT if arm == 0 else HandSide.RIGHT]
        ph = env._mojo.physics
        n = rb.as_matrix()[:, 0]
        n = n if n[2] > 0 else -n
        h = []
        for b in g._pad_bodies:
            xs = [ph.bind(gg._mjcf if hasattr(gg, "_mjcf") else gg.mjcf).xpos for gg in b.geoms if gg.is_collidable()]
            h.append(float(np.dot(np.mean(xs, axis=0) - c, n)))
        return max(h) - HALF[0]

    def _servo(self, arm, cmd, target, rt, vmax=None):
        """scripted_common.servo with the stopping-distance cap (no overshoot at the end of a move)."""
        return servo(self.p, self.v, arm, cmd, target, rt, vmax=vmax, stop_cap=True)

    def _pinch_plan(self, c, rb, inset=None, pitch=None, skew_w=None):
        """(pinch centre, approach a, finger axis f, overhang, required overhang). The edge facing the robot most, at
        the corner nearest it, is pinched squarely, its heading turned toward the arm's outward direction (skew_w of
        the way, at most max_skew). Closing, the lower finger meets the bottom face thickness/2 * tan(pitch) short of
        the centre (horizontally): it must be under the book (inset) and in front of the counter edge (required
        overhang)."""
        inset = self.p["inset"] if inset is None else inset
        phi = np.radians(self.p["pitch"] if pitch is None else pitch)
        skew_w = self.p["skew_w"] if skew_w is None else skew_w
        # pinch an edge squarely (pads flat along it): of the two edges at the corner nearest the robot, the one
        # facing the robot most; corner_off along it from that corner, blended smoothly into its middle as it turns
        # parallel to the counter edge (where the nearest corner switches ends, the pinch is already at the middle)
        M = rb.as_matrix()
        u = M[:2, 1] / np.linalg.norm(M[:2, 1])
        v = M[:2, 2] / np.linalg.norm(M[:2, 2])
        best = None
        for sy in (-1, 1):
            for sz in (-1, 1):
                q = c[:2] + sy * HALF[1] * u + sz * HALF[2] * v
                if best is None or q[0] < best[0][0]:
                    best = (q, sy, sz)
        q, sy, sz = best
        edges = [(-sz * v, -sy * u, 2 * HALF[1]), (-sy * u, -sz * v, 2 * HALF[2])]  # (inward normal, along, length)
        h, along, length = max(edges, key=lambda e: e[0][0])
        off = min(self.p["corner_off"], 0.5 * length)
        base = q + (off + smoothstep(float(h[0]), self.p["mid_lo"], self.p["mid_hi"]) * (0.5 * length - off)) * along
        # heading: toward the arm's outward direction (head_t off +x, mirrored per arm), at most max_skew off the edge
        # normal (the pads stay flat on the book faces for any heading; the wrist stays inside its limits)
        th = np.radians(self.p["head_t"]) * (1.0 if self.arm == 0 else -1.0)
        nat = np.array([np.cos(th), np.sin(th)])
        ang = np.arctan2(h[0] * nat[1] - h[1] * nat[0], h @ nat)
        ang = float(np.clip(skew_w * ang, -np.radians(self.p["max_skew"]), np.radians(self.p["max_skew"])))
        h = np.array([np.cos(ang) * h[0] - np.sin(ang) * h[1], np.sin(ang) * h[0] + np.cos(ang) * h[1]])
        overhang = EDGE_X - base[0]
        centre = base + inset * h
        h3 = np.array([h[0], h[1], 0.0])
        a = np.cos(phi) * h3 + np.sin(phi) * np.array([0, 0, -1.0])
        f = np.cos(phi) * np.array([0, 0, 1.0]) + np.sin(phi) * h3
        # the pinch height with the depth, one continuous rule: pinch_dz (below the book's mid-plane) at the full depth
        # (inset), pinch_dz_min at the shallowest (inset_min, the overhang limit)
        lam = (inset - self.p["inset_min"]) / max(self.p["inset"] - self.p["inset_min"], 1e-6)
        lam = float(np.clip(lam, 0.0, 1.0))
        dz = self.p["pinch_dz_min"] + lam * (self.p["pinch_dz"] - self.p["pinch_dz_min"])
        # the overhang the pinch needs (as if pinching at the book's mid-plane: closing, the lower finger meets the
        # bottom face thickness/2 * tan(pitch) short of the centre; its tip end reaches ~1.2 cm past its pad along the
        # approach, and its 2.1 cm width sticks out toward the counter when the heading is skewed off the edge normal)
        need = (inset - HALF[0] * np.tan(phi) + 0.012 * np.cos(phi)) * h[0] + 0.0105 * abs(h[1]) + self.p["edge_clear"]
        return np.array([centre[0], centre[1], c[2] + dz]), a, f, overhang, need

    def _depth(self, env, c, rb):
        """(inset, pitch, overhang, need, max_push): the deepest pinch the overhang allows as it is (with ov_margin),
        else the deepest a push can make room for (keeping the book's centre of mass com_margin behind the counter
        edge), else the shallowest."""
        p = self.p
        max_push = max(0.0, book_com(env)[0] - (EDGE_X + p["com_margin"]))
        # one continuous pinch depth lam in [0, 1] (inset and pitch from the shallow to the default pinch): the
        # geometry varies smoothly with the book's pose, not in jumps
        lams = np.linspace(1.0, 0.0, int(p["depth_steps"]))
        options = [(p["inset_min"] + l * (p["inset"] - p["inset_min"]),
                    p["pitch_min"] + l * (p["pitch"] - p["pitch_min"])) for l in lams]
        plans = [(ins, pit) + tuple(self._pinch_plan(c, rb, ins, pit)[3:]) for ins, pit in options]
        ok = [pl for pl in plans if pl[2] >= pl[3] + p["ov_margin"]]  # a pinch at the bare minimum overhang fails
        if ok:
            return ok[0] + (max_push,)
        pushable = [pl for pl in plans if pl[3] + p["push_min"] <= pl[2] + max_push]
        return (pushable[0] if pushable else plans[-1]) + (max_push,)

    def __call__(self, env, t):
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        if self.offset is None:
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0)
            self.home = cur.copy()
        cmd[:, :3] += self.offset
        c, rb = book_frame(env)
        a = np.zeros(14, np.float32)
        if self.phase == "init":
            if not hasattr(self, "arm"):
                # the side the pinch heading prefers: the sign of the pinched edge's inward normal y (unskewed)
                self.arm = 0
                hn = self._pinch_plan(c, rb, skew_w=0.0)[1]
                self.arm = 0 if hn[1] / np.linalg.norm(hn[:2]) > 0 else 1
                self.arm_hn = hn
            # the deepest pinch the current overhang allows (no push); else push (keeping the book's centre
            # com_margin behind the edge) for the default pinch, or the shallow one if that is all a push can give
            self.inset, self.pitch, ov, need, max_push = self._depth(env, c, rb)
            self.push_dist = min(max(need - ov + p["push_margin"], 0.0), max_push)
            self.phase = "push_pre" if ov < need + p["ov_margin"] and self.push_dist > 0.004 else "pinch_pre"
            if self.phase == "push_pre" and not getattr(self, "arm_set", False):
                # the push reaches behind the book's far edge: the arm on the book's side unless the pinch heading
                # strongly prefers the other (a far cross-body reach is out of the comfortable range)
                hn = self.arm_hn
                arm2 = 0 if hn[1] / np.linalg.norm(hn[:2]) + p["arm_wy_push"] * c[1] > 0 else 1
                self.arm_set = True
                if arm2 != self.arm:  # re-plan the pinch for the other arm (its heading skew differs)
                    self.arm, self.phase = arm2, "init"
                    return self.__call__(env, t)
            self.arm_set = True
        arm = self.arm
        self.k += 1
        ph, grip, nxt, vmax = self.phase, 0.0, None, None
        target = cmd[arm, :3]
        if ph.startswith("push"):
            # the push in the pinch's heading: the closed hand, pitched pp_pitch (steeper than the pinch: the fingers
            # stand nearly upright behind the far edge and draw it in with a nearly vertical face; at the pinch's own
            # pitch the slanted finger only wedges the edge down onto the counter), reaches over the book, its
            # fingertips go down behind the far edge (on the line through the centre: the book slides without
            # turning) and draw the book in by push_dist; then up, back over the book to the pinch
            grip = 1.0
            if not hasattr(self, "plan"):
                self.plan = self._pinch_plan(c, rb, self.inset, self.pitch)
            _, ap, _, _, _ = self.plan
            hh = np.array([ap[0], ap[1], 0.0]) / np.linalg.norm(ap[:2])
            phk = np.radians(p["pp_pitch"])
            ap = np.cos(phk) * hh + np.sin(phk) * np.array([0, 0, -1.0])
            # closed fingers side by side across the heading: a thin bar behind the edge, the pull loading neither
            # finger toward the other, and a cheap wrist turn on the way from the start to the pinch
            lat = np.array([-hh[1], hh[0], 0.0])
            if not hasattr(self, "hook_sg"):
                cr = R.from_rotvec(cur[arm, 3:])
                self.hook_sg = min((1.0, -1.0), key=lambda sg: (_rot(ap, sg * lat) * cr.inv()).magnitude())
            rt = _rot(ap, self.hook_sg * lat)
            # behind the far boundary on the line through the centre, followed every step
            tip_x, hy = far_x_at(c, rb, c[1])[0] + p["pp_behind"], float(c[1])
            site = lambda tx, tz: np.array([tx, hy, tz]) - PAD * ap
            if ph == "push_pre":  # over the far edge
                target = site(tip_x, COUNTER_Z + 2 * HALF[0] + p["pp_over"])
                # the lower shelf board hangs ~30 cm over the counter from x 0.516: high up, the hand stays in front
                # of it (it reaches forward only as far as it has come down)
                target[0] = min(target[0], p["pp_xcap"] + p["pp_xslope"] * max(0.0, p["pp_zcap"] - cur[arm, 2]))
                vmax = min(p["vtransit"], max(p["pp_vend"], p["pp_kend"] * float(np.linalg.norm(target - cur[arm, :3]))))
                if np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] or self.k > p["t_push_pre"] + p["t_push_over"]:
                    nxt = "push_down"
            elif ph == "push_down":  # down behind it, the fingertips pp_h over the counter
                target = site(tip_x, COUNTER_Z + p["pp_h"])
                vmax = p["pp_vdown"]
                if np.linalg.norm(target - cur[arm, :3]) < 0.01 or self.k > p["t_push_down"]:
                    nxt = "push"
                    self.push_start = c[0]
            elif ph == "push":  # draw the book in until it has moved push_dist (the hand leads by the remaining travel)
                moved = self.push_start - c[0]
                target = site(tip_x, COUNTER_Z + p["pp_h"])
                target[0] = cur[arm, 0] - max(0.0, self.push_dist - moved) - p["pp_lead"]
                vmax = p["vpush"]
                self.dxs = (getattr(self, "dxs", []) if self.k > 1 else []) + [float(cur[arm, 0])]
                stalled = self.k > 6 and abs(self.dxs[-4] - self.dxs[-1]) < 0.001  # the hand stopped
                if moved >= self.push_dist - 0.002 or stalled or self.k > p["t_push"]:
                    nxt = "push_up"
                    self.push_hold = cmd[arm, :3].copy()
            elif ph == "push_up":  # straight up off the edge, opening once clear of it (the fingers open along the
                # pull: opening at the edge shoves the book on)
                grip = 0.0 if cur[arm, 2] > self.push_hold[2] + 0.03 else 1.0
                # back off the edge first, then up (the finger's slanted face would ramp the book on as it rises)
                target = self.push_hold + p["pp_release"] * hh + [0.0, 0.0, p["pp_up"] if self.k > p["pp_rel_steps"] else 0.0]
                if cur[arm, 2] > self.push_hold[2] + p["pp_up"] - 0.015 or self.k > 20:
                    nxt = "pinch_pre"
                    # check the overhang the push left: re-plan the depth, or push once more if it fell short
                    self.n_push = getattr(self, "n_push", 1) + 1
                    self.inset, self.pitch, ov, need, max_push = self._depth(env, c, rb)
                    # again only when the travel the centre-of-mass limit still allows can reach the margin (a push
                    # stopped at that limit leaves a few mm the next one cannot make up)
                    reach = ov + max_push >= need + p["ov_margin"]
                    if ov < need + p["ov_margin"] and self.n_push <= p["max_pushes"] and reach:
                        more = min(max(need - ov + p["push_margin"], 0.0), max_push)
                        if more > p["repush_min"]:
                            nxt, self.push_dist = "push_pre", more
                            if hasattr(self, "plan"):
                                del self.plan
        else:
            if not hasattr(self, "plan") or ph == "pinch_pre":
                self.plan = self._pinch_plan(c, rb, self.inset, self.pitch)
            centre, ap, f, _, _ = self.plan
            # the gripper is symmetric about its approach axis: one convention for every episode (finger axis f)
            rt = _rot(ap, f)
            site = centre - PAD * ap
            if ph in ("pinch_level", "pinch_in"):
                # the fingers straddle the book with ~2 cm to spare above and below; the arm can run 1-2 cm low near
                # its reach (the upper pad then strikes the book's top edge and shoves it along the counter): the
                # hand height follows the measured pads, centring the book between them
                up = self._pad_clearance(env, arm, c, rb)
                self.pad_bias = getattr(self, "pad_bias", 0.0) + p["pad_center"] * max(0.0, p["pad_up"] - up)
                self.pad_bias = float(np.clip(self.pad_bias, 0.0, p["pad_bias_max"]))
                site = site + [0.0, 0.0, self.pad_bias]
            h3 = np.array([ap[0], ap[1], 0.0]) / np.linalg.norm(ap[:2])
            # the last approach is horizontal at the pinch height: one finger over the top face, one under the
            # bottom face (an approach along the tilted axis runs the lower finger into the book's edge)
            if ph == "pinch_pre":
                target = site - p["pre_back"] * h3 + [0, 0, p["pre_up"]]
                if hasattr(self, "push_hold"):
                    # back from behind the book: no lower than the hand came up until it is back in front of the book
                    # (the opened fingers turning toward the pinch pose sweep low)
                    if cur[arm, 0] > target[0] + p["pre_front"]:
                        target[2] = max(target[2], self.push_hold[2] + p["pp_up"])
                if np.linalg.norm(target - cur[arm, :3]) < 0.025 or self.k > 120:
                    nxt = "pinch_level"
            elif ph == "pinch_level":
                target = site - p["pre_back"] * h3
                centred = self._pad_clearance(env, arm, c, rb) > p["pad_up"] - p["pad_tol"]
                if (np.linalg.norm(target - cur[arm, :3]) < 0.01 and centred) or self.k > 30:
                    nxt = "pinch_in"
            elif ph == "pinch_in":
                target = site
                if np.linalg.norm(target - cur[arm, :3]) < p["grasp_tol"] or self.k > 30:
                    nxt = "close"
            elif ph == "close":
                target, grip = site, 1.0
                self.held_run = getattr(self, "held_run", 0) + 1 if holding(env, env.books[0], arm) else 0
                if self.held_run >= p["settle"] or self.k > p["close_steps"]:
                    nxt = "lift"
                    self.lift_from = cur[arm, :3].copy()
            elif ph == "lift":  # up and toward the robot in one move
                grip = 1.0
                target = self.lift_from + [0, 0, p["lift"]] - p["pull"] * h3
        if ph == "lift":
            self.lost = getattr(self, "lost", 0) + 1 if not holding(env, env.books[0], arm) else 0
            if self.lost >= 4 and self.retries < p["max_retries"]:
                nxt = "retry"
                self.retries += 1
                self.retry_from = cur[arm, :3].copy()
        if ph == "retry":  # open, back off toward the robot and up, then start over from the book's new pose
            grip = 0.0
            h3r = np.array([self.plan[1][0], self.plan[1][1], 0.0]) / np.linalg.norm(self.plan[1][:2])
            target = self.retry_from - 0.06 * h3r + [0, 0, 0.05]
            if self.k > 12:
                nxt = "home"
                for attr in ("plan", "held_run", "lost", "pad_bias", "n_push", "hook_sg"):
                    if hasattr(self, attr):
                        delattr(self, attr)
        if vmax is None and ph in ("push_up", "pinch_pre", "retry", "home"):
            vmax = p["vtransit"]
        if ph == "home":  # back to the start pose (a big wrist turn from the pinch pose can jam the arm), re-plan
            grip, target, rt = 0.0, self.home[arm, :3], R.from_rotvec(self.home[arm, 3:])
            if np.linalg.norm(target - cur[arm, :3]) < 0.04 or self.k > 50:
                nxt = "init"
        dp, dr = self._servo(arm, cmd, np.asarray(target, float), rt, vmax)
        a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
        a[12 + arm] = grip
        # the free hand settles into a lower ready pose (as people relax it): closer in height to the working hand
        # (two-arm height / speed differences count as coordination)
        o = 1 - arm
        rest = self.home[o, :3] + [0.0, p["idle_out"] * (1.0 if o == 0 else -1.0), -p["idle_drop"]]
        do, _ = self._servo(o, cmd, rest, R.from_rotvec(self.home[o, 3:]), p["idle_v"])
        a[6 * o: 6 * o + 3] = do
        if nxt:
            self.phase, self.k = nxt, 0
        return a
