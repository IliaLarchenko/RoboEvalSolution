"""Scripted StackTwoBlocks policy with privileged cube poses, for kick-start data (scripts/rollout.py --scripted stack).

It reaches the full reward (success + every stage); stage 2 needs both cubes held at the same time, which the demos
never do (one hand picks a cube and puts it on the other):

  1. each hand grasps the cube on its side, top-down, fingers square to the faces. Of the cube's four 90-deg
     symmetric grasps the left hand takes the one nearest the yaw its arm's swing toward the cube brings it to (the
     wrist need not counter-turn); the right hand's reference is turned the other way by half its swing (a wrist
     turned toward the other arm swings that forearm into it: the IK spreads the turn over the arm's joints); with
     the cubes close, the finger axis goes across the line between them (the hands are long along it). The hand
     stops turning up to grip_gap_deg (17-23 deg) short of the face, toward its natural yaw: less wrist turn and
     the two hands stay closer to parallel (a shorter retreat below); a 4 cm cube in the 8 cm opening still
     closes (the fingers turn it square).
     The right hand (the base, the same hand every episode) grasps first and lifts its cube to the carry height
     (6-10 mm, the only raise: nothing goes up and down again); the left hand comes down to its cube as soon as the
     right one holds and closes once the base cube is up.
     Every close is checked on the hand's observed opening (proprioception, 0.1 steps; a held cube reads 0.5) once
     the fingers stop: below grasp_lo (0.35: closed on an edge, a corner or nothing), or falling below it later
     while closed (the cube slipping out), the hand opens, rises over the cube where it now lies and grasps it again
     (the other hand waits: no lift / hold timeout runs out meanwhile)
  2. both cubes held for hold_n (10-12) steps, the base cube in the air (stage 2): one long, distinct "both held"
     phase
  3. then both hands move at once, toward each other along the line between the cubes (never away: both cubes far
     to one side still meet between them). The meeting point is near the middle (45-55%) pulled toward the table
     centre between the arms (weight 0.25-0.45) and no closer to the top cube than the two hands' footprints allow
     (_meet_point). The right hand carries its cube there, opens over the drop point (the cube drops the 6-10 mm)
     and backs off with 1 cm of rise (_retreat_offset): first ~3.7 cm across its own finger axis (the open fingers
     slide past the cube's faces), on the side away from the left hand, then on to the point that clears the two
     hands' footprints by the shortest path. The left hand meanwhile lifts its cube 1.3-1.7 cm (+ 6 mm of the arm's
     sag under the cube) over the base's top while heading for a standoff point short of the drop point, then
     slides onto the base as the right hand backs off (its target moves with the retreat's progress, so the two
     move together and it never gets ahead of it), keeping its grasp tilt (its yaw turning with half its arm's
     swing: the cube's yaw on the base does not matter, success is contact and no hand holding). Over the base it
     hovers until its cube is centred within 8 mm of the base's (or as near as a hand at rest gets, 15 mm) for
     hover_n (3-5) frames, comes straight down at <= 6 mm per frame still centring, and opens once its cube rests on
     the base within 12 mm (off by more than 15 mm on the way down: back up to the hover, once).
     A hand that has settled short of a target it cannot quite reach (an arm at the edge of its range) moves on
     instead of waiting out the phase's timeout.

Motion: per arm a velocity command toward the target (gain kp, |v| <= vmax 2.8-3.2 cm, |dv| <= amax per step) on the
commanded pose, rotation <= max_dr per step. Heights come from the settled cubes (they spawn 3 cm over the table).
The episode score stays below 1.0: the human reference stacks with one hand (no stage 2), while stage 2 needs both
hands down from the home pose and the base hand out of the way before the top cube comes down.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, down, ee_sites, holding, pose, servo, smoothstep

HALF = 0.02  # cube half size
ARM_BASE = {0: np.array([-0.03, 0.3]), 1: np.array([-0.03, -0.3])}  # shoulder (link0) xy per arm
GRASP_DZ = 0.105  # site above the cube centre at the grasp (demos 0.103-0.108): pads at the cube's mid-height
BASE_ARM = 1  # the right hand holds the base cube in every episode (one pattern to learn)
# hand collision footprint seen from above (hand.stl hull): half length along the finger axis, half width across it
HAND_HALF_LEN, HAND_HALF_WID = 0.103, 0.032

DEFAULTS = dict(pre_dz=0.025, lift_z=0.015, place_gap=0.004, close_steps=12, release_steps=3,
                kp=0.4, vmax=0.030, amax=0.0065, max_dr=0.08, grasp_tol=0.006, via_tol=0.02, over_tol=0.012,
                top_up=0.10, clear_dist=0.055, close_lo=0.18, close_hi=0.32,
                across_w=2.0, tilt_deg=18.0, tilt_sin=0.5, carry_z=0.008, meet_frac=0.5, center_w=0.35,
                wait_off=0.16, carry_tol=0.008, yaw_follow=1.0, yaw_follow_base=-0.5,
                free_yaw=1.0, free_yaw_top=0.5, carry_sag=0.012, hold_n=10, clear_margin=0.015,
                retreat_min=0.085, retreat_max=0.14, grip_gap_deg=20.0, forearm_extra=0.035,
                retreat_dz=0.01, place_reach=0.03, clear_scale=0.7, across_clear=0.037,
                home_bias=2.0, over_sag=0.006,
                # grasp check (the observed opening, 0.1 steps, 1 = open; a held 4 cm cube reads 0.5): a close is
                # judged once the fingers have stopped (the opening changed by <= 0.1 in the last frame; closing on
                # nothing they move ~0.4 per frame); below grasp_lo (or falling below it later while closed: the cube
                # slips out) the hand opens (until open_ok), rises over the cube's current pose and grasps again, at
                # most max_regrasp times
                grasp_lo=0.35, open_ok=0.85, reopen_max=8, max_regrasp=3,
                # aligned hover before the top cube goes down: over the base until the top cube's centre is within
                # align_tol of the base's (or within align_settle with the hand at rest: as near as the arm gets) for
                # hover_n frames (at most align_max frames), then down at <= place_v per step and let go once it rests
                # on the base within drop_tol; off by more than realign_tol on the way down -> back up to the hover
                # (once)
                align_tol=0.008, align_settle=0.015, hover_n=4, align_max=40, place_v=0.006, drop_tol=0.012,
                realign_tol=0.015, max_realign=1)
RANDOMIZE = dict(pre_dz=(0.022, 0.028), lift_z=(0.013, 0.017), place_gap=(0.003, 0.006),
                 kp=(0.36, 0.44), vmax=(0.028, 0.032), amax=(0.006, 0.007),
                 top_up=(0.08, 0.12), carry_z=(0.006, 0.01),
                 meet_frac=(0.45, 0.55), center_w=(0.25, 0.45), wait_off=(0.15, 0.17),
                 # the policy's own both-held overlap is short (a few frames' delay loses stage 2): hold_n counts
                 # only frames with both cubes really held. A longer hold or a late top hand makes its place touch
                 # the retreated base arm more often
                 tilt_deg=(15.0, 21.0), hold_n=(10, 12), clear_margin=(0.01, 0.02),
                 grip_gap_deg=(17.0, 23.0), clear_scale=(0.65, 0.75), retreat_min=(0.08, 0.09), hover_n=(3, 5))
HELD_PHASES = {"lift_b", "hold", "carry", "hold_both", "over", "align", "place"}  # closed on a cube


def _yaw(rot) -> float:
    return float(rot.as_euler("xyz")[2])


class ScriptedStack:
    """controller(env, t) -> 14-d env action; one instance per episode."""

    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.events: set[str] = set()
        self.phase = {0: "pre", 1: "pre"}
        self.k = {0: 0, 1: 0}
        self.finger: dict[int, float] = {}
        self.tilt: dict[int, R] = {}
        self.offset = None
        self.prev_site: dict[int, np.ndarray] = {}
        self.still: dict[int, int] = {}
        self.attempt = {0: 0, 1: 0}  # grasps judged bad so far (regrasps), per arm
        self.replan: set[int] = set()  # arms whose finger yaw / tilt are re-planned at their next grasp
        self.op_prev = {0: None, 1: None}  # last observed opening per hand
        self.align_run = 0
        self.realigns = 0

    def _cube(self, env, i):
        return pose(env.blocks[i].body)

    def _grip_rot(self, env, arm, cur_rot, cur_pos) -> R:
        if arm not in self.finger or arm in self.replan:  # (re-planned for a regrasp from the cube's current pose)
            self.replan.discard(arm)
            c0, rc = self._cube(env, self.cube[arm])
            psi = _yaw(rc)
            yc = cur_rot.as_matrix()[:, 1]
            # the yaw the hand comes to if it keeps its orientation relative to the arm as the arm swings toward the
            # cube (the shoulder turns, the wrist need not counter-turn): less wrist turn, and the forearm stays
            # on its side instead of swinging toward the other arm
            swing = np.arctan2(*(c0[:2] - ARM_BASE[arm])[::-1]) - np.arctan2(*(cur_pos[:2] - ARM_BASE[arm])[::-1])
            line = self._cube(env, self.cube[1 - arm])[0][:2] - self._cube(env, self.cube[arm])[0][:2]
            # the closer the cubes, the more the finger axis goes across the line between them (the hands are wide
            # along it) and the more each hand tilts away from the other: one smooth rule over the cube gap (w = 1
            # close, 0 far), no switch at a threshold distance
            w = 1.0 - smoothstep(float(np.linalg.norm(line)), self.p["close_lo"], self.p["close_hi"])
            follow = self.p["yaw_follow"] if arm != BASE_ARM else self.p["yaw_follow_base"]
            phi = np.arctan2(yc[1], yc[0]) + follow * swing
            cands = [psi + k * np.pi / 2 for k in range(4)]  # the cube's 90 deg symmetry
            cost = np.array([abs((c - phi + np.pi) % (2 * np.pi) - np.pi) for c in cands])  # wrist turn
            cost = cost + w * self.p["across_w"] * np.abs(np.cos(np.asarray(cands) - np.arctan2(line[1], line[0])))
            # the grasp does not need the fingers exactly square to the faces: the hand stops turning up to grip_gap
            # short of the chosen face yaw (toward its natural yaw), saving wrist turn; a 4 cm cube in the 8 cm
            # opening still closes square (the fingers turn it)
            best = cands[int(np.argmin(cost))]
            gap = np.radians(self.p["grip_gap_deg"])
            self.finger[arm] = best + float(np.clip((phi - best + np.pi) % (2 * np.pi) - np.pi, -gap, gap))
            # tilt the hand about its finger axis so the wrist (and the forearm behind it) leans away from the other
            # cube while the pads stay square to the faces: the arms meet near the middle otherwise. A finger axis
            # nearly along the line between the cubes tilts the wrist across it, not away (the side flips there):
            # the tilt fades out as the finger axis turns within asin(tilt_sin) of the line
            self.tilt[arm] = R.identity()
            if w > 0:
                across = abs(np.sin(self.finger[arm] - np.arctan2(line[1], line[0])))
                th = np.radians(self.p["tilt_deg"] * w) * smoothstep(float(across), 0.0, self.p["tilt_sin"])
                cand = [down(self.finger[arm]) * R.from_rotvec([0.0, sg * th, 0.0]) for sg in (1.0, -1.0)]
                toward = [float(np.dot(r.apply([0.0, 0.0, 1.0])[:2], line)) for r in cand]  # approach toward the other
                self.tilt[arm] = R.from_rotvec([0.0, (1.0 if toward[0] > toward[1] else -1.0) * th, 0.0])
        return down(self.finger[arm]) * self.tilt[arm]

    def _retreat_offset(self, env, cur, arm) -> np.ndarray:
        """Where the empty base hand backs off to (and the first leg of the way). Seen from above each hand is a
        HAND_HALF_LEN x HAND_HALF_WID footprint, long along its finger axis; along a direction r the two hands need
        clear_scale x (e_base(r) + e_top(r)) + clear_margin between their centres (clear_scale < 1: the top hand
        rides a cube higher, so the two hands' widest parts overlap only partly in height; calibrated against arm
        contacts). Candidate directions: away from the top hand or sideways to its approach line (it comes over the
        base along that line). The open fingers straddle the base cube along the finger axis, so the hand first
        moves across_clear across that axis (they slide past the cube's faces) and then on to the final point: of
        the candidates the one with the shortest such path, as a smooth soft-min (a mild preference for the base
        arm's own shoulder side breaks ties), with retreat_dz (1 cm) of rise (level, the fingertips graze the
        table). Sets self.retreat_first (the first leg) and returns the final offset. The hand does not turn for it:
        a wrist turn costs about as much joint path per radian as ~12 cm of retreat."""
        p = self.p
        b = self._cube(env, self.cube[arm])[0]
        away = b[:2] - cur[self.top_arm, :2]
        away = away / max(np.linalg.norm(away), 1e-6)
        fingers = [R.from_rotvec(cur[a, 3:]).apply([0.0, 1.0, 0.0])[:2] for a in (arm, self.top_arm)]
        fingers = [f / max(np.linalg.norm(f), 1e-6) for f in fingers]
        n = np.array([-fingers[0][1], fingers[0][0]])  # the base hand's across axis

        def need(r):  # summed half extents of the two footprints along unit direction(s) r [..., 2]
            out = 0.0
            for f in fingers:
                c = np.abs(r @ f)
                out = out + HAND_HALF_LEN * c + HAND_HALF_WID * np.sqrt(np.maximum(0.0, 1.0 - c * c))
            return out

        ang = np.radians(np.arange(0.0, 360.0, 5.0))
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        side = np.array([smoothstep(float(x), -0.15, 0.05) for x in dirs @ away])
        home = ARM_BASE[arm] - b[:2]
        home = home / max(np.linalg.norm(home), 1e-6)

        def d_min(r):  # backing off across its own forearm (not along it, toward its shoulder) the forearm stays
            # beside the drop point, where the top arm's wrist (link6) comes over the base: forearm_extra more there
            # (at 8-9 cm the two can touch, often unseen by the scorer's every-20th-frame contact sampling)
            return p["retreat_min"] + p["forearm_extra"] * np.sqrt(np.maximum(0.0, 1.0 - (r @ home) ** 2))

        d = np.clip(p["clear_scale"] * need(dirs) + p["clear_margin"], d_min(dirs), p["retreat_max"])
        # the first leg goes across on the side away from the top hand (toward the base arm's own shoulder when
        # the across axis is square to it, blended): never into the top hand waiting at its standoff
        lean = float(n @ away) + 0.3 * float(n @ home) * (1.0 - smoothstep(abs(float(n @ away)), 0.1, 0.3))
        sn = n * (1.0 if lean >= 0.0 else -1.0)
        first = sn * p["across_clear"]
        final = dirs * d[:, None]
        along = dirs @ sn
        side = side * np.array([smoothstep(float(x), -0.3, -0.1) for x in along])  # final point on that side
        path = np.where(along * d >= p["across_clear"],  # already that far across: about straight
                        d, p["across_clear"] + np.linalg.norm(final - first[None, :], axis=1))
        w = side * np.exp(-(path - path[side > 0.5].min()) / 0.01 + p["home_bias"] * (dirs @ home))
        r = (w[:, None] * dirs).sum(0)
        r = r / max(np.linalg.norm(r), 1e-6)
        dist = float(np.clip(p["clear_scale"] * need(r) + p["clear_margin"], d_min(r), p["retreat_max"]))
        self.retreat_d = dist
        self.retreat_first = np.array([first[0], first[1], p["retreat_dz"]])
        return np.array([r[0] * dist, r[1] * dist, p["retreat_dz"]])

    def _settle(self, arm: int, cur) -> int:
        """Steps arm `arm`'s measured hand has stayed within 2 mm per step (call once per arm per step): a target the
        arm cannot quite reach (IK at the edge of its range, a cube in the way) is not waited out to a timeout."""
        prev = self.prev_site.get(arm, cur[arm, :3])
        self.still[arm] = self.still.get(arm, 0) + 1 if np.linalg.norm(cur[arm, :3] - prev) < 0.002 else 0
        self.prev_site[arm] = cur[arm, :3].copy()
        return self.still[arm]

    def _meet_point(self, env):
        """Where the base cube is set down (and the top cube follows): on the straight line between the two cubes,
        so each hand only moves toward the other (never away, e.g. when both cubes start far to one side), near the
        middle (meet_frac) pulled toward the table centre between the arms (y = 0, weight center_w), and no closer to
        the top cube than the two hands' footprints along that line (as grasped) plus clear_margin. Smooth in the
        cube poses (a projection and clips). Also sets u_top / top_reach: the top hand waits on that line."""
        p = self.p
        b = self._cube(env, self.cube[self.base_arm])[0]
        t = self._cube(env, self.cube[self.top_arm])[0]
        d = (t - b)[:2]
        gap = max(float(np.linalg.norm(d)), 1e-6)
        u = d / gap
        need = 0.0
        for a in (0, 1):
            f = np.array([np.cos(self.finger[a]), np.sin(self.finger[a])])
            c = abs(float(f @ u))
            need += HAND_HALF_LEN * c + HAND_HALF_WID * np.sqrt(max(0.0, 1.0 - c * c))
        s_max = max(0.0, (gap - need - p["clear_margin"]) / gap)
        centre = np.array([0.5 * (b[0] + t[0]), 0.0])
        s_c = float(np.clip((centre - b[:2]) @ d / gap ** 2, 0.0, 1.0))
        frac = min((1.0 - p["center_w"]) * p["meet_frac"] + p["center_w"] * s_c, s_max)
        drop = b.copy()
        drop[:2] = b[:2] + frac * d
        drop[2] = self.z0
        self.drop = drop
        self.u_top = (t - drop)[:2] / max(float(np.linalg.norm((t - drop)[:2])), 1e-6)
        self.top_reach = max(float(np.linalg.norm((t - drop)[:2])), p["retreat_min"])

    def _standoff(self, cur) -> float:
        """How far from the drop point the top hand waits (along u_top, toward its own cube): the two hand
        footprints along that line plus clear_margin, at most wait_off and no further out than its cube was."""
        p = self.p
        need = 0.0
        for a in (0, 1):
            f = R.from_rotvec(cur[a, 3:]).apply([0.0, 1.0, 0.0])[:2]
            c = abs(float(np.dot(f, self.u_top))) / max(np.linalg.norm(f), 1e-6)
            need += HAND_HALF_LEN * c + HAND_HALF_WID * np.sqrt(max(0.0, 1.0 - c * c))
        return float(np.clip(need + p["clear_margin"], p["retreat_min"], min(p["wait_off"], self.top_reach)))

    def _site_over(self, c, arm, dz):
        """Site position with the pads at the cube centre (dz = GRASP_DZ) or dz along the approach axis above it."""
        z = (down(self.finger[arm]) * self.tilt[arm]).apply([0.0, 0.0, 1.0])
        return c - dz * z

    def __call__(self, env, t: int) -> np.ndarray:
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        if self.offset is None:
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0)
            ys = [self._cube(env, i)[0][1] for i in range(2)]
            left = int(np.argmax(ys))
            self.cube = {0: left, 1: 1 - left}  # each hand the cube on its side
            self.base_arm = BASE_ARM
            self.top_arm = 1 - self.base_arm
            b0 = self._cube(env, self.cube[self.base_arm])[0]
            t0 = self._cube(env, self.cube[self.top_arm])[0]
            # resting cube centre height: the cubes spawn 3 cm over the table and drop onto it in the first steps
            # (the lowest height seen before the grasps), so heights are not taken from the spawn pose
            self.z0 = float(b0[2])
            self.drop = None  # the meeting point: set when the carry starts (_meet_point, needs both grasps)
            self.retreat_d = p["retreat_max"]
        cmd[:, :3] += self.offset
        if not self.events:
            self.z0 = min(self.z0, *(self._cube(env, i)[0][2] for i in range(2)))
        a = np.zeros(14, np.float32)
        op = np.asarray(env.robot.qpos_grippers, float)  # observed openings [left, right]
        stopped = {}  # the fingers have stopped: the opening changed by at most one 0.1 step since the last frame
        for arm in (0, 1):
            stopped[arm] = self.op_prev[arm] is not None and abs(op[arm] - self.op_prev[arm]) < 0.15
            self.op_prev[arm] = op[arm]
        for arm in (0, 1):
            ci = self.cube[arm]
            rt = self._grip_rot(env, arm, R.from_rotvec(cur[arm, 3:]), cur[arm, :3])
            c = self._cube(env, ci)[0]
            ph, grip, nxt = self.phase[arm], 0.0, None
            settled = self._settle(arm, cur) >= 4
            self.k[arm] += 1
            k = self.k[arm]
            target = cmd[arm, :3]
            vmax, stop_cap = None, False
            if (ph in HELD_PHASES and op[arm] < p["grasp_lo"] and self.attempt[arm] < p["max_regrasp"]
                    and not (arm == self.top_arm and ph == "place" and c[2] - self._cube(env, self.cube[
                        self.base_arm])[0][2] < 2 * HALF + 0.006)):
                # closed and the opening fell below a cube's width: it slipped out (or was never in): open, grasp again
                ph = self._regrasp(arm)
                k = self.k[arm]
            if ph == "pre":
                target = self._site_over(c, arm, GRASP_DZ + p["pre_dz"])
                if np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] or k > 80:
                    nxt = "descend" if arm == self.base_arm else "hover"
            elif ph == "descend":  # (the top hand waits down here, open, until the base cube is up)
                target = self._site_over(c, arm, GRASP_DZ)
                near = np.linalg.norm(target - cur[arm, :3])
                if ((near < p["grasp_tol"] or (settled and near < 0.02) or k > 30)
                        and (arm == self.base_arm or "base_up" in self.events)):
                    nxt = "close"
            elif ph == "hover":  # top hand: at its pre-grasp point until the base hand has its cube
                target = self._site_over(c, arm, GRASP_DZ + p["pre_dz"])
                if "held1" in self.events:
                    nxt = "descend"
            elif ph == "close":
                target, grip = self._site_over(c, arm, GRASP_DZ), 1.0
                # judged on the observed opening once it has settled: a cube between the fingers (>= grasp_lo) or not
                if (k >= 2 and stopped[arm]) or k > p["close_steps"]:
                    if op[arm] < p["grasp_lo"] and self.attempt[arm] < p["max_regrasp"]:
                        nxt = self._regrasp(arm, now=False)
                    else:
                        self.events.add(f"held{arm}")
                        if arm == self.base_arm:
                            nxt = "lift_b"
                        else:  # (a regrasp after the hold goes straight on toward the base)
                            nxt = "over" if "hold_done" in self.events else "hold_both"
            elif ph == "reopen":  # a bad grasp: open where it is, then up over the cube's current pose and again
                target = cmd[arm, :3]
                if op[arm] >= p["open_ok"] or k >= p["reopen_max"]:
                    nxt = "pre"
            elif ph == "lift_b":  # base hand: its cube up to the carry height first (the top hand grasps only then)
                grip = 1.0  # one raise: the carry keeps this height until the cube is set down at the drop point
                target = self._site_over(c, arm, GRASP_DZ) + [0, 0, p["carry_z"] + p["carry_sag"]]
                if c[2] - self.z0 > 0.6 * p["carry_z"] or k > 25:
                    self.events.add("base_up")
                    nxt = "hold"
            elif ph == "hold":  # base hand: holds its cube up while the top hand grasps and both hold hold_n steps
                grip = 1.0
                target = self.hold_at if hasattr(self, "hold_at") else cmd[arm, :3]
                self.hold_at = target
                if "hold_done" in self.events:
                    nxt = "carry"
            elif ph == "hold_both":  # top hand: holds its cube on the table; both cubes held for hold_n steps
                target, grip = self._site_over(c, arm, GRASP_DZ), 1.0
                both = holding(env, env.blocks[ci], arm) and holding(env, env.blocks[self.cube[1 - arm]], 1 - arm)
                if self.phase[1 - arm] != "hold":  # the base hand grasps its cube again: the hold timeout restarts
                    self.k[arm] = k = 0
                self.both_run = getattr(self, "both_run", 0) + int(both)
                if self.both_run >= p["hold_n"] or k >= p["hold_n"] + 30:  # both move from here: the base to the
                    # drop point, the top up and toward it
                    self.events.add("hold_done")
                    if self.drop is None:
                        self._meet_point(env)
                    nxt = "over"  # up and toward its standoff in one diagonal move (no straight-up lift first)
            elif ph == "carry":  # base hand: the cube up a little and over the drop point, in one move
                grip = 1.0
                if self.drop is None:
                    self._meet_point(env)
                target = self.drop + [0, 0, p["carry_z"] + p["carry_sag"]] + (cur[arm, :3] - c)
                # off the table by a few mm is enough (holding a cube the arm sags ~1.3 cm below its command)
                off_drop = np.linalg.norm((self.drop - c)[:2])
                if ((off_drop < p["carry_tol"] or (settled and off_drop < 0.02)) and c[2] - self.z0 > 0.004) or k > 80:
                    # over the drop point: let go at the carry height (6-10 mm) and the cube drops onto the table
                    nxt = "release"
            elif ph == "release":  # base hand: open, then back off sideways toward its own side
                target = cmd[arm, :3]
                if k >= p["release_steps"]:
                    self.retreat_from = cur[arm, :3].copy()
                    self.retreat_off = self._retreat_offset(env, cur, arm)
                    self.retreat = self.retreat_from + self.retreat_off
                    nxt = "retreat"
            elif ph == "retreat":  # first across (the open fingers slide past the cube's faces), then on
                done_first = float(np.dot((cur[arm, :2] - self.retreat_from[:2]), self.retreat_first[:2])) / max(
                    float(np.dot(self.retreat_first[:2], self.retreat_first[:2])), 1e-9)
                g = smoothstep(done_first, 0.6, 1.0)
                target = self.retreat_from + (1.0 - g) * self.retreat_first + g * self.retreat_off
                if np.linalg.norm((cur[arm, :3] - c)[:2]) > max(p["clear_dist"], 0.8 * self.retreat_d):
                    self.events.add("base_clear")
                if np.linalg.norm((self.retreat - cur[arm, :3])[:2]) < 0.008 or (settled and g > 0.99):
                    self.events.add("base_clear")
                    nxt = "done"
            elif ph == "over":
                grip = 1.0
                b = self._cube(env, self.cube[self.base_arm])[0]
                # the carried cube heads for a standoff point short of the drop point (the two hand footprints
                # clear, as for the retreat) and slides from there onto the base as the base hand backs off: both
                # move at once, and the top hand can never get ahead of the retreat
                wait = self.drop + self._standoff(cur) * np.array([self.u_top[0], self.u_top[1], 0.0])
                backed = 0.0  # the base hand's progress along its own planned retreat (not off the cube: it may shift)
                if self.phase[self.base_arm] == "done":
                    backed = 1.0
                elif self.phase[self.base_arm] == "retreat":
                    left = float(np.linalg.norm((self.retreat - cur[self.base_arm, :3])[:2]))
                    backed = 1.0 - left / max(float(np.linalg.norm(self.retreat_off[:2])), 1e-6)
                frac = smoothstep(backed, 0.35, 0.9)
                aim = wait + frac * (b - wait)
                target = aim + (cur[arm, :3] - c)  # carried cube over the aim point
                # over_sag: part of the ~1.3 cm the arm sags under the cube, so the carried cube clears the base's
                # top edge by ~lift_z instead of brushing it
                target[2] = self.z0 + 2 * HALF + p["lift_z"] + p["over_sag"] + (cur[arm, 2] - c[2])
                gap_xy = float(np.linalg.norm((target - cur[arm, :3])[:2]))
                # settled short of the point (the arm cannot quite reach it): within place_reach is close enough,
                # a 4 cm cube rests on the base off by a cm or two
                if frac > 0.98 and (gap_xy < p["over_tol"] or (settled and gap_xy < p["place_reach"])):
                    nxt = "align"
                if k > 150:
                    nxt = "align"
            elif ph == "align":  # hover over the base until the carried cube is centred over it for hover_n frames
                grip, stop_cap = 1.0, True
                b = self._cube(env, self.cube[self.base_arm])[0]
                target = b + (cur[arm, :3] - c)
                target[2] = self.z0 + 2 * HALF + p["lift_z"] + p["over_sag"] + (cur[arm, 2] - c[2])
                off = float(np.linalg.norm((c - b)[:2]))
                aligned = off < p["align_tol"] or (settled and off < p["align_settle"])
                self.align_run = self.align_run + 1 if aligned else 0
                if self.align_run >= p["hover_n"] or k > p["align_max"]:
                    nxt = "place"
            elif ph == "place":  # straight down onto the base, slowly, still centring; let go once it rests there
                grip, stop_cap, vmax = 1.0, True, p["place_v"]
                b = self._cube(env, self.cube[self.base_arm])[0]
                target = b + [0, 0, 2 * HALF + p["place_gap"]] + (cur[arm, :3] - c)
                off = float(np.linalg.norm((c - b)[:2]))
                if (c[2] - b[2] < 2 * HALF + 0.006 and off < p["drop_tol"]) or k > 40:
                    nxt = "drop"
                elif off > p["realign_tol"] and self.realigns < p["max_realign"]:
                    self.realigns += 1
                    self.align_run = 0
                    nxt = "align"
            elif ph == "drop":
                target = cmd[arm, :3]
                if k >= p["release_steps"]:
                    self.retreat_top = cur[arm, :3] + [0, 0, p["top_up"]]
                    nxt = "up"
            elif ph == "up":
                target = self.retreat_top
            if arm == self.top_arm and ph in ("over", "align", "place", "drop"):
                # the carried cube's yaw does not matter (any yaw rests on the base): the hand turns with the swing
                az = float(np.arctan2(*(cur[arm, :2] - ARM_BASE[arm])[::-1]))
                if not hasattr(self, "az_top"):
                    self.az_top = (az, rt)
                rt = R.from_rotvec([0.0, 0.0, p["free_yaw_top"] * (az - self.az_top[0])]) * self.az_top[1]
            if ph in ("retreat", "done"):
                # the empty hand's yaw is free: it turns with the arm's swing (the wrist need not counter-turn)
                az = float(np.arctan2(*(cur[arm, :2] - ARM_BASE[arm])[::-1]))
                if not hasattr(self, "az_rel"):
                    self.az_rel = (az, R.from_rotvec(cmd[arm, 3:]))
                rt = R.from_rotvec([0.0, 0.0, p["free_yaw"] * (az - self.az_rel[0])]) * self.az_rel[1]
            dp, dr = servo(p, self.v, arm, cmd, np.asarray(target, float), rt, vmax=vmax, stop_cap=stop_cap)
            a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
            a[12 + arm] = grip
            if nxt:
                self.phase[arm], self.k[arm] = nxt, 0
                self.still[arm] = 0  # "settled" counts within a phase only (a hand that held still before it starts)
        return a

    def takeover(self, env, last_action) -> None:
        """Recovery's track stage: this controller, advanced on its own reference run, takes over another run's
        state (a policy episode at its deviation from the scripted path). The velocity state continues the hands'
        last commanded steps (no jerk); a hand past its grasp keeps the wrist it has (its cube is held, or let go, at
        the policy's own yaw: no wrist snap to this run's planned grasp); with the base cube already set down, the
        drop point is where it lies (the top hand's standoff line from there)."""
        cmd = cmd_pose(env)
        cur = ee_sites(env)
        a = np.asarray(last_action, np.float64)
        for arm in (0, 1):
            self.v[arm] = a[6 * arm: 6 * arm + 3]
            self.prev_site[arm], self.still[arm] = cur[arm, :3].copy(), 0
            self.op_prev[arm] = float(np.asarray(env.robot.qpos_grippers, float)[arm])
            if arm in self.finger and self.phase[arm] not in ("pre", "descend", "hover", "close", "reopen"):
                self.tilt[arm] = down(self.finger[arm]).inv() * R.from_rotvec(cmd[arm, 3:])
        for attr in ("az_top", "az_rel"):  # re-anchored on the current wrists at their phases' next step
            if hasattr(self, attr):
                delattr(self, attr)
        if self.phase[self.base_arm] in ("release", "retreat", "done") and self.drop is not None:
            b = self._cube(env, self.cube[self.base_arm])[0]
            t = self._cube(env, self.cube[self.top_arm])[0]
            self.drop[:2] = b[:2]
            self.u_top = (t - self.drop)[:2] / max(float(np.linalg.norm((t - self.drop)[:2])), 1e-6)
            self.top_reach = max(float(np.linalg.norm((t - self.drop)[:2])), self.p["retreat_min"])

    def _regrasp(self, arm: int, now: bool = True) -> str:
        """A bad grasp of arm `arm` (the opening says no cube between the fingers, or one slipping out): it opens
        (phase "reopen"), rises over the cube where it now lies and grasps it again (pre, descend, close), its finger
        yaw re-planned from the cube's current yaw. While the base hand grasps again the top hand waits (no base_up).
        now: switch at once (from a holding phase); else returns the phase for the caller to switch to."""
        self.attempt[arm] += 1
        self.replan.add(arm)
        if arm == self.base_arm:
            self.events.discard("base_up")
            if hasattr(self, "hold_at"):
                del self.hold_at
        elif hasattr(self, "az_top"):  # the carried cube's yaw is re-planned with the new grasp
            del self.az_top
        if now:
            self.phase[arm], self.k[arm] = "reopen", 1
            self.still[arm] = 0
        return "reopen"
