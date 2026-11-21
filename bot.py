"""Mech Royale bot: survival-first battle royale AI.

Priorities each decision (10 Hz):
  1. stay inside the zone (the next circle, early)
  2. fight what is shooting at us: face it (front armor), lead shots, strafe, use cover
  3. retreat behind cover to regenerate when hurt
  4. otherwise grow: better weapons, cubes, crates, NPC drones, supply drops
"""
from __future__ import annotations

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mechfight as mf  # noqa: E402
from world import World  # noqa: E402

TAU = 2 * math.pi
TRACE = os.environ.get("MB_TRACE") == "1"
# tunables (overridable via env for local A/B tests)
P = {
    "retreat_hp": 0.25, "retreat_hp_multi": 0.30, "recover_hp": 0.85,
    "pref_laser": 220.0, "pref_gun": 160.0, "pref_lightning": 28.0,
    "cube_fight_r": 30.0, "weapon_fight_r": 45.0,
    "ffa_evidence": 4.0, "ffa_deaths": 4.0, "ffa_prey_hp": 350.0, "ffa_engage_r": 70.0, "ffa_retreat_hp": 0.5,
    "ffa_retreat_hp_multi": 0.65, "ffa_explore_until": 100.0,
    "astar_budget": 0.012, "climb_max": 0.0, "loot_timeout": 20.0,
    "heal_still_hp": 0.95, "seek_hp": 2.0, "lead_mem": 15.0, "hunt_hp": 2.0, "hunt_mem": 10.0, "hunt_r": 220.0, "explore_until": 240.0,
    "latency": 0.08, "inst_lead": 0.1,
    "strafe_min": 0.5, "strafe_span": 1.0, "retreat_threat": 1.0,
    "farm_crate_r": 70.0, "farm_npc_r": 75.0, "farm_crate_hp": 0.45, "farm_npc_hp": 0.5,
    "arm_r": 220.0, "arm_fight_r": 50.0, "arm_fight_r2": 90.0, "vel_window": 1.2, "vel_blend_t": 0.8,
    "hp_w": 0.12, "dist_w": 1.0, "v_lightning": 3.0, "v_laser": 4.0, "v_gun": 2.0,
}
for _k in list(P):
    if os.environ.get("MB_" + _k.upper()):
        P[_k] = float(os.environ["MB_" + _k.upper()])
WEAPON_VALUE = {"laser": 4.0, "lightning": 3.0, "gun": 2.0, "mortar": 1.0}
WEAPON_VALUE.update(laser=P["v_laser"], lightning=P["v_lightning"], gun=P["v_gun"])
UPGRADE_PRIORITY = [
    "armor", "plating", "nanites", "laser_damage", "laser_charge", "lightning_efficiency",
    "gun_damage", "gun_rate", "servos", "lightning_range", "gun_mag", "gun_velocity",
    "capacitor", "reactor", "jets", "gyros", "lightning_chain", "laser_pierce",
    "sensors", "mortar_reload", "mortar_velocity", "mortar_splash",
]


def wrap(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


def hyp(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm2(x: float, y: float):
    d = math.hypot(x, y)
    if d < 1e-9:
        return 0.0, 0.0
    return x / d, y / d


class Bot:
    name = "survivor"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.slot = hello["slot"]
        self.rng = random.Random(hello.get("seed", 0))
        mech = self.cfg["mech"]
        map_dir = hello.get("map", {}).get("dir", "/arena")
        if not os.path.exists(os.path.join(map_dir, "v2", "world.json")):
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena")
            map_dir = here
        self.W = World(map_dir, step_up=mech.get("step_up", 1.2), max_slope_deg=mech.get("max_slope_deg", 50),
                       mech_height=mech.get("height", 9))
        import gc
        gc.collect()
        gc.freeze()                     # map arrays are permanent: keep the collector off them
        gc.set_threshold(50000, 50, 100)
        self.wcfg = self.cfg["weapons"]
        self.path = None
        self.path_goal = None
        self.path_t = -99.0
        self.path_i = 0
        self.strafe = 1.0
        self.strafe_t = 0.0
        self.last_hit_t = -99.0      # last time we took weapon damage
        self.last_dmg_bearing = None
        self.last_dmg_by = None
        self.last_deal_t = -99.0
        self.threat_ids = {}          # id -> last time it damaged us
        self.retreating = False
        self.cover = None
        self.cover_t = -99.0
        self.stuck_since = None
        self.unstick_until = -1.0
        self.unstick_dir = None
        self.hist = []
        self.scan_phase = self.rng.random() * TAU
        self.last_seen = {}           # id -> (t, pos, vel, hp)
        self.sound_focus = None       # (t, bearing)
        self.mode = ""
        self.blocked_pickups = {}
        self.target_id = None
        self.charge_hold = 0.0
        self.vhist = {}
        self.ffa = os.environ.get("MB_FORCE_FFA") == "1"
        self.ffa_evidence = 0.0
        self.prev_hp = {}
        self.ffa_victims = set()
        self.my_kills = 0
        self.n_mechs = int(self.cfg["match"].get("mechs", 16))
        self.mem_pk = {}      # id -> pickup dict   (items don't move: remember them across glances)
        self.mem_cube = {}    # id -> cube dict
        self.mem_crate = {}   # id -> crate dict
        self.mem_npc = {}     # id -> {"id", "pos", "hp", "t"}
        self.loot_commit = None
        self.loot_since = 0.0
        self.partial_goal = None
        self.nav_jump = False
        self.nav_boost = False
        self.climb_base = {}
        self.climb_fail = {}     # roof goal cell -> failed attempts     # roof goal cell -> ground approach point
        self.unreachable = []    # (t, x, y) goals the pathfinder could not reach
        self.sound_leads = []   # (t, x, y): estimated positions of things we heard
        self.explore = None
        self.explore_until = -1.0

    # ------------------------------------------------------------------ helpers
    def zone_now(self, z, t):
        if t <= z["shrink_start"]:
            return z["center"], z["radius"]
        if t >= z["shrink_end"]:
            return z["next_center"], z["next_radius"]
        f = (t - z["shrink_start"]) / max(1e-6, z["shrink_end"] - z["shrink_start"])
        c = [z["center"][0] + (z["next_center"][0] - z["center"][0]) * f,
             z["center"][1] + (z["next_center"][1] - z["center"][1]) * f]
        return c, z["radius"] + (z["next_radius"] - z["radius"]) * f

    def safe_inside(self, p, z, t, margin=0.0, horizon=0.0):
        """p inside the zone now and `horizon` seconds from now."""
        for dt in (0.0, horizon):
            c, r = self.zone_now(z, t + dt)
            if hyp(p, c) > r - margin:
                return False
        return True

    def zone_home(self, me_xy, z, t):
        """Point we want to hold: inside the next circle, near us, not dead center."""
        nc, nr = z["next_center"], z["next_radius"]
        if nr <= 1:
            return list(nc)
        margin = min(max(20.0, nr * 0.35), nr * 0.7)
        d = hyp(me_xy, nc)
        lim = nr - margin
        if d <= lim:
            return [me_xy[0], me_xy[1]]
        ux, uy = norm2(me_xy[0] - nc[0], me_xy[1] - nc[1])
        return [nc[0] + ux * lim, nc[1] + uy * lim]

    def smooth_vel(self, m, flight):
        """Blend instantaneous velocity with the recent average: strafing targets revert to their mean."""
        v = m.get("vel", [0.0, 0.0, 0.0])
        h = self.vhist.get(m["id"])
        if not h or len(h) < 3 or h[-1][0] - h[0][0] < 0.3:
            return v
        dt = h[-1][0] - h[0][0]
        ax, ay = (h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt
        k = max(0.0, min(1.0, 1.0 - flight / P["vel_blend_t"]))
        return [v[0] * k + ax * (1 - k), v[1] * k + ay * (1 - k), v[2]]

    def center_of(self, m):
        scale = 1 + 0.05 * m.get("cubes", 0) if m["kind"] == "mech" else 1
        h = 4.5 * scale if m["kind"] == "mech" else 1.75
        return [m["pos"][0], m["pos"][1], m["pos"][2] + h]

    # ------------------------------------------------------------------ navigation
    def roof_height(self, q):
        """Surface height at q if it is a roof (prism top), else None."""
        c = self.W.cell(q[0], q[1])
        return self.W.surf[c] if self.W.isprism[c] else None

    def climb_target(self, me, goal):
        """True when goal is on a roof we are below and could jet up to."""
        rh = self.roof_height(goal)
        if rh is None:
            return False
        rise = rh - me["pos"][2]
        lo = 2.0 if me["grounded"] else -0.6   # mid-climb: keep going until we clear the edge
        return lo < rise <= P["climb_max"]

    def find_base(self, goal):
        """Nearest ground cell at the foot of the building holding goal."""
        W = self.W
        gc = W.cell(goal[0], goal[1])
        if gc in self.climb_base:
            return self.climb_base[gc]
        roof = W.surf[gc]
        i0, j0 = gc % W.nx, gc // W.nx
        best = None
        for r in range(1, 25):
            for dj in range(-r, r + 1):
                for di in ((-r, r) if abs(dj) != r else range(-r, r + 1)):
                    i, j = i0 + di, j0 + dj
                    if not (0 < i < W.nx - 1 and 0 < j < W.ny - 1):
                        continue
                    c = j * W.nx + i
                    if W.walk[c] and not W.isprism[c] and W.surf[c] < roof - 2:
                        best = W.center(c)
                        break
                if best:
                    break
            if best:
                break
        self.climb_base[gc] = best
        return best

    def nav(self, me, goal, t, urgent=False):
        """Move direction toward goal; climbs onto roofs with jump + jets when the goal is up there."""
        p = me["pos"]
        rh = self.roof_height(goal)
        if rh is not None and p[2] >= rh - 0.6 and hyp(p, goal) < 60 and (me["grounded"] or p[2] >= rh + 0.3):
            # at (or above) roof level: just walk/drift onto it, keep a little lift while airborne
            ux, uy = norm2(goal[0] - p[0], goal[1] - p[1])
            if not me["grounded"] and p[2] < rh + 0.5:
                self.nav_boost = True
            self.mode_note = "climb"
            return ux, uy
        if self.climb_target(me, goal):
            base = self.find_base(goal)
            gc = self.W.cell(goal[0], goal[1])
            if base is not None and self.climb_fail.get(gc, 0) < 2:
                if hyp(p, base) > (3.5 if me["grounded"] else 9.0):
                    return self.nav_ground(me, base, t, urgent)
                rise = rh - p[2]
                need = (rise / 10.0 + 1.2) * self.cfg["mech"].get("jet_drain", 15) + 10
                ux, uy = norm2(goal[0] - p[0], goal[1] - p[1])
                if me["grounded"]:
                    if me["energy"] < need:
                        return 0.0, 0.0          # wait at the wall for energy
                    self.nav_jump = True
                elif me["energy"] < 2 and p[2] < rh - 0.5:
                    self.climb_fail[gc] = self.climb_fail.get(gc, 0) + 1
                    if self.climb_fail[gc] >= 2:
                        self.unreachable.append((t, goal[0], goal[1]))
                self.nav_boost = p[2] < rh + 1.0
                self.mode_note = "climb"
                return ux, uy
        return self.nav_ground(me, goal, t, urgent)

    def nav_ground(self, me, goal, t, urgent=False):
        """Move direction (unit 2D) toward goal following an A* path."""
        p = me["pos"]
        if hyp(p, goal) < 2.0:
            return 0.0, 0.0
        need = (self.path is None or self.path_goal is None or hyp(self.path_goal, goal) > 5.0
                or t - self.path_t > (1.0 if urgent else 2.0))
        if need:
            if self.W.walk_line(p, goal) and hyp(p, goal) < 60:
                self.path = [tuple(goal[:2])]
            else:
                path = self.W.astar(p[:2], goal[:2], budget_s=P["astar_budget"])
                partial = bool(path) and hyp(path[-1], goal) > 6.0
                if partial and self.partial_goal is not None and hyp(self.partial_goal, goal) < 3.0:
                    self.unreachable.append((t, goal[0], goal[1]))
                self.partial_goal = list(goal[:2]) if partial else None
                self.path = path or [tuple(goal[:2])]
            self.path_goal = list(goal[:2])
            self.path_t = t
            self.path_i = 0
        path = self.path
        # advance along path
        while self.path_i < len(path) - 1 and hyp(p, path[self.path_i]) < 3.0:
            self.path_i += 1
        # look ahead to the farthest directly walkable waypoint
        best = self.path_i
        for k in range(self.path_i + 1, min(len(path), self.path_i + 10)):
            if self.W.walk_line(p, path[k]):
                best = k
            else:
                break
        self.path_i = best
        wp = path[self.path_i]
        return norm2(wp[0] - p[0], wp[1] - p[1])

    def pick_dir(self, me, want, zone, t, probe=7.0):
        """Choose a walkable direction close to `want` (unit 2D)."""
        p = me["pos"]
        if want == (0.0, 0.0):
            return want
        base = math.atan2(want[1], want[0])
        for off in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.3, -2.3, math.pi):
            a = base + off
            q = (p[0] + math.cos(a) * probe, p[1] + math.sin(a) * probe)
            if not self.W.in_bounds(q[0], q[1], 6):
                continue
            if not self.W.walk_line(p, q):
                continue
            if zone is not None and not self.safe_inside(q, zone, t, margin=3.0, horizon=2.0) \
                    and self.safe_inside(p, zone, t, margin=3.0, horizon=2.0):
                continue
            return math.cos(a), math.sin(a)
        return want

    # ------------------------------------------------------------------ aiming
    def lead(self, src, tgt_pos, tgt_vel, speed):
        tx, ty, tz = tgt_pos
        tt = P["latency"]
        for _ in range(3):
            px = tgt_pos[0] + tgt_vel[0] * tt
            py = tgt_pos[1] + tgt_vel[1] * tt
            pz = tgt_pos[2] + tgt_vel[2] * tt * 0.5
            d = math.sqrt((px - src[0]) ** 2 + (py - src[1]) ** 2 + (pz - src[2]) ** 2)
            tt = d / speed + P["latency"]
            tx, ty, tz = px, py, pz
        return [tx, ty, tz]

    @staticmethod
    def aim_to(src, dst):
        dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def mortar_aim(self, src, dst, speed, g):
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        d = math.hypot(dx, dy)
        dz = dst[2] - src[2]
        v2 = speed * speed
        disc = v2 * v2 - g * (g * d * d + 2 * dz * v2)
        if disc < 0 or d < 1:
            return None
        sq = math.sqrt(disc)
        hi = math.atan2(v2 + sq, g * d)
        lo = math.atan2(v2 - sq, g * d)
        return math.atan2(dy, dx), lo, hi

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        self.nav_jump = False
        self.nav_boost = False
        self.mode_note = ""
        t = obs["t"]
        me = obs["self"]
        p = me["pos"]
        z = obs["zone"]
        act = {"tick": obs["tick"]}
        hpf = me["hp"] / max(1.0, me["max_hp"])
        stats = me.get("stats", {})
        muzzle_h = stats.get("muzzle_height", 7)
        eye = [p[0], p[1], p[2] + stats.get("eye_height", 8)]
        muzzle = [p[0], p[1], p[2] + muzzle_h]
        my_c = [p[0], p[1], p[2] + 4.5]
        regen_delay = stats.get("regen_delay", 5)

        # ---------------- events
        for e in obs["events"]:
            ty = e["type"]
            if ty == "damaged":
                if e.get("by") is not None:
                    self.last_hit_t = t
                    self.last_dmg_bearing = e.get("bearing")
                    self.last_dmg_by = e.get("by")
                    self.threat_ids[e["by"]] = t
            elif ty == "hit":
                self.last_deal_t = t
            elif ty == "kill":
                self.my_kills += 1
            elif ty == "take_failed":
                self.blocked_pickups[e["id"]] = t
                if e.get("reason") == "gone":
                    self.mem_pk.pop(e["id"], None)
            elif ty == "took":
                self.mem_pk.pop(e["id"], None)
        # ---------------- upgrades
        if obs.get("offer"):
            ch = obs["offer"]["choices"]
            best, bi = 999, 0
            for i, c in enumerate(ch):
                r = UPGRADE_PRIORITY.index(c["id"]) if c["id"] in UPGRADE_PRIORITY else 50
                if r < best:
                    best, bi = r, i
            act["pick"] = bi

        weapons = me["weapons"]
        kinds = {w["kind"]: i for i, w in enumerate(weapons) if w}
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for m in enemies + npcs:
            if m["kind"] == "mech":
                self.last_seen[m["id"]] = (t, m["pos"], m["vel"], m.get("hp", 1400))
            h = self.vhist.setdefault(m["id"], [])
            if not h or h[-1][0] < t:
                h.append((t, m["pos"][0], m["pos"][1]))
            while len(h) > 2 and t - h[0][0] > P["vel_window"]:
                h.pop(0)
        for k in [k for k, v in self.last_seen.items() if t - v[0] > 12]:
            del self.last_seen[k]
        self.update_ffa(obs, enemies, npcs, me, t)
        self.update_item_memory(obs, me, t)
        for n in npcs:
            self.mem_npc[n["id"]] = {"id": n["id"], "pos": n["pos"], "hp": n.get("hp", 350), "t": t}
        for e in obs["events"]:
            if e["type"] == "hit" and 100 <= e.get("target", 0) < 1000 and e["target"] in self.mem_npc:
                self.mem_npc[e["target"]]["hp"] -= e.get("amount", 0)
        for k in [k for k, v in self.mem_npc.items() if t - v["t"] > 25 or v["hp"] <= 0]:
            del self.mem_npc[k]
        for snd in obs["sounds"]:
            if snd["kind"] in ("gun", "laser", "lightning", "mortar", "laser_charge", "step", "boost", "jet",
                               "crate", "npc_gun", "reload", "explosion"):
                est = {"near": 35.0, "mid": 110.0, "far": 230.0}.get(snd["dist"], 110.0)
                b = snd["bearing"]
                self.sound_leads.append((t - snd.get("age", 0), p[0] + math.cos(b) * est, p[1] + math.sin(b) * est))
        self.sound_leads = [l for l in self.sound_leads if t - l[0] < P["lead_mem"]][-40:]
        for s in obs["sounds"]:
            if s["kind"] in ("gun", "laser", "lightning", "mortar", "laser_charge", "step", "boost", "jet") \
                    and s["dist"] in ("near", "mid"):
                if self.sound_focus is None or t - self.sound_focus[0] > 1.0 or s["dist"] == "near":
                    self.sound_focus = (t, s["bearing"])

        # ---------------- target selection
        def score_enemy(m):
            d = hyp(p, m["pos"])
            hp = m.get("hp", 1400)
            s = d * P["dist_w"] + hp * P["hp_w"]
            if m["id"] in self.threat_ids and t - self.threat_ids[m["id"]] < 3:
                s -= 40
            if m["id"] == self.target_id:
                s -= 25
            return s
        vis_enemies = sorted(enemies, key=score_enemy)
        target = None
        for m in vis_enemies[:4]:
            if self.W.los(muzzle, self.center_of(m)):
                target = m
                break
        if target is None and vis_enemies:
            target = vis_enemies[0]
        self.target_id = target["id"] if target else None

        near_enemies = [m for m in enemies if hyp(p, m["pos"]) < 160]
        under_fire = t - self.last_hit_t < 2.5
        recently_hit = t - self.last_hit_t < regen_delay

        # ---------------- retreat hysteresis
        n_threats = len(near_enemies)
        rh, rhm = (P["ffa_retreat_hp"], P["ffa_retreat_hp_multi"]) if self.ffa else (P["retreat_hp"], P["retreat_hp_multi"])
        if hpf < rh or (hpf < rhm and n_threats >= 2):
            self.retreating = True
        if P["retreat_threat"] and target is not None:
            tf = target.get("hp", 1400) / max(1.0, target.get("max_hp", 1400))
            finishing = tf < 0.2 and hpf > 0.2 and n_threats <= 2
            if not finishing and ((hpf < 0.35 and tf > hpf + 0.1) or (n_threats >= 2 and hpf < 0.55)
                                  or (n_threats >= 3 and hpf < 0.8)):
                self.retreating = True
        if hpf > P["recover_hp"] or (not enemies and not recently_hit and hpf > 0.7):
            self.retreating = False

        # ---------------- zone
        zc, zr = self.zone_now(z, t)
        in_zone = hyp(p, zc) < zr - 1
        dist_home = 0
        home = self.zone_home(p, z, t)
        speed = stats.get("walk_speed", 11.7)
        dist_home = hyp(p, home)
        # time until current circle edge passes us if we stay
        must_move = False
        if not self.safe_inside(p, z, t, margin=6.0, horizon=dist_home / speed * 1.4 + 6.0):
            must_move = True
        # once next circle is known and shrink is approaching, move inside early
        if t > z["shrink_start"] - 25 and hyp(p, z["next_center"]) > z["next_radius"] - 8:
            must_move = True

        goal = None
        mode = "hold"
        boost = False
        move = (0.0, 0.0)

        if not in_zone or must_move:
            goal = home
            mode = "zone"
            boost = (not in_zone) or (dist_home / speed > max(3.0, z["shrink_end"] - t))
        if goal is None and self.retreating and (enemies or recently_hit):
            mode = "retreat"
        if goal is None and mode != "retreat" and target is not None:
            mode = "fight"
            if self.ffa:
                attacking = target["id"] in self.threat_ids and t - self.threat_ids[target["id"]] < 6
                dt_ = hyp(p, target["pos"])
                weak = target.get("hp", 1400) < P["ffa_prey_hp"] and dt_ < 120
                if not attacking and not weak and dt_ > P["ffa_engage_r"]:
                    mode = "avoid"

        # ---------------- loot planning (only when nothing better)
        loot_goal = None
        take_id = None
        drop_slot = None
        # value of the slot a pickup would replace: an empty slot is free, else our worst weapon
        weakest = 0.0 if None in weapons else min(WEAPON_VALUE[w["kind"]] for w in weapons if w)
        for pk in list(self.mem_pk.values()):
            if pk["kind"] in kinds or (pk["id"] in self.blocked_pickups and self.blocked_pickups[pk["id"]] > t - 5):
                continue
            v = WEAPON_VALUE.get(pk["kind"], 0)
            if v <= weakest:
                continue
            d = hyp(p, pk["pos"])
            if d < self.cfg["loot"]["pickup_radius"] - 0.1 and abs(pk["pos"][2] - p[2]) < 3.9:
                charging_laser = "laser" in kinds and (weapons[kinds["laser"]].get("charge", 0) > 0)
                if not charging_laser:
                    take_id = pk["id"]
                    if None not in weapons:
                        # replace the lowest-value weapon
                        drop_slot = min((i for i, w in enumerate(weapons) if w),
                                        key=lambda i: WEAPON_VALUE[weapons[i]["kind"]])
            on_roof = pk["pos"][2] - self.W.ground(pk["pos"][0], pk["pos"][1]) > 3 and abs(pk["pos"][2] - p[2]) > 2.5 \
                and not (self.climb_target(me, pk["pos"]) and not enemies)
            if on_roof or self.is_unreachable(pk["pos"], t) or not self.safe_inside(pk["pos"], z, t, margin=5, horizon=20):
                continue
            score = d - v * 25
            if self.loot_commit == pk["id"]:
                score -= 30  # commitment: don't dither between items
            if loot_goal is None or score < loot_goal[0]:
                loot_goal = (score, pk["pos"], "weapon", pk["id"])
        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot
        for cb in list(self.mem_cube.values()):
            if self.blocked_pickups.get(cb["id"], -1) > t or self.is_unreachable(cb["pos"], t):
                continue
            d = hyp(p, cb["pos"])
            if cb["pos"][2] - self.W.ground(cb["pos"][0], cb["pos"][1]) > 3 and abs(cb["pos"][2] - p[2]) > 2.5 \
                    and not (self.climb_target(me, cb["pos"]) and not enemies):
                continue
            if not self.safe_inside(cb["pos"], z, t, margin=5, horizon=15):
                continue
            score = d - 40 - (30 if self.loot_commit == cb["id"] else 0)
            if d < 160 and (loot_goal is None or score < loot_goal[0]):
                loot_goal = (score, cb["pos"], "cube", cb["id"])
        # supply drops: full heal + cubes
        drop_goal = None
        for dr in obs.get("drops", []):
            d = hyp(p, dr["pos"])
            contested = any(hyp(m["pos"], dr["pos"]) < 50 for m in enemies)
            if d < 220 and dr["lands_in"] < d / speed + 8 and not contested \
                    and (dr["holder"] is None or dr["holder"] == me["id"]) \
                    and self.safe_inside(dr["pos"], z, t, margin=5, horizon=d / speed + 6):
                drop_goal = dr
                break

        # laser/lightning first: a power weapon decides duels (house bots grab lasers early)
        arm_goal = None
        if P["arm_r"] > 0 and "laser" not in kinds and "lightning" not in kinds:
            best = None
            for pk in self.mem_pk.values():
                if pk["kind"] not in ("laser", "lightning"):
                    continue
                if self.blocked_pickups.get(pk["id"], -1) > t - 5 or self.is_unreachable(pk["pos"], t):
                    continue
                if pk["pos"][2] - self.W.ground(pk["pos"][0], pk["pos"][1]) > 3 and abs(pk["pos"][2] - p[2]) > 2.5:
                    continue
                d = hyp(p, pk["pos"])
                if d < P["arm_r"] and self.safe_inside(pk["pos"], z, t, margin=5, horizon=d / speed + 5):
                    if best is None or d < best[0]:
                        best = (d, pk)
            if best is not None:
                arm_goal = best[1]

        # shootables when idle
        # farming: crates (1 cube, maybe a weapon) and NPC drones (2 cubes) from memory, actively
        shoot_obj = None
        farm_goal = None
        if target is None and t - self.last_hit_t > 1.5:
            cands = []
            if hpf > P["farm_crate_hp"]:
                for c in list(self.mem_crate.values()):
                    d = hyp(p, c["pos"])
                    if d < P["farm_crate_r"] and c["pos"][2] - self.W.ground(c["pos"][0], c["pos"][1]) < 3 \
                            and not self.is_unreachable(c["pos"], t) \
                            and self.safe_inside(c["pos"], z, t, margin=5, horizon=d / speed + 8):
                        cands.append((d + 5, c, "crate"))
            if hpf > P["farm_npc_hp"]:
                for n in list(self.mem_npc.values()):
                    d = hyp(p, n["pos"])
                    if d < P["farm_npc_r"] and self.safe_inside(n["pos"], z, t, margin=5, horizon=d / speed + 8):
                        cands.append((d - 20, n, "npc"))
            cands.sort(key=lambda c: c[0])
            for sc, o, k in cands[:4]:
                cpt = [o["pos"][0], o["pos"][1], o["pos"][2] + (1.75 if k == "npc" else 1.2)]
                if hyp(p, o["pos"]) < 120 and self.W.los(muzzle, cpt):
                    shoot_obj = (o, k, cpt)
                    break
            if cands:
                farm_goal = cands[0]

        if arm_goal is not None and mode in ("fight", "hold", "avoid"):
            close = target is not None and (hyp(p, target["pos"]) < P["arm_fight_r"] or
                                            (target["id"] in self.threat_ids and t - self.threat_ids[target["id"]] < 3
                                             and hyp(p, target["pos"]) < P["arm_fight_r2"]))
            if not close:
                mode = "arm"

        # ---------------- movement decision
        aim_point = None
        aim_vel = [0.0, 0.0, 0.0]
        if mode == "zone":
            move = self.nav(me, goal, t, urgent=True)
            if target is not None:
                aim_point = self.center_of(target)
                aim_vel = self.smooth_vel(target, hyp(p, target["pos"]) / 252.0)
        elif mode == "retreat":
            move, boost = self.retreat_move(me, enemies, z, t, hpf)
            if target is not None:
                aim_point = self.center_of(target)
                aim_vel = self.smooth_vel(target, hyp(p, target["pos"]) / 252.0)
        elif mode == "arm":
            move = self.nav(me, arm_goal["pos"], t, urgent=True)
            self.loot_commit = arm_goal["id"]
            if target is not None:      # keep shooting while we walk
                aim_point = self.center_of(target)
                aim_vel = self.smooth_vel(target, hyp(p, target["pos"]) / 252.0)
        elif mode == "avoid":
            # free-for-all: stay out of a healthy stranger's fight; keep watching it
            move, boost = self.retreat_move(me, enemies, z, t, hpf)
            boost = False
            aim_point = self.center_of(target)
            aim_vel = self.smooth_vel(target, hyp(p, target["pos"]) / 252.0)
        elif mode == "fight":
            move, boost = self.fight_move(me, target, kinds, weapons, z, t, hpf)
            # grab nearby cubes mid-fight (+HP, +damage) while still shooting
            near_cubes = [c for c in self.mem_cube.values() if hyp(p, c["pos"]) < P["cube_fight_r"]
                          and abs(c["pos"][2] - p[2]) < 3.5]
            if near_cubes and hyp(p, target["pos"]) > 15:
                cb = min(near_cubes, key=lambda c: hyp(p, c["pos"]))
                move = norm2(cb["pos"][0] - p[0], cb["pos"][1] - p[1])
                mode = "fight:cube"
            elif loot_goal is not None and loot_goal[2] == "weapon" and hyp(p, loot_goal[1]) < P["weapon_fight_r"] \
                    and hyp(p, target["pos"]) > 20:
                # a better weapon close by is worth a detour even under fire
                move = self.nav(me, loot_goal[1], t, urgent=True)
                mode = "fight:weapon"
            aim_point = self.center_of(target)
            aim_vel = self.smooth_vel(target, hyp(p, target["pos"]) / 252.0)
        else:
            if drop_goal is not None and not under_fire:
                mode = "drop"
                if hyp(p, drop_goal["pos"]) > drop_goal["radius"] * 0.5:
                    move = self.nav(me, drop_goal["pos"], t)
                    boost = me["energy"] > 50 and hyp(p, drop_goal["pos"]) > 40
            elif loot_goal is not None and not under_fire:
                mode = "loot:" + loot_goal[2]
                if self.loot_commit != loot_goal[3]:
                    self.loot_commit = loot_goal[3]
                    self.loot_since = t
                elif t - self.loot_since > P["loot_timeout"]:
                    # can't reach it (ledge, roof, knoll): give up on it for a while
                    self.blocked_pickups[loot_goal[3]] = t + 60
                    self.loot_commit = None
                move = self.nav(me, loot_goal[1], t)
            elif shoot_obj is not None or farm_goal is not None:
                o, k = (shoot_obj[0], shoot_obj[1]) if shoot_obj is not None else (farm_goal[1], farm_goal[2])
                mode = "farm:" + k
                d = hyp(p, o["pos"])
                stand = 30.0 if k == "crate" else 38.0
                if shoot_obj is None or d > stand:
                    move = self.nav(me, o["pos"], t)
                elif d < 12 and k == "npc":
                    move = norm2(p[0] - o["pos"][0], p[1] - o["pos"][1])
            elif under_fire and target is None:
                mode = "hunted"
                # hit by something unseen: get moving, break line, turn toward it
                move, boost = self.retreat_move(me, [], z, t, hpf)
            elif (prey := self.pick_prey(p, t, z, hpf)) is not None:
                # a remembered enemy: go finish it (kills are what score)
                mode = "hunt"
                move = self.nav(me, prey, t)
            elif hpf < P["heal_still_hp"] and t - self.last_hit_t < 30 and self.safe_inside(p, z, t, 10, 15):
                mode = "heal"  # stand still: silent, regenerating
            elif not self.ffa and hpf >= P["seek_hp"] and (lead := self.pick_lead(p, z, t)) is not None:
                mode = "seek"  # go toward what we heard: house bots only count if we kill them
                move = self.nav(me, lead, t)
            else:
                mode = "hold"
                ex = self.explore_goal(p, z, t)
                if ex is not None:
                    mode = "explore"
                    move = self.nav(me, ex, t)
                elif dist_home > 8:
                    move = self.nav(me, home, t)
                    move = (move[0] * 0.8, move[1] * 0.8)
            if shoot_obj is not None and aim_point is None:
                aim_point = shoot_obj[2]
                aim_vel = shoot_obj[0].get("vel", [0, 0, 0])

        # ---------------- dodge projectiles / shells / lasers
        dodge = self.dodge(obs, me, enemies)
        if dodge is not None:
            dx, dy, urg = dodge
            mx, my = move
            move = norm2(mx * (1 - urg) + dx * urg * 2, my * (1 - urg) + dy * urg * 2)
            if urg > 0.6 and me["energy"] > 25:
                boost = True

        # ---------------- walkability / stuck handling
        if self.mode_note == "climb":
            mode = mode + "+climb"   # straight at the wall: no detours, no unstick
            self.hist.clear()
        else:
            if move != (0.0, 0.0):
                move = self.pick_dir(me, move, z, t)
            move = self.unstick(me, move, t, act)

        # ---------------- aiming & firing
        fire = [False, False]
        if aim_point is not None:
            aim, fire = self.weapons_logic(me, weapons, kinds, aim_point, aim_vel, target, muzzle, t, obs)
            if mode == "avoid":
                fire = [False, False]  # don't reveal ourselves to a fight we're not in
            act["aim"] = aim
            if TRACE and target is not None:
                d_ = hyp(p, target["pos"])
                mf.log(f"T t={t:.1f} mode={mode} tgt={target['id']} d={d_:.0f} hp={target.get('hp', 0):.0f} "
                       f"aimerr={abs(wrap(me['aim'][0] - aim[0])):.3f} fire={fire} w={list(kinds)} "
                       f"ammo={weapons[0].get('ammo') if weapons[0] else None} "
                       f"rl={weapons[0].get('reload') if weapons[0] else None} los={self.W.los(muzzle, aim_point)}")
        else:
            act["aim"] = self.scan_aim(me, t, move)
            # keep a charged laser from wasting; release does nothing harmful
            if "laser" in kinds and weapons[kinds["laser"]].get("charge", 0) > 0:
                fire[kinds["laser"]] = False
        if "gun" in kinds:
            g = weapons[kinds["gun"]]
            if g.get("reload", 0) <= 0 and g.get("ammo", 1) < g["stats"]["mag"] * 0.5 and target is None \
                    and shoot_obj is None:
                act["reload"] = True
        act["fire"] = fire
        if "laser" in kinds and weapons[kinds["laser"]].get("charge", 0) > 0:
            move = move  # half speed while charging: nothing to do
        if boost and me["energy"] < 12:
            boost = False
        if self.nav_jump:
            act["jump"] = True
        if self.nav_boost:
            boost = True
        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost and (move != (0.0, 0.0) or self.nav_boost))
        self.mode = mode
        mf.draw.label(f"{mode} hp{me['hp']:.0f} e{me['energy']:.0f} {(time.perf_counter() - t0) * 1000:.0f}ms")
        return act

    # ------------------------------------------------------------------ item memory
    def is_unreachable(self, q, t):
        for (ts, x, y) in self.unreachable:
            if t - ts < 90 and math.hypot(q[0] - x, q[1] - y) < 4.0:
                return True
        return False

    def update_item_memory(self, obs, me, t):
        p = me["pos"]
        eye = [p[0], p[1], p[2] + me.get("stats", {}).get("eye_height", 8)]
        yaw = me["aim"][0]
        half_fov = math.radians(self.cfg["vision"]["fov_deg"]) / 2
        for mem, key in ((self.mem_pk, "pickups"), (self.mem_cube, "cubes"), (self.mem_crate, "crates")):
            now = {it["id"]: it for it in obs.get(key, [])}
            mem.update(now)
            checks = 0
            for iid in list(mem):
                if iid in now:
                    continue
                q = mem[iid]["pos"]
                d = hyp(p, q)
                visible_spot = d < 40 or (d < 150 and abs(wrap(math.atan2(q[1] - p[1], q[0] - p[0]) - yaw)) < half_fov)
                if not visible_spot:
                    continue
                if d < 40:
                    del mem[iid]
                    continue
                if checks < 6:
                    checks += 1
                    if self.W.los(eye, [q[0], q[1], q[2] + 1.0], step=2.5):
                        del mem[iid]

    # ------------------------------------------------------------------ match type
    def update_ffa(self, obs, enemies, npcs, me, t):
        """Practice house bots never fight each other, so they only die to us (or, late, to the zone).
        Mechs dying that we did not kill while the zone is still gentle means a free-for-all."""
        if self.ffa:
            return
        others_dead = (self.n_mechs - obs["alive"]) - self.my_kills
        if others_dead >= P["ffa_deaths"] and obs["zone"].get("dps", 0) <= 35:
            self.ffa = True
            mf.log(f"t={t:.1f} free-for-all detected ({others_dead} deaths we did not cause)")
            return
        # a laser beam whose first contact is another mech, fired by a mech aiming at it
        for b in obs.get("beams", []):
            by = b.get("by")
            path = b.get("path") or []
            if b.get("kind") != "laser" or by is None or by == me["id"] or by >= 16 or len(path) < 2:
                continue
            first = path[1]
            if math.dist(first, [me["pos"][0], me["pos"][1], me["pos"][2] + 4.5]) < 10:
                continue
            for m in enemies:
                if m["id"] != by and math.dist(first, self.center_of(m)) < 4:
                    self.ffa_evidence += 1
                    self.ffa_victims.add(m["id"])
        if self.ffa_evidence >= P["ffa_evidence"] and len(self.ffa_victims) >= 2:
            self.ffa = True
            mf.log(f"t={t:.1f} free-for-all detected (laser fights)")

    # ------------------------------------------------------------------ behaviors
    def pick_prey(self, p, t, z, hpf):
        """Last known position of a recently seen enemy worth chasing, or None."""
        if hpf < P["hunt_hp"]:
            return None
        best, bs = None, 1e9
        for mid, (ts, pos, vel, hp) in self.last_seen.items():
            age = t - ts
            if age > P["hunt_mem"]:
                continue
            q = [pos[0] + vel[0] * min(age, 2.0), pos[1] + vel[1] * min(age, 2.0)]
            d = hyp(p, q)
            if d > P["hunt_r"] or d < 4:
                continue
            if not self.safe_inside(q, z, t, margin=8, horizon=d / 11.7 + 5):
                continue
            if self.ffa and hp > P["ffa_prey_hp"]:
                continue
            s = d + hp * 0.1 + age * 15
            if s < bs:
                bs, best = s, q
        return best

    def pick_lead(self, p, z, t):
        best, bs = None, 1e9
        for (ts, x, y) in self.sound_leads:
            d = math.hypot(x - p[0], y - p[1])
            if d < 15 or not self.W.in_bounds(x, y, 10):
                continue
            if not self.safe_inside((x, y), z, t, margin=10, horizon=d / 11.7 + 5):
                continue
            s = (t - ts) * 20 + d * 0.5
            if s < bs:
                bs, best = s, (x, y)
        return best

    def explore_goal(self, p, z, t):
        """Wander inside the next circle to find crates, drones and loot."""
        if t > (P["ffa_explore_until"] if self.ffa else P["explore_until"]):
            return None
        if self.explore is not None and hyp(p, self.explore) > 6 and t < self.explore_until:
            return self.explore
        nc, nr = z["next_center"], z["next_radius"]
        zc, zr = self.zone_now(z, t)
        for _ in range(16):
            a = self.rng.random() * TAU
            r = 40 + self.rng.random() * 110
            q = (p[0] + math.cos(a) * r, p[1] + math.sin(a) * r)
            if not self.W.in_bounds(q[0], q[1], 20) or not self.W.walk[self.W.cell(*q)] \
                    or self.W.isprism[self.W.cell(*q)]:
                continue
            if hyp(q, zc) > zr - 25 or (nr > 30 and hyp(q, nc) > nr * 0.9):
                continue
            self.explore = q
            self.explore_until = t + 25
            return q
        return None

    def fight_move(self, me, target, kinds, weapons, z, t, hpf):
        p = me["pos"]
        tp = target["pos"]
        d = hyp(p, tp)
        if "lightning" in kinds and me["energy"] > 20 and hpf > 0.5:
            pref = P["pref_lightning"]
        elif "laser" in kinds:
            pref = P["pref_laser"]
        elif "mortar" in kinds and "gun" not in kinds:
            pref = 120.0
        else:
            pref = P["pref_gun"]
        if t > self.strafe_t:
            self.strafe = -self.strafe
            self.strafe_t = t + P["strafe_min"] + self.rng.random() * P["strafe_span"]
        ux, uy = norm2(tp[0] - p[0], tp[1] - p[1])
        lx, ly = -uy * self.strafe, ux * self.strafe
        radial = max(-1.0, min(1.0, (d - pref) / 25.0))
        mx = lx * 0.85 + ux * radial
        my = ly * 0.85 + uy * radial
        boost = False
        if d < pref * 0.5 and me["energy"] > 40 and pref > 50:
            boost = True
        return norm2(mx, my), boost

    def retreat_move(self, me, enemies, z, t, hpf):
        """Move to a spot that breaks line of sight to the known enemies."""
        p = me["pos"]
        threats = [m["pos"] for m in enemies]
        for mid, (ts, pos, vel, hp) in self.last_seen.items():
            if t - ts < 6 and all(hyp(pos, q) > 3 for q in threats):
                threats.append(pos)
        if not threats and self.last_dmg_bearing is not None and t - self.last_hit_t < 5:
            b = self.last_dmg_bearing
            threats.append([p[0] + math.cos(b) * 80, p[1] + math.sin(b) * 80, p[2]])
        if self.cover is not None and t - self.cover_t < 1.2 and hyp(p, self.cover) > 2.5:
            goal = self.cover
        else:
            goal = self.find_cover(me, threats, z, t)
            self.cover = goal
            self.cover_t = t
        if goal is None:
            # run directly away from the threat centroid
            if threats:
                cx = sum(q[0] for q in threats) / len(threats)
                cy = sum(q[1] for q in threats) / len(threats)
                return norm2(p[0] - cx, p[1] - cy), me["energy"] > 30
            return (0.0, 0.0), False
        if hyp(p, goal) < 2.5:
            return (0.0, 0.0), False
        return self.nav(me, goal, t, urgent=True), me["energy"] > 30

    def find_cover(self, me, threats, z, t):
        p = me["pos"]
        W = self.W
        if not threats:
            return None
        teyes = [[q[0], q[1], q[2] + 8] for q in threats[:5]]
        best, bs = None, 1e9
        for r in (12.0, 24.0, 40.0):
            for k in range(12):
                a = k * TAU / 12 + (r * 0.1)
                q = (p[0] + math.cos(a) * r, p[1] + math.sin(a) * r)
                if not W.in_bounds(q[0], q[1], 8):
                    continue
                c = W.cell(*q)
                if not W.walk[c]:
                    continue
                if not self.safe_inside(q, z, t, margin=4, horizon=r / 11 + 5):
                    continue
                qz = W.surf[c]
                qc = [q[0], q[1], qz + 4.5]
                seen = 0
                for e in teyes:
                    if hyp(e, q) < 45 or W.los(e, qc, step=2.5):
                        seen += 1
                dist_sum = sum(min(1.0, hyp(e, q) / 150) for e in teyes)
                s = seen * 100 + r * 0.6 - dist_sum * 10
                if s < bs:
                    bs, best = s, [q[0], q[1]]
            if best is not None and bs < 50:
                break
        return best

    def dodge(self, obs, me, enemies):
        p = me["pos"]
        c = [p[0], p[1], p[2] + 4.5]
        best = None
        g = self.wcfg["mortar"]["gravity"]
        for pr in obs["projectiles"]:
            v = pr["vel"]
            q = pr["pos"]
            if pr["kind"] == "mortar":
                # predict landing
                x, y, zz = q
                vx, vy, vz = v
                dt = 0.05
                land = None
                for k in range(160):
                    x += vx * dt
                    y += vy * dt
                    vz -= g * dt
                    zz += vz * dt
                    if zz <= self.W.surface(x, y):
                        land = (x, y, k * dt)
                        break
                if land is None:
                    continue
                rad = self.wcfg["mortar"]["splash_radius"] + 4
                ex = p[0] + me["vel"][0] * min(land[2], 1.0)
                ey = p[1] + me["vel"][1] * min(land[2], 1.0)
                dd = math.hypot(ex - land[0], ey - land[1])
                if dd < rad and land[2] < 3.0:
                    dx, dy = norm2(p[0] - land[0], p[1] - land[1])
                    if dx == 0 and dy == 0:
                        dx, dy = 1.0, 0.0
                    urg = 1.0 - dd / rad * 0.5
                    if best is None or urg > best[2]:
                        best = (dx, dy, urg)
                continue
            rx, ry, rz = c[0] - q[0], c[1] - q[1], c[2] - q[2]
            vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
            if vv < 1:
                continue
            ts = (rx * v[0] + ry * v[1] + rz * v[2]) / vv
            if ts <= 0 or ts > 0.7:
                continue
            mx, my = rx - v[0] * ts, ry - v[1] * ts
            miss = math.hypot(mx, my)
            if miss < 4.0:
                dx, dy = norm2(mx, my)
                if dx == 0 and dy == 0:
                    dx, dy = norm2(-v[1], v[0])
                urg = 0.5 * (1 - miss / 4.0) + 0.2
                if best is None or urg > best[2]:
                    best = (dx, dy, urg)
        # laser telegraph: an enemy charging with aim on us -> sidestep hard
        for m in enemies:
            if m.get("charging"):
                ya = math.atan2(p[1] - m["pos"][1], p[0] - m["pos"][0])
                if abs(wrap(m["aim"][0] - ya)) < 0.15:
                    dx, dy = norm2(-(p[1] - m["pos"][1]), p[0] - m["pos"][0])
                    s = self.strafe
                    best = (dx * s, dy * s, 0.7)
        return best

    def unstick(self, me, move, t, act):
        p = me["pos"]
        self.hist.append((t, p[0], p[1]))
        while self.hist and t - self.hist[0][0] > 1.0:
            self.hist.pop(0)
        if t < self.unstick_until and self.unstick_dir is not None:
            act["jump"] = me["grounded"]
            if me["energy"] > 15:
                act["boost"] = True
            return self.unstick_dir
        wanting = move != (0.0, 0.0)
        if wanting and len(self.hist) >= 8:
            t0, x0, y0 = self.hist[0]
            moved = math.hypot(p[0] - x0, p[1] - y0)
            if moved < 2.0 and t - t0 > 0.8:
                # stuck against something: jump-jet in the wanted direction, then try a side
                a = math.atan2(move[1], move[0]) + self.rng.choice((0.0, 0.8, -0.8))
                self.unstick_dir = (math.cos(a), math.sin(a))
                self.unstick_until = t + 0.8
                self.path = None
                self.hist.clear()
                act["jump"] = me["grounded"]
                return self.unstick_dir
        return move

    def scan_aim(self, me, t, move):
        """No target: look where danger likely is."""
        if self.last_dmg_bearing is not None and t - self.last_hit_t < 4:
            return [self.last_dmg_bearing, 0.0]
        if self.sound_focus is not None and t - self.sound_focus[0] < 2.5:
            return [self.sound_focus[1], 0.0]
        # most recent remembered enemy
        best = None
        for mid, (ts, pos, vel, hp) in self.last_seen.items():
            if t - ts < 8 and (best is None or ts > best[0]):
                best = (ts, pos)
        p = me["pos"]
        if best is not None:
            return [math.atan2(best[1][1] - p[1], best[1][0] - p[0]), 0.0]
        base = math.atan2(move[1], move[0]) if move != (0.0, 0.0) else me["aim"][0]
        sweep = math.sin(t * 0.9 + self.scan_phase) * 1.4
        if move == (0.0, 0.0):
            return [wrap(self.scan_phase + t * 0.8), 0.0]
        return [wrap(base + sweep), 0.0]

    def weapons_logic(self, me, weapons, kinds, aim_point, aim_vel, target, muzzle, t, obs):
        p = me["pos"]
        fire = [False, False]
        dist = math.dist(muzzle, aim_point)
        actual = me["aim"]
        is_mech = target is not None and abs(aim_point[0] - target["pos"][0]) < 1e-6
        radius = 2.5 if is_mech else 1.4
        tol = math.atan2(radius * 0.8, max(dist, 1.0)) + 0.004
        desired = None
        los = self.W.los(muzzle, aim_point)
        # instant weapons (laser, lightning) aim straight at the target one decision ahead;
        # the gun leads. Pick the aim for the weapon that matters this tick, then let each
        # weapon fire only when ITS own solution is aligned.
        inst_pt = [aim_point[0] + aim_vel[0] * P["inst_lead"], aim_point[1] + aim_vel[1] * P["inst_lead"],
                   aim_point[2]]
        inst = self.aim_to(muzzle, inst_pt)
        gun_aim = None
        if "gun" in kinds:
            g = weapons[kinds["gun"]]
            lp = self.lead(muzzle, aim_point, aim_vel, g["stats"]["speed"])
            gun_aim = self.aim_to(muzzle, lp)
        laser_hot = False
        if "laser" in kinds:
            L = weapons[kinds["laser"]]
            laser_hot = (L.get("charge", 0) > 0 or L["cooldown"] < 0.2) and dist < L["stats"]["range"] * 0.97
        light_close = "lightning" in kinds and dist < weapons[kinds["lightning"]]["stats"]["range"] * 1.1
        if laser_hot or light_close or gun_aim is None:
            aim = inst
        else:
            aim = gun_aim
        if gun_aim is not None:
            g = weapons[kinds["gun"]]
            err = abs(wrap(actual[0] - gun_aim[0])) + abs(actual[1] - gun_aim[1])
            if g.get("reload", 0) <= 0 and g.get("ammo", 1) > 0 and dist < g["stats"]["range"] * 0.95 \
                    and err < tol * 1.3 and los:
                fire[kinds["gun"]] = True
        if "laser" in kinds:
            L = weapons[kinds["laser"]]
            st = L["stats"]
            charge = L.get("charge", 0)
            full = st["charge"]
            err = abs(wrap(actual[0] - inst[0])) + abs(actual[1] - inst[1])
            in_range = dist < st["range"] * 0.97
            if charge > 0:
                # hold until full and aligned, then release; never let hold_s auto-fire it blind
                if charge >= full * 0.97 and err < tol and los and in_range:
                    fire[kinds["laser"]] = False
                elif charge >= full + st.get("hold_s", 2) - 0.3 and err < tol * 3:
                    fire[kinds["laser"]] = False
                else:
                    fire[kinds["laser"]] = True
            elif L["cooldown"] <= 0.05 and in_range and los and (is_mech or dist < 80):
                fire[kinds["laser"]] = True
        if "lightning" in kinds:
            Lg = weapons[kinds["lightning"]]
            st = Lg["stats"]
            yaw_err = abs(wrap(actual[0] - math.atan2(aim_point[1] - p[1], aim_point[0] - p[0])))
            if dist < st["range"] * 0.95 and yaw_err < math.radians(st["cone_deg"]) * 0.45 and los \
                    and me["energy"] > st["energy"] + 4 and Lg["cooldown"] <= 0.05:
                fire[kinds["lightning"]] = True
        if "mortar" in kinds:
            M = weapons[kinds["mortar"]]
            st = M["stats"]
            hd = math.hypot(aim_point[0] - p[0], aim_point[1] - p[1])
            if 25 < hd < 230 and is_mech and len(kinds) == 1 or ("mortar" in kinds and len(kinds) == 2
                                                                    and "gun" not in kinds and "laser" not in kinds
                                                                    and hd > 50):
                tof = hd / (st["speed"] * 0.7)
                lp = [aim_point[0] + aim_vel[0] * tof, aim_point[1] + aim_vel[1] * tof, aim_point[2] - 4.0]
                src = [p[0], p[1], p[2] + 9.5]
                sol = self.mortar_aim(src, lp, st["speed"], st["gravity"])
                if sol is not None:
                    yaw, lo, hi = sol
                    pitch = lo if self.W.los(src, lp) else hi
                    mp = me.get("stats", {})
                    pitch = max(-0.7, min(1.3, pitch))
                    aim = [yaw, pitch]
                    err = abs(wrap(actual[0] - yaw)) + abs(actual[1] - pitch)
                    if M["cooldown"] <= 0 and err < 0.04:
                        fire[kinds["mortar"]] = True
            elif "gun" in kinds or "laser" in kinds or "lightning" in kinds:
                pass
            if 25 < hd < 230 and is_mech and ("gun" in kinds) and not fire[kinds["gun"]] and not los:
                # target behind cover: lob a shell instead of wasting bullets
                tof = hd / (st["speed"] * 0.7)
                lp = [aim_point[0] + aim_vel[0] * tof, aim_point[1] + aim_vel[1] * tof, aim_point[2] - 4.0]
                src = [p[0], p[1], p[2] + 9.5]
                sol = self.mortar_aim(src, lp, st["speed"], st["gravity"])
                if sol is not None:
                    yaw, lo, hi = sol
                    aim = [yaw, max(-0.7, min(1.3, hi))]
                    err = abs(wrap(actual[0] - aim[0])) + abs(actual[1] - aim[1])
                    if M["cooldown"] <= 0 and err < 0.04:
                        fire[kinds["mortar"]] = True
        pitch = max(-0.7, min(1.3, aim[1]))
        return [aim[0], pitch], fire


if __name__ == "__main__":
    mf.run(Bot)
