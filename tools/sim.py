"""Rough local simulator for tuning (NOT the official game).

Our bot (bot.Bot) vs 15 "house" hunters that share vision, never fight each other, and focus us.
Simplified physics: ground movement on the nav grid, no jumping/jetting.

  python3 tools/sim.py --matches 6 --seed 1
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mechfight as mf  # noqa: E402
import bot as botmod  # noqa: E402
from world import World  # noqa: E402

DT = 0.1
HOUSE_ACC = float(os.environ.get("HOUSE_ACC", "1.0"))
HOUSE_FIRE = float(os.environ.get("HOUSE_FIRE", "0.6"))
WAVES = os.environ.get("WAVES", "1") == "1"   # house bots engage weakest tier first
TIER_RANK = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_cfg():
    with gzip.open(os.path.join(ROOT, "check", "stream.jsonl.gz"), "rt") as f:
        return json.loads(f.readline())["config"]


class Mech:
    def __init__(self, mid, pos, cfg, house_tier=None):
        m = cfg["mech"]
        self.id = mid
        self.pos = list(pos)
        self.vel = [0.0, 0.0, 0.0]
        self.aim = [0.0, 0.0]
        self.cmd_aim = [0.0, 0.0]
        self.hp = m["hp"]
        self.max_hp = m["hp"]
        self.energy = m["energy"]
        self.spent_t = -9
        self.last_combat = -99
        self.cubes = 0
        self.alive = True
        self.tier = house_tier
        g = dict(cfg["weapons"]["gun"])
        self.weapons = [{"kind": "gun", "cooldown": 0.0, "ammo": g["mag"], "reload": 0.0, "stats": g}, None]
        self.events = []
        self.charging = False
        self.offer = None
        self.upgrades = []
        self.armor = 0.0
        self.kills = 0
        self.death_t = None
        self.memory = {}

    def kinds(self):
        return [w["kind"] for w in self.weapons if w]


class Obj:
    def __init__(self, oid, kind, pos, hp, radius, height):
        self.id, self.kind, self.pos, self.hp = oid, kind, list(pos), hp
        self.radius, self.height, self.alive = radius, height, True
        self.home = list(pos)
        self.cubes = 0
        self.vel = [0.0, 0.0, 0.0]
        self.aggro = None
        self.cd = 0.0


class Sim:
    def __init__(self, W, cfg, seed, tiers):
        self.W = W
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.t = 0.0
        self.tick = 0
        self.mechs = []
        self.bullets = []
        self.shells = []
        self.pickups = []
        self.cubes = []
        self.next_id = 8000
        self.cube_id = 5000
        pts = []
        while len(pts) < 16:
            c = self.rand_walk(margin=40)
            if all(math.hypot(c[0] - q[0], c[1] - q[1]) > 110 for q in pts):
                pts.append(c)
        for i in range(16):
            x, y = pts[i]
            self.mechs.append(Mech(i, [x, y, W.surface(x, y)], cfg, None if i == 0 else tiers[i - 1]))
        for _ in range(cfg["loot"]["floor_count"]):
            x, y = self.rand_walk(margin=20)
            k = self.rng.choice(["mortar", "laser", "lightning"])
            self.pickups.append({"id": self.next_id, "kind": k, "pos": [x, y, W.surface(x, y)]})
            self.next_id += 1
        self.objs = []
        for k in range(cfg["crates"]["count"]):
            x, y = self.rand_walk(margin=20)
            self.objs.append(Obj(1000 + k, "crate", [x, y, W.surface(x, y)], cfg["crates"]["hp"], 1.4, 2.4))
        for k in range(cfg["npc"]["count"]):
            x, y = self.rand_walk(margin=30)
            self.objs.append(Obj(100 + k, "npc", [x, y, W.surface(x, y)], cfg["npc"]["hp"], 1.8, 3.5))
        z = cfg["zone"]
        self.zphase = 0
        self.zc, self.zr = list(z["start_center"]), z["start_radius"]
        self.plan_zone()
        self.house_known = None  # shared vision: (t, pos, vel)
        self.stat = {}

    def rand_walk(self, margin):
        W = self.W
        while True:
            x = self.rng.uniform(W.minx + margin, W.maxx - margin)
            y = self.rng.uniform(W.miny + margin, W.maxy - margin)
            c = W.cell(x, y)
            if W.walk[c] and not W.isprism[c]:
                return x, y

    def plan_zone(self):
        ph = self.cfg["zone"]["phases"]
        if self.zphase >= len(ph):
            self.znc, self.znr = self.zc, 0
            self.zs, self.ze, self.zdps = 1e9, 1e9, ph[-1]["dps"]
            return
        p = ph[self.zphase]
        start = getattr(self, "ze", 0.0)
        self.zs = start + p["hold_s"]
        self.ze = self.zs + p["shrink_s"]
        r = p["radius"]
        for _ in range(100):
            a = self.rng.random() * 2 * math.pi
            d = self.rng.random() * max(0.0, self.zr - r) * 0.8
            c = [self.zc[0] + math.cos(a) * d, self.zc[1] + math.sin(a) * d]
            if self.W.minx + 60 < c[0] < self.W.maxx - 60 and self.W.miny + 60 < c[1] < self.W.maxy - 60:
                break
        self.znc, self.znr, self.zdps = c, r, p["dps"]

    def zone_now(self):
        t = self.t
        if t <= self.zs:
            return self.zc, self.zr
        f = min(1.0, (t - self.zs) / (self.ze - self.zs))
        return ([self.zc[0] + (self.znc[0] - self.zc[0]) * f, self.zc[1] + (self.znc[1] - self.zc[1]) * f],
                self.zr + (self.znr - self.zr) * f)

    def zone_obs(self):
        return {"center": list(self.zc), "radius": self.zr, "next_center": list(self.znc), "next_radius": self.znr,
                "shrink_start": self.zs, "shrink_end": self.ze, "dps": self.zdps}

    # ----------------------------------------------------------- perception
    def sees(self, a, b):
        dx, dy = b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]
        d = math.hypot(dx, dy)
        if d > 200:
            return False
        if d > 45 and abs(wrap(math.atan2(dy, dx) - a.aim[0])) > math.radians(60):
            return False
        return self.W.los([a.pos[0], a.pos[1], a.pos[2] + 8], [b.pos[0], b.pos[1], b.pos[2] + 4.5], step=2.0)

    def actor(self, m):
        return {"id": m.id, "kind": "mech", "pos": [round(v, 2) for v in m.pos], "vel": m.vel[:],
                "aim": m.aim[:], "hp": m.hp, "max_hp": m.max_hp, "cubes": m.cubes, "weapons": m.kinds(),
                "charging": m.charging}

    # ----------------------------------------------------------- damage
    def damage(self, src, dst, amount, weapon, origin):
        if not dst.alive:
            return
        if isinstance(dst, Obj):
            mult = (1 + 0.04 * min(src.cubes, 10)) if src else 1
            dst.hp -= amount * mult
            if src:
                src.events.append({"type": "hit", "target": dst.id, "amount": amount * mult, "weapon": weapon})
                if src.id == 0:
                    self.stat["dealt_obj"] = self.stat.get("dealt_obj", 0) + amount * mult
                if dst.kind == "npc":
                    dst.aggro = src
            if dst.hp <= 0:
                dst.alive = False
                n = 1 if dst.kind == "crate" else 2
                for k in range(n):
                    self.cubes.append({"id": self.cube_id, "pos": [dst.pos[0] + k, dst.pos[1], dst.pos[2]]})
                    self.cube_id += 1
                if dst.kind == "crate" and self.rng.random() < 0.5:
                    self.pickups.append({"id": self.next_id, "kind": self.rng.choice(["mortar", "laser", "lightning"]),
                                         "pos": [dst.pos[0] - 1, dst.pos[1], dst.pos[2]]})
                    self.next_id += 1
            return
        bearing = math.atan2(origin[1] - dst.pos[1], origin[0] - dst.pos[0])
        rear = abs(wrap(bearing - dst.aim[0])) > math.pi / 2
        mult = (1 + 0.04 * min(src.cubes, 10)) if src else 1
        a = amount * mult * (1.5 if rear else 1.0) * (1 - dst.armor)
        dst.hp -= a
        dst.last_combat = self.t
        if src is not None and src.id == 0:
            self.stat["dealt_" + weapon] = self.stat.get("dealt_" + weapon, 0) + a
        if dst.id == 0:
            self.stat["taken_" + weapon] = self.stat.get("taken_" + weapon, 0) + a
            if rear:
                self.stat["taken_rear"] = self.stat.get("taken_rear", 0) + a
        if src:
            src.last_combat = self.t
            src.events.append({"type": "hit", "target": dst.id, "amount": a, "weapon": weapon})
        dst.events.append({"type": "damaged", "amount": a, "weapon": weapon, "bearing": bearing,
                           "by": src.id if src else None, "rear": rear})
        if src is not None and src.tier is not None and dst.tier is None:
            self.house_known = (self.t, dst.pos[:], dst.vel[:])
        if dst.hp <= 0:
            dst.alive = False
            dst.death_t = self.t
            if src:
                src.kills += 1
                src.events.append({"type": "kill", "target": dst.id})
            for k in range(dst.cubes + 2):
                a2 = self.rng.random() * 6.28
                self.cubes.append({"id": self.cube_id, "pos": [dst.pos[0] + math.cos(a2) * 2,
                                                               dst.pos[1] + math.sin(a2) * 2, dst.pos[2]]})
                self.cube_id += 1
            for w in dst.weapons:
                if w and w["kind"] != "gun":
                    self.pickups.append({"id": self.next_id, "kind": w["kind"], "pos": dst.pos[:]})
                    self.next_id += 1

    def ray_hit(self, shooter, o, d, rng):
        """First mech hit by a ray from o along unit d within rng (cylinder test)."""
        best, bt = None, rng
        for m in self.mechs + self.objs:
            if not m.alive or m is shooter:
                continue
            if isinstance(m, Obj):
                r, h = m.radius, m.height
            else:
                r = 2.5 * (1 + 0.05 * m.cubes)
                h = 9 * (1 + 0.05 * m.cubes)
            fx, fy = m.pos[0] - o[0], m.pos[1] - o[1]
            dd = d[0] * d[0] + d[1] * d[1]
            if dd < 1e-9:
                continue
            tt = (fx * d[0] + fy * d[1]) / dd
            if tt < 0 or tt > bt:
                continue
            px, py = o[0] + d[0] * tt - m.pos[0], o[1] + d[1] * tt - m.pos[1]
            if px * px + py * py > r * r:
                continue
            zz = o[2] + d[2] * tt
            if m.pos[2] <= zz <= m.pos[2] + h:
                best, bt = m, tt
        return best, bt

    # ----------------------------------------------------------- step
    def apply_action(self, m, a):
        cfg = self.cfg
        M = cfg["mech"]
        if "aim" in a and a["aim"]:
            m.cmd_aim = [a["aim"][0], max(M["min_pitch"], min(M["max_pitch"], a["aim"][1]))]
        mv = a.get("move", [0, 0])
        mx, my = mv[0], mv[1]
        l = math.hypot(mx, my)
        if l > 1:
            mx, my = mx / l, my / l
        boost = a.get("boost") and m.energy > 1 and l > 0.1
        spd = M["boost_speed"] if boost else M["walk_speed"]
        if m.charging:
            spd *= 0.5
        if boost:
            m.energy -= M["boost_drain"] * DT
            m.spent_t = self.t
        tvx, tvy = mx * spd, my * spd
        acc = M["ground_accel"] * DT
        dvx, dvy = tvx - m.vel[0], tvy - m.vel[1]
        dl = math.hypot(dvx, dvy)
        if dl > acc:
            dvx, dvy = dvx / dl * acc, dvy / dl * acc
        m.vel[0] += dvx
        m.vel[1] += dvy
        # move with sliding
        W = self.W
        for vx, vy in ((m.vel[0], m.vel[1]), (m.vel[0], 0.0), (0.0, m.vel[1])):
            nxp, nyp = m.pos[0] + vx * DT, m.pos[1] + vy * DT
            c0 = W.cell(m.pos[0], m.pos[1])
            c1 = W.cell(nxp, nyp)
            if not W.in_bounds(nxp, nyp, 3):
                continue
            ok = c1 == c0 or W.walk[c1] or W.surf[c1] <= W.surf[c0] - 0.5
            diag = 1.414 if (c1 % W.nx != c0 % W.nx and c1 // W.nx != c0 // W.nx) else 1.0
            if ok and W.surf[c1] - W.surf[c0] <= (1.2 if (W.isprism[c0] or W.isprism[c1]) else W.slope_rise * diag):
                m.pos[0], m.pos[1] = nxp, nyp
                m.pos[2] = W.surf[c1]
                if (vx, vy) != (m.vel[0], m.vel[1]):
                    m.vel[0], m.vel[1] = vx, vy
                break
        else:
            m.vel[0] = m.vel[1] = 0.0
        # aim slew
        tr = M["turn_rate"] * DT
        dy = wrap(m.cmd_aim[0] - m.aim[0])
        m.aim[0] = wrap(m.aim[0] + max(-tr, min(tr, dy)))
        pr = M["pitch_rate"] * DT
        m.aim[1] += max(-pr, min(pr, m.cmd_aim[1] - m.aim[1]))
        # take
        if a.get("take") is not None:
            pk = next((p for p in self.pickups if p["id"] == a["take"]), None)
            if pk is None:
                m.events.append({"type": "take_failed", "id": a["take"], "reason": "gone"})
            elif math.hypot(pk["pos"][0] - m.pos[0], pk["pos"][1] - m.pos[1]) > 4:
                m.events.append({"type": "take_failed", "id": a["take"], "reason": "range"})
            elif pk["kind"] in m.kinds():
                m.events.append({"type": "take_failed", "id": a["take"], "reason": "have"})
            else:
                slot = 1 if m.weapons[1] is None else (0 if m.weapons[0] is None else a.get("drop"))
                if slot is None:
                    m.events.append({"type": "take_failed", "id": a["take"], "reason": "full"})
                else:
                    old = m.weapons[slot]
                    self.pickups.remove(pk)
                    st = dict(cfg["weapons"][pk["kind"]])
                    w = {"kind": pk["kind"], "cooldown": max(1.0, st.get("cooldown", 0)), "stats": st}
                    if pk["kind"] == "laser":
                        w["charge"] = 0.0
                    if pk["kind"] == "gun":
                        w.update(ammo=st["mag"], reload=0.0)
                    m.weapons[slot] = w
                    m.events.append({"type": "took", "id": pk["id"], "kind": pk["kind"], "slot": slot})
                    if old:
                        self.pickups.append({"id": self.next_id, "kind": old["kind"], "pos": m.pos[:]})
                        self.next_id += 1
        # fire
        fire = a.get("fire", [False, False])
        m.charging = False
        for i, w in enumerate(m.weapons):
            if not w:
                continue
            w["cooldown"] = max(0.0, w["cooldown"] - DT)
            trig = bool(fire[i]) if i < len(fire) else False
            k = w["kind"]
            st = w["stats"]
            cy, sy = math.cos(m.aim[0]), math.sin(m.aim[0])
            cp, sp = math.cos(m.aim[1]), math.sin(m.aim[1])
            d = [cp * cy, cp * sy, sp]
            o = [m.pos[0] + d[0] * 3, m.pos[1] + d[1] * 3, m.pos[2] + 7 + d[2] * 3]
            if k == "gun":
                if w["reload"] > 0:
                    w["reload"] = max(0.0, w["reload"] - DT)
                    if w["reload"] == 0:
                        w["ammo"] = st["mag"]
                if a.get("reload") and w["reload"] == 0 and w["ammo"] < st["mag"]:
                    w["reload"] = st["reload"]
                if trig and w["reload"] == 0 and w["ammo"] > 0 and w["cooldown"] <= 0:
                    nshots = 0
                    while w["cooldown"] <= 0 and w["ammo"] > 0 and nshots < 2:
                        sprd = math.radians(st["spread_deg"])
                        yy = m.aim[0] + self.rng.gauss(0, sprd)
                        pp = m.aim[1] + self.rng.gauss(0, sprd)
                        v = [math.cos(pp) * math.cos(yy) * st["speed"], math.cos(pp) * math.sin(yy) * st["speed"],
                             math.sin(pp) * st["speed"]]
                        if m.id == 0:
                            self.stat["shots"] = self.stat.get("shots", 0) + 1
                        self.bullets.append({"o": m, "pos": o[:], "vel": v, "left": st["range"],
                                             "dmg": st["damage"], "id": self.next_id})
                        self.next_id += 1
                        w["ammo"] -= 1
                        w["cooldown"] += st["cooldown"]
                        nshots += 1
                    if w["ammo"] == 0:
                        w["reload"] = st["reload"]
            elif k == "laser":
                if w["cooldown"] > 0:
                    w["charge"] = 0.0
                    continue
                if trig:
                    w["charge"] = min(st["charge"] + st["hold_s"], w["charge"] + DT)
                    m.charging = True
                    if w["charge"] < st["charge"] + st["hold_s"] - 1e-6:
                        continue
                if w["charge"] > 0 and (not trig or w["charge"] >= st["charge"] + st["hold_s"] - 1e-6):
                    frac = st["min_frac"] + (1 - st["min_frac"]) * min(1.0, w["charge"] / st["charge"])
                    hit, tt = self.ray_hit(m, o, d, st["range"])
                    if hit is not None:
                        hp = [o[0] + d[0] * tt, o[1] + d[1] * tt, o[2] + d[2] * tt]
                        if self.W.los(o, hp):
                            self.damage(m, hit, st["damage"] * frac, "laser", o)
                    w["charge"] = 0.0
                    w["cooldown"] = st["cooldown"]
                    m.charging = False
            elif k == "lightning":
                if trig and w["cooldown"] <= 0 and m.energy >= st["energy"]:
                    m.energy -= st["energy"]
                    m.spent_t = self.t
                    w["cooldown"] = st["cooldown"]
                    best, bd = None, st["range"]
                    for e in self.mechs + [o for o in self.objs if o.alive and
                                           math.hypot(o.pos[0] - m.pos[0], o.pos[1] - m.pos[1]) < 50]:
                        if e is m or not e.alive:
                            continue
                        dd = math.hypot(e.pos[0] - m.pos[0], e.pos[1] - m.pos[1])
                        if dd < bd and abs(wrap(math.atan2(e.pos[1] - m.pos[1], e.pos[0] - m.pos[0]) - m.aim[0])) \
                                < math.radians(st["cone_deg"]) and \
                                self.W.los([m.pos[0], m.pos[1], m.pos[2] + 7], [e.pos[0], e.pos[1], e.pos[2] + 4.5]):
                            best, bd = e, dd
                    if best is not None:
                        self.damage(m, best, st["damage"], "lightning", m.pos)
            elif k == "mortar":
                if trig and w["cooldown"] <= 0:
                    w["cooldown"] = st["cooldown"]
                    o2 = [m.pos[0] + d[0] * 3, m.pos[1] + d[1] * 3, m.pos[2] + 9.5]
                    self.shells.append({"o": m, "pos": o2, "vel": [d[0] * st["speed"], d[1] * st["speed"],
                                                                   d[2] * st["speed"]], "st": st,
                                        "id": self.next_id, "age": 0})
                    self.next_id += 1
        # pickup cubes
        for cb in list(self.cubes):
            if math.hypot(cb["pos"][0] - m.pos[0], cb["pos"][1] - m.pos[1]) < 3.5:
                self.cubes.remove(cb)
                m.cubes += 1
                inc = cfg["mech"]["hp"] * 0.04 if m.cubes <= 10 else 0
                m.max_hp += inc
                m.hp += inc
                m.events.append({"type": "pickup", "cubes": m.cubes})
                if m.cubes in cfg["cubes"]["upgrade_thresholds"] and m.tier is None:
                    m.offer = {"id": m.cubes, "expires": self.t + 5,
                               "choices": [{"id": "armor", "name": "", "desc": ""},
                                           {"id": "plating", "name": "", "desc": ""},
                                           {"id": "servos", "name": "", "desc": ""}]}
        if a.get("pick") is not None and m.offer:
            cid = m.offer["choices"][int(a["pick"])]["id"]
            if cid == "armor":
                m.armor = min(0.24, m.armor + 0.08)
            elif cid == "plating":
                m.max_hp *= 1.15
                m.hp *= 1.15
            m.offer = None

    def physics(self):
        # bullets
        for b in list(self.bullets):
            o = b["pos"]
            v = b["vel"]
            step = [v[0] * DT, v[1] * DT, v[2] * DT]
            ln = math.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2)
            d = [step[0] / ln, step[1] / ln, step[2] / ln]
            hit, tt = self.ray_hit(b["o"], o, d, min(ln, b["left"]))
            end = [o[0] + d[0] * tt, o[1] + d[1] * tt, o[2] + d[2] * tt] if hit else \
                [o[0] + step[0], o[1] + step[1], o[2] + step[2]]
            if not self.W.los(o, end):
                self.bullets.remove(b)
                continue
            if hit is not None:
                self.damage(b["o"], hit, b["dmg"], "gun", o)
                self.bullets.remove(b)
                continue
            b["pos"] = end
            b["left"] -= ln
            if b["left"] <= 0:
                self.bullets.remove(b)
        for s in list(self.shells):
            st = s["st"]
            s["age"] += DT
            o = s["pos"][:]
            s["vel"][2] -= st["gravity"] * DT
            s["pos"] = [o[0] + s["vel"][0] * DT, o[1] + s["vel"][1] * DT, o[2] + s["vel"][2] * DT]
            p = s["pos"]
            boom = p[2] <= self.W.surface(p[0], p[1]) or not self.W.los(o, p) or s["age"] > st["ttl"]
            for m in self.mechs:
                if m.alive and m is not s["o"] and math.hypot(m.pos[0] - p[0], m.pos[1] - p[1]) < 2.5 \
                        and m.pos[2] <= p[2] <= m.pos[2] + 9:
                    boom = True
            if boom:
                self.shells.remove(s)
                for m in self.mechs:
                    if not m.alive:
                        continue
                    dd = math.dist([m.pos[0], m.pos[1], m.pos[2] + 4.5], p)
                    if dd < st["splash_radius"] + 2.5:
                        f = 1 - (1 - st["splash_edge"]) * min(1, max(0, dd - 2.5) / st["splash_radius"])
                        dmg = st["splash"] * f + (st["damage"] if dd < 3 else 0)
                        if m is s["o"]:
                            dmg *= 0.5
                        self.damage(s["o"], m, dmg, "mortar", p)
        # npc drones: shoot their attacker (or mechs close to home)
        for o in self.objs:
            if not o.alive or o.kind != "npc":
                continue
            o.cd -= DT
            tgt = o.aggro if (o.aggro is not None and o.aggro.alive and
                              math.hypot(o.aggro.pos[0] - o.pos[0], o.aggro.pos[1] - o.pos[1]) < 90) else None
            if tgt is None:
                o.aggro = None
                for m in self.mechs:
                    if m.alive and math.hypot(m.pos[0] - o.pos[0], m.pos[1] - o.pos[1]) < 45:
                        tgt = m
                        break
            if tgt is not None and o.cd <= 0:
                o.cd = 1.0
                d = math.hypot(tgt.pos[0] - o.pos[0], tgt.pos[1] - o.pos[1])
                if self.W.los([o.pos[0], o.pos[1], o.pos[2] + 3], [tgt.pos[0], tgt.pos[1], tgt.pos[2] + 4.5]) \
                        and self.rng.random() < max(0.15, 0.8 - d / 120):
                    self.damage(None, tgt, 25, "npc", o.pos)
        # zone, regen, energy
        zc, zr = self.zone_now()
        for m in self.mechs:
            if not m.alive:
                continue
            if math.hypot(m.pos[0] - zc[0], m.pos[1] - zc[1]) > zr:
                m.hp -= self.zdps * DT
                m.last_combat = self.t
                m.events.append({"type": "damaged", "amount": self.zdps * DT, "weapon": "zone", "bearing": None,
                                 "by": None})
                if m.hp <= 0:
                    m.alive = False
                    m.death_t = self.t
            elif self.t - m.last_combat > self.cfg["mech"]["regen_delay"]:
                m.hp = min(m.max_hp, m.hp + self.cfg["mech"]["regen_per_s"] * DT)
            if self.t - m.spent_t > 0.6:
                m.energy = min(100, m.energy + 18 * DT)
        if self.t >= self.ze:
            self.zc, self.zr = self.znc, self.znr
            self.zphase += 1
            self.plan_zone()

    # ----------------------------------------------------------- house AI
    def house_act(self, m, me):
        """House bot: shares vision of us, hunts us, strafes unpredictably, uses whatever it holds."""
        tier = m.tier
        acc = {"easy": 0.10, "medium": 0.05, "hard": 0.025, "expert": 0.012}[tier] * HOUSE_ACC
        lead = tier != "easy"
        a = {"tick": self.tick}
        zc, zr = self.zone_now()
        kinds = m.kinds()
        target = None
        active = True
        if WAVES:
            low = min(TIER_RANK[h.tier] for h in self.mechs[1:] if h.alive)
            active = TIER_RANK[tier] <= low
            # inactive bots still defend themselves when we're close
            if not active and me.alive and math.hypot(me.pos[0] - m.pos[0], me.pos[1] - m.pos[1]) < 60 \
                    and self.sees(m, me):
                active = True
        if active and me.alive and self.house_known and self.t - self.house_known[0] < 1.0:
            target = self.house_known
        goal = None
        if target is None:
            # grab a laser/lightning nearby when the slot is empty
            if m.weapons[1] is None and tier != "easy":
                best = None
                for pk in self.pickups:
                    if pk["kind"] == "mortar":
                        continue
                    d = math.hypot(pk["pos"][0] - m.pos[0], pk["pos"][1] - m.pos[1])
                    if d < 90 and (best is None or d < best[0]):
                        best = (d, pk)
                if best:
                    goal = best[1]["pos"]
            if goal is None and active and me.alive and self.house_known and self.t - self.house_known[0] < 30:
                goal = self.house_known[1]
            if goal is None and math.hypot(m.pos[0] - self.znc[0], m.pos[1] - self.znc[1]) > self.znr * 0.5:
                goal = self.znc
        if math.hypot(m.pos[0] - zc[0], m.pos[1] - zc[1]) > zr - 10:
            goal, target = self.znc, None
        mv = [0.0, 0.0]
        if not hasattr(m, "sdir"):
            m.sdir, m.suntil = 1, 0.0
        if target is not None:
            tp = target[1]
            d = math.hypot(tp[0] - m.pos[0], tp[1] - m.pos[1])
            ux, uy = (tp[0] - m.pos[0]) / max(d, 1e-6), (tp[1] - m.pos[1]) / max(d, 1e-6)
            if "lightning" in kinds:
                pref = 30
            else:
                pref = {"easy": 60, "medium": 55, "hard": 45, "expert": 40}[tier]
            r = max(-1, min(1, (d - pref) / 20))
            if self.t > m.suntil:
                if self.rng.random() < 0.7:
                    m.sdir = -m.sdir
                m.suntil = self.t + self.rng.uniform(0.5, 1.4)
            sl = 0.3 if tier == "easy" else 0.9
            mv = [ux * r - uy * m.sdir * sl, uy * r + ux * m.sdir * sl]
            src = [m.pos[0], m.pos[1], m.pos[2] + 7]
            c = [tp[0], tp[1], tp[2] + 4.5]
            if lead:
                tt = d / 252 + 0.05
                c = [c[0] + target[2][0] * tt, c[1] + target[2][1] * tt, c[2]]
            yaw = math.atan2(c[1] - src[1], c[0] - src[0]) + self.rng.gauss(0, acc)
            pitch = math.atan2(c[2] - src[2], d) + self.rng.gauss(0, acc * 0.5)
            a["aim"] = [yaw, pitch]
            err = abs(wrap(m.aim[0] - yaw))
            vis = self.W.los(src, [tp[0], tp[1], tp[2] + 4.5])
            fire = [False, False]
            for i, w in enumerate(m.weapons):
                if not w:
                    continue
                if w["kind"] == "gun":
                    fire[i] = vis and err < 0.06 and d < 230 and self.rng.random() < HOUSE_FIRE
                elif w["kind"] == "laser":
                    fire[i] = vis and d < 270 and (w.get("charge", 0) < w["stats"]["charge"] or err > 0.02)
                elif w["kind"] == "lightning":
                    fire[i] = vis and d < 44 and err < 0.2 and m.energy > 5
            a["fire"] = fire
        elif goal is not None:
            d = math.hypot(goal[0] - m.pos[0], goal[1] - m.pos[1])
            if d > 3:
                if not hasattr(m, "path") or m.path_t + 3 < self.t or not m.path or \
                        math.hypot(m.path[-1][0] - goal[0], m.path[-1][1] - goal[1]) > 10:
                    m.path = self.W.astar(m.pos[:2], goal[:2], budget_s=0.004) or [tuple(goal[:2])]
                    m.path_t = self.t
                    m.pi = 0
                while m.pi < len(m.path) - 1 and math.hypot(m.path[m.pi][0] - m.pos[0],
                                                              m.path[m.pi][1] - m.pos[1]) < 3:
                    m.pi += 1
                wp = m.path[m.pi]
                dd = math.hypot(wp[0] - m.pos[0], wp[1] - m.pos[1]) or 1
                mv = [(wp[0] - m.pos[0]) / dd, (wp[1] - m.pos[1]) / dd]
                a["aim"] = [math.atan2(mv[1], mv[0]), 0.0]
        a["move"] = mv
        for pk in self.pickups:
            if math.hypot(pk["pos"][0] - m.pos[0], pk["pos"][1] - m.pos[1]) < 4 and pk["kind"] != "mortar" \
                    and pk["kind"] not in kinds and m.weapons[1] is None and tier != "easy":
                a["take"] = pk["id"]
                break
        return a

    # ----------------------------------------------------------- our obs
    def obs_for(self, m):
        def vis_item(p):
            dx, dy = p[0] - m.pos[0], p[1] - m.pos[1]
            d = math.hypot(dx, dy)
            return d < 45 or (d < 200 and abs(wrap(math.atan2(dy, dx) - m.aim[0])) < math.radians(60))
        seen = []
        for e in self.mechs:
            if e is m or not e.alive:
                continue
            if self.sees(m, e):
                m.memory[e.id] = self.t
            if self.t - m.memory.get(e.id, -99) <= 5:
                seen.append(self.actor(e))
        projs = []
        for b in self.bullets:
            if b["o"] is not m and math.dist(b["pos"], m.pos) < 150:
                projs.append({"id": b["id"], "kind": "gun", "pos": b["pos"][:], "vel": b["vel"][:]})
        for s in self.shells:
            if s["o"] is not m and math.dist(s["pos"], m.pos) < 250:
                projs.append({"id": s["id"], "kind": "mortar", "pos": s["pos"][:], "vel": s["vel"][:]})
        def vis_item(p):
            dx, dy = p[0] - m.pos[0], p[1] - m.pos[1]
            d = math.hypot(dx, dy)
            return d < 45 or (d < 200 and abs(wrap(math.atan2(dy, dx) - m.aim[0])) < math.radians(60))
        zc, zr = self.zone_now()
        ev, m.events = m.events, []
        stats = dict(self.cfg["mech"])
        stats.update(muzzle_height=7 * (1 + 0.05 * m.cubes), eye_height=8 * (1 + 0.05 * m.cubes))
        return {"type": "obs", "tick": self.tick, "t": round(self.t, 2),
                "alive": sum(1 for e in self.mechs if e.alive),
                "self": {"id": m.id, "pos": m.pos[:], "vel": m.vel[:], "aim": m.aim[:], "hp": m.hp,
                         "max_hp": m.max_hp, "energy": m.energy, "max_energy": 100, "grounded": True,
                         "cubes": m.cubes, "upgrades": [], "weapons": json.loads(json.dumps(m.weapons)),
                         "in_zone": math.hypot(m.pos[0] - zc[0], m.pos[1] - zc[1]) < zr, "stats": stats},
                "mechs": seen + [{"id": o.id, "kind": "npc", "pos": o.pos[:], "vel": [0.0, 0.0, 0.0], "aim": [0, 0],
                                  "hp": o.hp, "max_hp": 350} for o in self.objs
                                 if o.alive and o.kind == "npc" and vis_item(o.pos) and
                                 math.hypot(o.pos[0] - m.pos[0], o.pos[1] - m.pos[1]) < 160],
                "projectiles": projs,
                "crates": [{"id": o.id, "pos": o.pos[:], "hp": o.hp} for o in self.objs
                           if o.alive and o.kind == "crate" and vis_item(o.pos) and
                           math.hypot(o.pos[0] - m.pos[0], o.pos[1] - m.pos[1]) < 160],
                "cubes": [c for c in self.cubes if vis_item(c["pos"])],
                "pickups": [p for p in self.pickups if vis_item(p["pos"])],
                "sounds": [], "beams": [], "events": ev, "zone": self.zone_obs(), "offer": m.offer, "drops": []}

    def run(self, mybot, max_t=480, verbose=False):
        me = self.mechs[0]
        times = []
        while self.t < max_t:
            alive = [m for m in self.mechs if m.alive]
            if not me.alive or len(alive) <= 1:
                break
            # shared house vision of us
            for h in alive[1:]:
                if self.sees(h, me):
                    self.house_known = (self.t, me.pos[:], me.vel[:])
                    break
            o = self.obs_for(me)
            mf.draw.clear()
            t0 = time.perf_counter()
            a = mybot.act(o)
            times.append(time.perf_counter() - t0)
            self.apply_action(me, a)
            for h in alive[1:]:
                self.apply_action(h, self.house_act(h, me))
            self.physics()
            self.t += DT
            self.tick += 2
            if verbose and self.tick % 100 == 0:
                print(f"t={self.t:5.1f} mode={mybot.mode:12s} hp={me.hp:6.0f} cubes={me.cubes} "
                      f"w={me.kinds()} kills={me.kills} alive={len(alive)}", file=sys.stderr)
        alive_n = sum(1 for m in self.mechs if m.alive)
        place = alive_n + (0 if me.alive else 1) if not me.alive else 1 if alive_n == 1 else alive_n
        if me.alive and alive_n > 1:
            place = 1 + sum(1 for m in self.mechs[1:] if m.alive and m.hp > me.hp)
        return {"place": place, "kills": me.kills, "t": round(self.t, 1), "cubes": me.cubes,
                "weapons": me.kinds(), "max_ms": round(max(times) * 1000, 1) if times else 0,
                "stat": {k: round(v) for k, v in sorted(self.stat.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--example", action="store_true", help="run the kit example bot instead")
    args = ap.parse_args()
    cfg = load_cfg()
    W = World(os.path.join(ROOT, "arena"))
    tiers = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
    scores = []
    for k in range(args.matches):
        sim = Sim(W, cfg, args.seed * 1000 + k, tiers)
        hello = {"type": "hello", "slot": 0, "seed": k, "map": {"dir": os.path.join(ROOT, "arena")}, "config": cfg}
        if args.example:
            src = open(os.path.join(ROOT, "..", "kit_download", "extracted", "mechfight-kit", "python",
                                    "example_bot.py")).read().replace("mf.run(Bot)", "")
            ns = {}
            exec(compile(src, "example_bot", "exec"), ns)
            b = ns["Bot"](hello)
            b.mode = "example"
        else:
            b = botmod.Bot(hello)
        r = sim.run(b, verbose=args.v)
        scores.append(16 - r["place"])
        print(k, r, flush=True)
    print("mean score", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
