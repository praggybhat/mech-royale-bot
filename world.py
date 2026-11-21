"""Static map model: terrain, solids, walkable grid, line of sight, A* pathfinding.

Standard library only. Built once at `hello` (well under the 20 s budget).
"""
from __future__ import annotations

import array
import heapq
import json
import math
import os
import time
from typing import List, Optional, Tuple

CELL = 2.0  # nav / solid grid resolution (m)
INF = float("inf")


class World:
    def __init__(self, map_dir: str, step_up: float = 1.2, max_slope_deg: float = 50.0,
                 mech_height: float = 9.0):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        self.tnx, self.tny = int(t["nx"]), int(t["ny"])
        self.tminx, self.tminy = float(t["minX"]), float(t["minY"])
        self.tstep = float(t["step"])
        scale = float(t.get("scale", 0.01))
        raw = array.array("h")
        with open(os.path.join(map_dir, "v2", t.get("file", "terrain.i16")), "rb") as f:
            raw.frombytes(f.read())
        if array.array("h", [1]).tobytes() != b"\x01\x00":
            raw.byteswap()
        self.terr = [v * scale for v in raw]

        b = w["bounds"]
        self.minx, self.maxx = float(b["minX"]), float(b["maxX"])
        self.miny, self.maxy = float(b["minY"]), float(b["maxY"])
        self.nx = int(math.ceil((self.maxx - self.minx) / CELL))
        self.ny = int(math.ceil((self.maxy - self.miny) / CELL))
        n = self.nx * self.ny
        self.step_up = step_up
        self.slope_rise = CELL * math.tan(math.radians(max_slope_deg)) * 0.9

        # ground height at each cell center
        ground = [0.0] * n
        for j in range(self.ny):
            y = self.miny + (j + 0.5) * CELL
            row = j * self.nx
            for i in range(self.nx):
                ground[row + i] = self.ground(self.minx + (i + 0.5) * CELL, y)
        self.gnd = ground
        # solids: per cell lowest base / highest top of prisms touching it (for LOS)
        sbase = [INF] * n
        stop = [-INF] * n
        surf = list(ground)
        isprism = bytearray(n)
        for p in w["prisms"]:
            base, top = float(p["base"]), float(p["top"])
            rings = [p["ring"]]
            holes = p.get("holes") or []
            cells = self._raster(rings[0], holes)
            for c in cells:
                if base < sbase[c]:
                    sbase[c] = base
                if top > stop[c]:
                    stop[c] = top
                g = ground[c]
                if base >= g + mech_height:
                    continue  # overhead: walk under
                # walkable surface becomes the top of the solid (roof or bridge deck)
                if top > surf[c]:
                    surf[c] = top
                    isprism[c] = 1
        self.sbase, self.stop = sbase, stop
        self.surf = surf
        self.isprism = isprism
        # walkable: a ground mech can stand here and has some clearance around it
        walk = bytearray(n)
        su = step_up
        sr = self.slope_rise
        nx, ny = self.nx, self.ny
        for j in range(ny):
            for i in range(nx):
                c = j * nx + i
                h = surf[c]
                ok = 1
                if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                    ok = 0
                else:
                    lim = su if isprism[c] else sr
                    for dj in (-1, 0, 1):
                        r = (j + dj) * nx
                        for di in (-1, 0, 1):
                            d = r + i + di
                            # clearance only from walls (buildings, bridges, stadium); a terrain slope is not a
                            # wall: its steepness is enforced per move (uphill limit), and downhill is always ok
                            if not isprism[d]:
                                continue
                            if surf[d] - h > su:
                                ok = 0
                                break
                        if not ok:
                            break
                walk[c] = ok
        self.walk = walk
        self.nbr = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                    (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]

    # ---------------------------------------------------------------- geometry
    def _raster(self, ring, holes) -> List[int]:
        """Cells whose center is inside ring and outside holes (scanline)."""
        def edges(r):
            pts = [(r[k], r[k + 1]) for k in range(0, len(r) - 1, 2)]
            return [(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
        polys = [edges(ring)] + [edges(h) for h in holes if len(h) >= 6]
        ys = ring[1::2]
        y0, y1 = min(ys), max(ys)
        j0 = max(0, int((y0 - self.miny) / CELL) - 1)
        j1 = min(self.ny - 1, int((y1 - self.miny) / CELL) + 1)
        out = []
        for j in range(j0, j1 + 1):
            y = self.miny + (j + 0.5) * CELL
            xs = []
            for poly in polys:
                for (ax, ay), (bx, by) in poly:
                    if (ay <= y < by) or (by <= y < ay):
                        xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                i0 = int(math.ceil((xs[k] - self.minx) / CELL - 0.5))
                i1 = int(math.floor((xs[k + 1] - self.minx) / CELL - 0.5))
                i0 = max(i0, 0)
                i1 = min(i1, self.nx - 1)
                row = j * self.nx
                out.extend(range(row + i0, row + i1 + 1))
        return out

    def ground(self, x: float, y: float) -> float:
        fx = (x - self.tminx) / self.tstep
        fy = (y - self.tminy) / self.tstep
        i = int(fx)
        j = int(fy)
        if i < 0:
            i, fx = 0, 0.0
        if j < 0:
            j, fy = 0, 0.0
        if i >= self.tnx - 1:
            i, fx = self.tnx - 2, float(self.tnx - 1)
        if j >= self.tny - 1:
            j, fy = self.tny - 2, float(self.tny - 1)
        ax = fx - i
        ay = fy - j
        t = self.terr
        k = j * self.tnx + i
        h00, h10 = t[k], t[k + 1]
        h01, h11 = t[k + self.tnx], t[k + self.tnx + 1]
        return (h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay

    def cell(self, x: float, y: float) -> int:
        i = int((x - self.minx) / CELL)
        j = int((y - self.miny) / CELL)
        i = min(max(i, 0), self.nx - 1)
        j = min(max(j, 0), self.ny - 1)
        return j * self.nx + i

    def center(self, c: int) -> Tuple[float, float]:
        return (self.minx + (c % self.nx + 0.5) * CELL, self.miny + (c // self.nx + 0.5) * CELL)

    def surface(self, x: float, y: float) -> float:
        return self.surf[self.cell(x, y)]

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.minx + margin < x < self.maxx - margin) and (self.miny + margin < y < self.maxy - margin)

    def los(self, a, b, step: float = 1.5) -> bool:
        """3D segment a->b clear of terrain and prisms (trees ignored)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy)
        n = int(d / step) + 1
        if n <= 1:
            return True
        sb, st = self.sbase, self.stop
        nx = self.nx
        mnx, mny = self.minx, self.miny
        for k in range(1, n):
            f = k / n
            x = a[0] + dx * f
            y = a[1] + dy * f
            z = a[2] + dz * f
            i = int((x - mnx) / CELL)
            j = int((y - mny) / CELL)
            if i < 0 or j < 0 or i >= nx or j >= self.ny:
                return False
            c = j * nx + i
            if sb[c] < z < st[c]:
                return False
            if z < self.gnd[c] - 0.3:
                return False
        return True

    def walk_line(self, a, b) -> bool:
        """Can a ground mech walk straight from a to b (2D)?"""
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        n = int(d / (CELL * 0.5)) + 1
        walk = self.walk
        prev = self.cell(a[0], a[1])
        hp = self.surf[prev]
        for k in range(1, n + 1):
            f = k / n
            c = self.cell(a[0] + dx * f, a[1] + dy * f)
            if c == prev:
                continue
            h = self.surf[c]
            if not walk[c] and h - hp > -0.5:
                return False
            lim = self.step_up if (self.isprism[c] or self.isprism[prev]) else self.slope_rise
            if h - hp > lim:
                return False
            prev, hp = c, h
        return True

    def nearest_walk(self, c: int, maxr: int = 12) -> int:
        if self.walk[c]:
            return c
        i0, j0 = c % self.nx, c // self.nx
        best, bd = -1, INF
        for r in range(1, maxr + 1):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        d = j * self.nx + i
                        if self.walk[d]:
                            dd = di * di + dj * dj
                            if dd < bd:
                                bd, best = dd, d
            if best >= 0:
                return best
        return c

    # ---------------------------------------------------------------- A*
    def astar(self, start_xy, goal_xy, budget_s: float = 0.012, penalty=None,
              weight: float = 1.6) -> Optional[List[Tuple[float, float]]]:
        """Path of waypoints from start to goal. Directional edges: step up <= limit, any step down.
        Returns None if not found within budget (caller falls back to straight line)."""
        t0 = time.perf_counter()
        nx = self.nx
        s = self.cell(*start_xy)
        g = self.nearest_walk(self.cell(*goal_xy))
        surf, walk, isp = self.surf, self.walk, self.isprism
        su, sr = self.step_up, self.slope_rise
        gi, gj = g % nx, g // nx
        if s == g:
            return [tuple(goal_xy)]
        gs = {s: 0.0}
        came = {s: -1}
        h0 = math.hypot(s % nx - gi, s // nx - gj)
        openq = [(h0 * weight, 0.0, s)]
        closed = set()
        best_c, best_h = s, h0
        it = 0
        nbr = self.nbr
        ny = self.ny
        found = False
        while openq:
            f, gc, c = heapq.heappop(openq)
            if c in closed:
                continue
            if c == g:
                found = True
                break
            closed.add(c)
            ci, cj = c % nx, c // nx
            hc = math.hypot(ci - gi, cj - gj)
            if hc < best_h:
                best_h, best_c = hc, c
            it += 1
            if (it & 255) == 0 and time.perf_counter() - t0 > budget_s:
                break
            h = surf[c]
            cp = isp[c]
            for di, dj, cost in nbr:
                i, j = ci + di, cj + dj
                if i <= 0 or j <= 0 or i >= nx - 1 or j >= ny - 1:
                    continue
                d = j * nx + i
                if not walk[d] or d in closed:
                    continue
                dh = surf[d] - h
                if dh > (su if (cp or isp[d]) else (sr * cost)):
                    continue
                if di and dj:  # no corner cutting
                    if not walk[cj * nx + i] or not walk[j * nx + ci]:
                        continue
                ng = gc + cost
                if penalty is not None:
                    ng += penalty(d)
                if ng < gs.get(d, INF):
                    gs[d] = ng
                    came[d] = c
                    heapq.heappush(openq, (ng + weight * math.hypot(i - gi, j - gj), ng, d))
        end = g if found else best_c
        if end == s:
            return None if not found else [tuple(goal_xy)]
        cells = []
        c = end
        while c != -1:
            cells.append(c)
            c = came.get(c, -1)
        cells.reverse()
        pts = [self.center(c) for c in cells[1:]]
        if found:
            pts[-1] = (goal_xy[0], goal_xy[1]) if self.walk[self.cell(*goal_xy)] else pts[-1]
        return pts


if __name__ == "__main__":
    import sys
    t0 = time.perf_counter()
    W = World(sys.argv[1] if len(sys.argv) > 1 else "arena")
    t1 = time.perf_counter()
    print("build", round(t1 - t0, 2), "s grid", W.nx, W.ny, "walkable", sum(W.walk) / len(W.walk))
    for a, b in [((-217, -22), (-23, 2)), ((250, -259), (-42, 7)), ((400, 300), (-400, -250))]:
        t2 = time.perf_counter()
        p = W.astar(a, b, budget_s=1.0)
        print(a, b, "len", len(p) if p else None, "ms", round((time.perf_counter() - t2) * 1000, 1),
              p[-1] if p else None)
