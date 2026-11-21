import ast
import sys

name = sys.argv[1]
rows = [l for l in open(f"/tmp/ab_{name}.txt") if l[:1].isdigit()]
rs = [ast.literal_eval(l.split(" ", 1)[1]) for l in rows]
if not rs:
    print(f"{name:14s} no results (see /tmp/ab_{name}.err)")
    sys.exit()
sc = [16 - r["place"] for r in rs]
k = [r["kills"] for r in rs]
t = [r["t"] for r in rs]
ffa = sum(1 for r in rs if r.get("ffa"))
mean = sum(sc) / len(sc)
sd = (sum((x - mean) ** 2 for x in sc) / max(1, len(sc) - 1)) ** 0.5
print(f"{name:14s} score {mean:.2f} ±{sd / len(sc) ** 0.5:.2f}  kills {sum(k) / len(k):.2f}  "
      f"life {sum(t) / len(t):.0f}s  ffa_flag {ffa}/{len(rs)}  n={len(rs)}")
