#!/usr/bin/env python3
"""Run N private real-engine matches (P in parallel) for one bot variant and summarize.

    python3 tools/livebatch.py NAME N [P] [MB_KEY=val ...]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name, n = sys.argv[1], int(sys.argv[2])
par = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 3
envs = [a for a in sys.argv[3:] if "=" in a]


def one(i):
    out = os.path.join(ROOT, "runs", name)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "live.py"), "--out", out, "--timeout", "900",
                        "--env", *envs], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (r.stderr or r.stdout)[-300:]}


with ThreadPoolExecutor(par) as ex:
    res = list(ex.map(one, range(n)))
ok = [r for r in res if "place" in r and r["place"]]
for r in res:
    print(json.dumps(r))
if ok:
    pts = [16 - r["place"] for r in ok]
    print(f"{name}: points {sum(pts)/len(pts):.2f}  kills {sum(r['kills'] for r in ok)/len(ok):.2f}  "
          f"dealt {sum(r['dealt'] for r in ok)/len(ok):.0f}  taken {sum(r['taken'] for r in ok)/len(ok):.0f}  "
          f"life {sum(r['t'] for r in ok)/len(ok):.0f}s  n={len(ok)}  {pts}")
