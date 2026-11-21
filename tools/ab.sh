#!/bin/bash
# usage: tools/ab.sh MATCHES SEED "name|ENV=1 ENV2=2" "name2|..." ...
N=$1; SEED=$2; shift 2
for spec in "$@"; do
  name=${spec%%|*}; envs=${spec#*|}
  ( env $envs python3 tools/sim.py --matches $N --seed $SEED > /tmp/ab_$name.txt 2>/dev/null; \
    python3 - "$name" <<'PY'
import sys,ast
name=sys.argv[1]
rows=[l for l in open(f"/tmp/ab_{name}.txt") if l[0].isdigit()]
rs=[ast.literal_eval(l.split(' ',1)[1]) for l in rows]
sc=[16-r['place'] for r in rs]; k=[r['kills'] for r in rs]; t=[r['t'] for r in rs]
print(f"{name:14s} score {sum(sc)/len(sc):.2f}  kills {sum(k)/len(k):.2f}  life {sum(t)/len(t):.0f}s  n={len(sc)}")
PY
  ) &
done
wait
