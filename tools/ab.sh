#!/bin/bash
# usage: tools/ab.sh MATCHES SEED "name|ENV=1 ENV2=2" "name2|..." ...
N=$1; SEED=$2; shift 2
for spec in "$@"; do
  name=${spec%%|*}; envs=${spec#*|}
  ( env $envs python3 tools/sim.py --matches $N --seed $SEED > /tmp/ab_$name.txt 2>/tmp/ab_$name.err; \
    python3 tools/abstat.py "$name" ) &
done
wait
