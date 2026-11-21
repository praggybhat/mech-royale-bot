# Mech Royale bot

A survival-first bot for the Builders Buildathon Mech Royale track (`mechfight/1` protocol).

- `bot.py`: decision logic (zone, combat, aiming with lead, dodging, cover, loot, upgrades)
- `world.py`: map model built from `/arena` (walkable grid, line of sight, A* pathfinding)
- `mechfight.py`: the kit's protocol wrapper
- `tools/sim.py`: rough local simulator for tuning (not the official game)

Check locally:

```sh
python3 check/check.py -- python3 -u bot.py
docker build -t my-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
```
