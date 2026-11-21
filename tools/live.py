#!/usr/bin/env python3
"""Play one private match on the real engine (play.joinbuilders.org); empty seats are the server's house bots.

    python3 tools/live.py --out runs/live [--env MB_X=1 ...]

Creates a private room, joins as the only player, starts the match, and bridges the game socket to
`python3 -u bot.py` running as a subprocess. Prints one JSON result line: place, kills, our damage stats.
Standard library only. A test tool, not part of the submitted image.
"""
import argparse
import base64
import json
import os
import secrets
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.request

HOST = "play.joinbuilders.org"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WS:
    def __init__(self, path):
        ctx = ssl.create_default_context()
        self.s = ctx.wrap_socket(socket.create_connection((HOST, 443), timeout=20), server_hostname=HOST)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: {HOST}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: https://{HOST}\r\n"
                        f"User-Agent: mech-bot-tester\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            c = self.s.recv(4096)
            if not c:
                raise RuntimeError("handshake failed")
            buf += c
        head, self.buf = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError(head.decode(errors="replace")[:200])
        self.s.settimeout(None)
        self.lock = threading.Lock()

    def _need(self, n):
        while len(self.buf) < n:
            c = self.s.recv(65536)
            if not c:
                raise EOFError
            self.buf += c
        r, self.buf = self.buf[:n], self.buf[n:]
        return r

    def recv(self):
        msg = b""
        while True:
            h = self._need(2)
            fin, op, ln = h[0] & 0x80, h[0] & 0x0F, h[1] & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._need(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._need(8))[0]
            mask = self._need(4) if h[1] & 0x80 else None
            data = self._need(ln)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 8:
                raise EOFError
            if op == 9:
                self._send(10, data)
                continue
            if op == 10:
                continue
            msg += data
            if fin:
                return msg.decode(errors="replace")

    def _send(self, op, data):
        n = len(data)
        hdr = bytes([0x80 | op])
        hdr += bytes([0x80 | n]) if n < 126 else (bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536
                                                   else bytes([0x80 | 127]) + struct.pack(">Q", n))
        m = secrets.token_bytes(4)
        with self.lock:
            self.s.sendall(hdr + m + bytes(b ^ m[i % 4] for i, b in enumerate(data)))

    def send(self, obj):
        self._send(1, json.dumps(obj, separators=(",", ":")).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "live"))
    ap.add_argument("--name", default="tester")
    ap.add_argument("--env", nargs="*", default=[], help="MB_KEY=value overrides for the bot")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sid = secrets.token_hex(16)
    req = urllib.request.Request(f"https://{HOST}/mm/code/new?sid={sid}", headers={"User-Agent": "mech-bot-tester"})
    info = json.loads(urllib.request.urlopen(req, timeout=15).read())
    room, code = info["room"], info["code"]
    ws = WS(f"/r/{room}/play?token=&sid={sid}&name={a.name}&code={code}")
    tag = time.strftime("%H%M%S") + "-" + code
    log = open(os.path.join(a.out, tag + ".jsonl"), "w")
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    bot = None
    started = False
    stats = {"code": code, "hits": 0, "dealt": 0.0, "taken": 0.0, "t_end": None}
    last_t = [0.0]

    def pump(proc):
        for line in proc.stdout:
            line = line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("type") == "ready":
                continue
            ws.send(m)

    deadline = time.time() + a.timeout
    result = None
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except EOFError:
            break
        try:
            m = json.loads(raw)
        except ValueError:
            continue
        ty = m.get("type")
        if ty == "lobby":
            you = m.get("you", {})
            if not you.get("ready"):
                ws.send({"type": "ready", "name": a.name})
            if you.get("host") and m.get("phase") == "waiting" and not started:
                started = True
                time.sleep(0.5)
                ws.send({"type": "start"})
        elif ty == "hello":
            m["map"]["dir"] = os.path.join(ROOT, "arena")
            bot = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "bot.py")], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=open(os.path.join(a.out, tag + ".err"), "w"),
                                   env=env)
            threading.Thread(target=pump, args=(bot,), daemon=True).start()
            bot.stdin.write((json.dumps(m) + "\n").encode())
            bot.stdin.flush()
            ws.send({"type": "loaded"})
            log.write(json.dumps({"hello_slot": m.get("slot")}) + "\n")
        elif ty == "obs":
            last_t[0] = m.get("t", 0)
            for e in m.get("events", []):
                if e["type"] == "hit" and e.get("target", 999) < 16:
                    stats["hits"] += 1
                    stats["dealt"] += e.get("amount", 0)
                elif e["type"] == "damaged":
                    stats["taken"] += e.get("amount", 0)
            if m["tick"] % 20 == 0:
                me = m["self"]
                log.write(json.dumps({"t": m["t"], "pos": me["pos"], "hp": me["hp"], "alive": m["alive"],
                                      "w": [w["kind"] if w else None for w in me["weapons"]],
                                      "cubes": me["cubes"], "seen": len(m["mechs"])}) + "\n")
            if bot:
                try:
                    bot.stdin.write((raw + "\n").encode())
                    bot.stdin.flush()
                except (BrokenPipeError, ValueError):
                    pass
        elif ty == "end":
            result = {"place": m.get("place"), "kills": m.get("kills"), "t": last_t[0]}
            log.write(json.dumps({"end": m}) + "\n")
            if bot:
                try:
                    bot.stdin.write((raw + "\n").encode())
                    bot.stdin.close()
                except (BrokenPipeError, ValueError):
                    pass
            break
        else:
            log.write(json.dumps({"msg": ty}) + "\n")
    try:
        ws.s.close()
    except OSError:
        pass
    if bot and bot.poll() is None:
        bot.kill()
    out = dict(stats)
    if result:
        out.update(result)
    out["dealt"] = round(out["dealt"])
    out["taken"] = round(out["taken"])
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
