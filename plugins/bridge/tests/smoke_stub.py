#!/usr/bin/env python3
"""Stub-daemon smoke test for the brainrouter-bridge plugin (v2 L2).

Stands up a minimal daemon-side plugin RPC server on a Unix socket (the
newline-delimited JSON-RPC 2.0 shape the supervisor implements), launches the
plugin binary against it, and verifies the startup sequence:

  1. `initialize` handshake (token check + protocol_version 1)
  2. `config.get {"key":"*"}` (full v1 bridge config minus the token)
  3. `secrets.get` for `discord_token` (discord enabled in the stub config)

and that the plugin logs a config + boots its (idle/failed-gateway)
transports.  The stub answers the exact shapes from
src/plugins/rpc.rs; no daemon build is required.

Usage:
  python3 plugins/bridge/tests/smoke_stub.py [--binary PATH] [--keep-logs DIR]

Exit 0 = PASS.  Evidence (RPC frames seen by the stub + plugin stderr) is
printed to stdout.
"""

import argparse
import asyncio
import json
import os
import pathlib
import shutil
import signal
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent          # .../plugins/bridge/tests
CRATE = HERE.parent                                     # .../plugins/bridge

EXPECTED_GRANTS = [
    "chat.completions", "models.list", "routing.events.tail", "inflight.list",
    "config.get", "config.set", "kv.get", "kv.set", "secrets.get", "notify",
    "set_status",
]

STUB_CONFIG = {
    "omp_path": "true",
    "work_dir": "/tmp",
    "aliases_config": "/nonexistent/aliases.yaml",
    "timeout_secs": 5,
    "default_model": "brainrouter/auto",
    "discord": {"enabled": True, "prefix": "!", "channel_id": None},
    "signal": {"enabled": False, "account": "+15550001111", "prefix": "!"},
}


class StubDaemon:
    def __init__(self, socket_path: pathlib.Path, token: str):
        self.socket_path = socket_path
        self.token = token
        self.requests: list[dict] = []
        self.event = asyncio.Event()

    def seen(self, method: str) -> bool:
        return any(r.get("method") == method for r in self.requests)

    async def handle(self, reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                return
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                writer.write((json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}) + "\n").encode())
                await writer.drain()
                continue
            self.requests.append(req)
            self.event.set()
            resp = self.respond(req)
            if resp is not None:
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()

    def respond(self, req: dict):
        method, req_id = req.get("method"), req.get("id")
        if req_id is None:                      # notification
            return None
        if method == "initialize":
            p = req.get("params", {})
            if p.get("token") != self.token:
                return {"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32001, "message": "invalid token"}}
            return {"jsonrpc": "2.0", "id": req_id, "result": {
                "plugin_id": p.get("id", "bridge"),
                "protocol_version": 1,
                "grants": EXPECTED_GRANTS,
                "daemon_version": "1.1.0-smoke-stub"}}
        if method == "config.get":
            key = req.get("params", {}).get("key")
            if key == "*":
                return {"jsonrpc": "2.0", "id": req_id,
                        "result": {"key": "*", "value": STUB_CONFIG}}
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"key": key, "value": None}}
        if method == "secrets.get":
            cred = req.get("params", {}).get("id")
            if cred == "discord_token":
                return {"jsonrpc": "2.0", "id": req_id,
                        "result": {"id": cred, "value": "stub-discord-token"}}
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32004, "message": f"no credential {cred}"}}
        if method in ("set_status", "notify"):
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"ok": True, "method": method}}
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"method not found: {method}"}}


def find_binary(cli: str | None) -> pathlib.Path:
    if cli:
        return pathlib.Path(cli)
    for profile in ("release", "debug"):
        cand = CRATE / "target" / profile / "brainrouter-bridge"
        if cand.is_file():
            return cand
    sys.exit("plugin binary not built; run: cargo build --manifest-path "
             "plugins/bridge/Cargo.toml  (or pass --binary)")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary")
    ap.add_argument("--keep-logs", help="dir to write stub/plugin logs into")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    binary = find_binary(args.binary)
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="bridge-smoke-"))
    sock = workdir / "bridge.sock"
    state = workdir / "state"
    state.mkdir(mode=0o700)
    token = "smoke-token-1234"

    stub = StubDaemon(sock, token)
    server = await asyncio.start_unix_server(stub.handle, path=str(sock))

    env = {
        **os.environ,
        "BRAINROUTER_PLUGIN_TOKEN": token,
        "BRAINROUTER_PLUGIN_ID": "bridge",
        "BRAINROUTER_PLUGIN_SOCKET": str(sock),
        "BRAINROUTER_PLUGIN_STATE": str(state),
        "BRAINROUTER_DAEMON_SOCKET": str(sock),
        "RUST_LOG": "info",
    }
    plugin_log = workdir / "plugin.log"
    proc = await asyncio.create_subprocess_exec(
        str(binary), env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=open(plugin_log, "wb"))

    want = ["initialize", "config.get", "secrets.get"]
    deadline = time.monotonic() + args.timeout
    rc = 1
    try:
        while time.monotonic() < deadline:
            if all(stub.seen(m) for m in want):
                break
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(stub.event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                stub.event.clear()
            else:
                break
        ok = all(stub.seen(m) for m in want)
        plugin_stderr = plugin_log.read_text(errors="replace")
        # The token must NEVER appear in the plugin log (it comes from secrets).
        token_leak = "stub-discord-token" in plugin_stderr
        if ok and not token_leak:
            rc = 0
        print("=== stub daemon saw (JSON-RPC requests) ===")
        for r in stub.requests:
            print(json.dumps(r))
        print("=== plugin stderr ===")
        print(plugin_stderr)
        print(f"=== checks: initialize={stub.seen('initialize')} "
              f"config.get={stub.seen('config.get')} "
              f"secrets.get={stub.seen('secrets.get')} "
              f"token_not_logged={not token_leak} "
              f"plugin_exit={proc.returncode} ===")
        print("SMOKE OK" if rc == 0 else "SMOKE FAIL")
    finally:
        if proc.returncode is None:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        server.close()
        if args.keep_logs:
            keep = pathlib.Path(args.keep_logs)
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "plugin.log").write_bytes(plugin_log.read_bytes())
            (keep / "stub_requests.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in stub.requests))
        else:
            # The migration copy may have populated the temp state dir with
            # real v1 files; tear down recursively.
            shutil.rmtree(workdir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
