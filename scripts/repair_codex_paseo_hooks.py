"""Inspect or disable obsolete user Paseo hooks through the running Codex server.

Requires websocket-client for the local WebSocket transport. Inspection is the
default; --apply backs up local configuration and hot-reloads only enabled flags.
No hook trust, notification, conversation, or worker process is changed.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import time
import tomllib


EVENTS = {"preToolUse", "postToolUse", "permissionRequest", "userPromptSubmit", "stop"}


def paseo_hooks(data: list[dict], home: Path) -> list[dict]:
    """Select only this user's known Paseo handlers, never project/plugin policy."""
    source = os.path.normcase(os.path.abspath(home / "hooks.json"))
    selected: dict[str, dict] = {}
    for entry in data:
        for hook in entry.get("hooks", []):
            if hook.get("source") != "user" or hook.get("isManaged") or hook.get("pluginId"):
                continue
            path = os.path.normcase(os.path.abspath(hook.get("sourcePath", "")))
            command = hook.get("command") or ""
            known = "paseo-codex-windows.ps1" in command or (
                "PASEO_TERMINAL_ID" in command and "hooks codex" in command
            )
            if path == source and hook.get("eventName") in EVENTS and known:
                selected[hook["key"]] = hook
    return list(selected.values())


def disable_edits(hooks: list[dict], state: dict) -> list[dict]:
    edits = []
    for hook in hooks:
        if not hook["enabled"]:
            continue
        # Replace just this handler's state, preserving its existing trust hash.
        # Quoting the key with TOML syntax avoids splitting Windows paths on dots.
        value = dict(state.get(hook["key"], {}))
        value["enabled"] = False
        edits.append({"keyPath": "hooks.state." + json.dumps(hook["key"]),
                      "value": value, "mergeStrategy": "replace"})
    return edits


class LocalServer:
    def __init__(self, path: Path):
        import websocket

        # Python on Windows may omit AF_UNIX even though Winsock supports it.
        sock = socket.socket(getattr(socket, "AF_UNIX", 1), socket.SOCK_STREAM)
        try:
            if os.name == "nt":
                class Address(ctypes.Structure):
                    _fields_ = [("family", ctypes.c_ushort), ("path", ctypes.c_char * 108)]

                encoded = os.fsencode(path)
                if len(encoded) >= 108:
                    raise ValueError("Control socket path exceeds sockaddr_un capacity")
                addr = Address(1, encoded)
                lib = ctypes.WinDLL("Ws2_32.dll")
                lib.connect.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
                lib.connect.restype = ctypes.c_int
                sock.setblocking(False)
                result = lib.connect(sock.fileno(), ctypes.byref(addr), ctypes.sizeof(addr))
                if result != 0:
                    error = lib.WSAGetLastError()
                    if error != 10035:
                        raise OSError(error, "Connecting to Codex control socket failed")
                    import select
                    _, writable, errors = select.select([], [sock], [sock], 15)
                    if errors or not writable or sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                        raise TimeoutError("Codex control socket connection failed")
                sock.setblocking(True)
            else:
                sock.settimeout(15)
                sock.connect(str(path))
            # The control socket carries WebSocket frames, not JSONL. The stdio
            # proxy forwards raw bytes and also needs this HTTP Upgrade handshake.
            self.ws = websocket.create_connection("ws://localhost", socket=sock, timeout=20)
        except BaseException:
            sock.close()
            raise
        self.seq = 0
        self.notifications: list[dict] = []
        self.call("initialize", {"clientInfo": {"name": "paseo-hook-repair", "version": "1"},
                                 "capabilities": {"experimentalApi": True}})
        self.ws.send(json.dumps({"method": "initialized", "params": {}}))

    def call(self, method: str, params: dict) -> dict:
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params}))
        deadline = time.monotonic() + 20
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{method}: response deadline exceeded")
            self.ws.settimeout(remaining)
            response = json.loads(self.ws.recv())
            if response.get("id") != self.seq:
                self.notifications.append(response)
                continue
            if "error" in response:
                raise RuntimeError(f"{method}: {response['error']}")
            return response["result"]

    def close(self):
        self.ws.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-home", type=Path,
                        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")))
    parser.add_argument("--cwd", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply and os.environ.get("PASEO_TERMINAL_ID"):
        raise RuntimeError("This is an active Paseo session; its notifications are still needed")
    home = args.codex_home.resolve()
    config = home / "config.toml"
    server = LocalServer(home / "app-server-control" / "app-server-control.sock")
    try:
        cwds = args.cwd or [str(Path.cwd())]
        before = server.call("hooks/list", {"cwds": cwds})
        if any(d.get("errors") for d in before["data"]):
            raise RuntimeError("Hook discovery contains configuration errors; refusing to write")
        hooks = paseo_hooks(before["data"], home)
        raw_config = config.read_bytes()
        state = tomllib.loads(raw_config.decode("utf-8-sig")).get("hooks", {}).get("state", {})
        edits = disable_edits(hooks, state)
        report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "cwds": cwds, "selected_hooks": hooks, "apply": args.apply,
                  "edits": edits, "hook_discovery_errors": [],
                  "trust_hashes_changed": False, "hooks_json_changed": False}
        if args.apply and edits:
            effective = server.call("config/read", {"includeLayers": True})
            user_layers = [layer for layer in effective.get("layers", [])
                           if layer.get("name", {}).get("type") == "user"]
            if len(user_layers) != 1:
                raise RuntimeError("Cannot identify the user config version; refusing to write")
            backup = home / "hook-repairs" / ("disable-paseo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            backup.mkdir(parents=True, exist_ok=False)
            shutil.copy2(config, backup / "config.toml.before")
            shutil.copy2(home / "hooks.json", backup / "hooks.json.before")
            if config.read_bytes() != raw_config:
                raise RuntimeError("Configuration changed concurrently; rerun inspection")
            report["write_result"] = server.call("config/batchWrite", {
                "edits": edits, "reloadUserConfig": True,
                "expectedVersion": user_layers[0]["version"]})
            after = server.call("hooks/list", {"cwds": cwds})
            verified = paseo_hooks(after["data"], home)
            if any(h["enabled"] for h in verified) or {h["key"] for h in verified} != {h["key"] for h in hooks}:
                raise RuntimeError(f"Hook disable verification failed; backup is {backup}")
            new_state = tomllib.loads(config.read_text(encoding="utf-8-sig")).get("hooks", {}).get("state", {})
            assert all(new_state[h["key"]].get("trusted_hash") == state.get(h["key"], {}).get("trusted_hash") for h in hooks)
            assert (home / "hooks.json").read_bytes() == (backup / "hooks.json.before").read_bytes()
            report["after"] = after
            report["backup"] = str(backup)
            (backup / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        server.close()


if __name__ == "__main__":
    main()
