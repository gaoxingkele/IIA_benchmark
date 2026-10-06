"""Wait for active queues, retry transient failures, and refresh the audit report."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "papers/literature/flow_matching"
RUNTIME = ROOT / "data/public_datasets/flow_matching/acquisition_runtime"


def running(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def retry_queue(queue):
    while any(running(pid) for pid in [queue["pid"], *queue.get("wait_pids", [])]):
        time.sleep(5)
    original = ROOT / queue["manifest"]
    registry = json.loads((ROOT / queue["registry"]).read_text(encoding="utf-8"))
    sources = registry["sources"]
    if queue.get("download_script", "").endswith("download_flow_matching_alternatives.py"):
        large = "--only-large" in queue.get("download_arguments", [])
        sources = [s for s in sources if (s.get("backend") == "range_curl") == large]
    known = {}
    for attempt in range(1, 3):
        manifest = json.loads(original.read_text(encoding="utf-8"))
        known.update({r["id"]: r for r in manifest.get("records", [])})
        selected = pending_sources(sources, known)
        if not selected:
            break
        retry_registry = RUNTIME / f"{queue['name']}_retry_{attempt}.json"
        retry_registry.write_text(json.dumps({"sources": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
        original = BASE / f"{queue['name']}_automatic_retry_{attempt}_manifest.json"
        command = [sys.executable, queue.get("download_script", "scripts/data_acquisition/download_flow_matching_bundle.py"),
                   "--registry", str(retry_registry), "--manifest", str(original), "--workers", "3",
                   *queue.get("download_arguments", [])]
        with (RUNTIME / f"{queue['name']}_retry_{attempt}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if original.exists():
        known.update({r["id"]: r for r in json.loads(original.read_text(encoding="utf-8")).get("records", [])})
    pending = pending_sources(sources, known)
    gated = [s["id"] for s in sources if known.get(s["id"], {}).get("status") not in (None, "available", "failed")]
    return {"status": "incomplete" if pending or gated else "complete",
            "pending_ids": [s["id"] for s in pending], "gated_ids": gated}


def pending_sources(sources, records):
    """Recover interrupted queues whose submitted files never produced records."""
    return [s for s in sources if
            records.get(s["id"], {}).get("status") in (None, "failed") or
            (records.get(s["id"], {}).get("status") == "available" and not (ROOT / s["path"]).exists())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queues", type=Path, required=True, help="Runtime JSON: name, pid, registry, manifest")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This continuation runner checks Windows process handles.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    queues = json.loads(args.queues.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = {pool.submit(retry_queue, q): q["name"] for q in queues}
        while True:
            state = {name: "running" if not f.done() else
                     ("failed: " + str(f.exception()) if f.exception() else f.result())
                     for f, name in futures.items()}
            snapshot = {"updated_at": datetime.now(timezone.utc).isoformat(), "queues": state}
            (RUNTIME / f"{args.queues.stem}_status.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            subprocess.run([sys.executable, "scripts/data_acquisition/report_flow_matching_bundle.py"],
                           cwd=ROOT, stdout=subprocess.DEVNULL, check=False)
            if all(f.done() for f in futures):
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
