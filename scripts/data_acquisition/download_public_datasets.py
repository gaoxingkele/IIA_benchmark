"""Download registered public datasets with aria2 and auditable checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "datasets" / "public_sources.json"
PROXY = "http://127.0.0.1:17890"
ARIA_CANDIDATES = (
    Path(r"C:\Users\10175\AppData\Local\aria2\aria2-1.37.0-win-64bit-build1\aria2c.exe"),
    Path(r"C:\Users\10175\AppData\Local\aria2c.exe"),
)


def load_sources() -> list[dict[str, Any]]:
    with REGISTRY.open("r", encoding="utf-8") as stream:
        return json.load(stream)["sources"]


def find_aria2() -> str | None:
    executable = shutil.which("aria2c")
    if executable:
        return executable
    for candidate in ARIA_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_ok(path: Path, expected: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not expected:
        return True
    algorithm, value = expected.split(":", 1)
    return checksum(path, algorithm) == value


def download_file(source: dict[str, Any], *, proxy: str | None = PROXY) -> None:
    target = ROOT / source["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if checksum_ok(target, source.get("checksum")):
        print(f"exists {source['id']}: {target.relative_to(ROOT)}")
        return
    aria2 = find_aria2()
    if aria2:
        command = [
            aria2,
            "--split=16",
            "--max-connection-per-server=16",
            "--min-split-size=1M",
            "--continue=true",
            "--file-allocation=none",
            "--max-tries=5",
            "--retry-wait=5",
            "--timeout=60",
            "--connect-timeout=30",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--content-disposition=false",
            "-d",
            str(target.parent),
            "-o",
            target.name,
        ]
        if proxy:
            command.append(f"--all-proxy={proxy}")
        command.append(source["url"])
        print(f"download {source['id']} with aria2")
        subprocess.run(command, check=True)
    else:
        print(f"download {source['id']} with urllib (aria2 unavailable)")
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            if proxy
            else urllib.request.ProxyHandler({})
        )
        with opener.open(source["url"]) as response, target.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if not checksum_ok(target, source.get("checksum")):
        raise RuntimeError(f"checksum failed for {source['id']}: {target}")


def clone_git(source: dict[str, Any], *, proxy: str | None = PROXY) -> None:
    target = ROOT / source["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target / ".git").exists():
        print(f"exists {source['id']}: {target.relative_to(ROOT)}")
        return
    print(f"clone {source['id']}: {source['url']}")
    command = ["git"]
    if proxy:
        command.extend(["-c", f"http.proxy={proxy}"])
    command.extend(["clone", "--depth", "1", source["url"], str(target)])
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="Source id; repeatable")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass the configured proxy (useful after a measured connectivity check).",
    )
    parser.add_argument("--round", type=int, choices=(1, 2, 3), help="Only fetch one expansion round")
    args = parser.parse_args()
    sources = load_sources()
    requested = set(args.dataset)
    known = {source["id"] for source in sources}
    if requested - known:
        print(f"unknown dataset ids: {', '.join(sorted(requested - known))}", file=sys.stderr)
        return 2
    selected = []
    for source in sources:
        if requested and source["id"] not in requested:
            continue
        if not requested and not source.get("default", False):
            continue
        if args.round and source.get("round") != args.round:
            continue
        if source.get("large") and not args.include_large and source["id"] not in requested:
            print(f"skip {source['id']}: large; pass --include-large or --dataset")
            continue
        selected.append(source)
    proxy = None if args.direct else PROXY
    for source in selected:
        if source["kind"] == "file":
            download_file(source, proxy=proxy)
        elif source["kind"] == "git":
            clone_git(source, proxy=proxy)
        else:
            raise ValueError(f"unsupported source kind: {source['kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
