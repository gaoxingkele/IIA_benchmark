"""Integrity tests for registered Git/LFS snapshots, using real file bytes."""
import hashlib
from pathlib import Path

import pytest

from scripts.flow_matching.verify_registered_snapshot import audit_registry, verify_source, write_report


def source(data, path="data/item.bin", lfs=False):
    result = {"id": "item", "path": path, "size_bytes": len(data), "author_revision": "fixed"}
    if lfs:
        result["checksum"] = {"algorithm": "sha256", "value": hashlib.sha256(data).hexdigest()}
    else:
        result["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return result


def land(root, item, data):
    target = root / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


@pytest.mark.parametrize("lfs", [False, True])
def test_publisher_hash_and_local_sha256(tmp_path, lfs):
    data = b"scientific dataset\n"
    item = source(data, lfs=lfs)
    target = land(tmp_path, item, data)
    checked = verify_source(item, tmp_path)
    assert checked["status"] == "verified"
    assert checked["sha256"] == hashlib.sha256(data).hexdigest()
    target.write_bytes(b"X" * len(data))
    assert verify_source(item, tmp_path, checked)["status"] == "checksum_mismatch"


def test_git_blob_is_not_plain_file_sha1(tmp_path):
    data = b"same bytes, different hash protocol"
    item = source(data)
    land(tmp_path, item, data)
    item["git_blob_sha1"] = hashlib.sha1(data).hexdigest()
    assert verify_source(item, tmp_path)["status"] == "checksum_mismatch"


def test_full_sized_partial_and_lfs_pointer_not_complete(tmp_path):
    data = b"full preallocated bytes"
    item = source(data)
    land(tmp_path, {**item, "path": item["path"] + ".part"}, data)
    assert verify_source(item, tmp_path)["status"] == "partial"
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 42\n"
    item = source(pointer, lfs=True)
    land(tmp_path, item, pointer)
    assert verify_source(item, tmp_path)["status"] == "lfs_pointer"


def test_path_escape_rejected(tmp_path):
    for path in ("../outside", "C:/Windows/system.ini", "/outside"):
        assert verify_source(source(b"", path=path), tmp_path)["status"] == "invalid"


def test_report_never_overwrites_raw(tmp_path):
    data = b"preserved"
    item = source(data)
    target = land(tmp_path, item, data)
    with pytest.raises(ValueError, match="raw data"):
        write_report({}, target, tmp_path, [item])
    assert target.read_bytes() == data


def test_active_aria2_control_and_missing_hash_not_complete(tmp_path):
    data = b"downloaded but active"
    item = source(data)
    target = land(tmp_path, item, data)
    Path(str(target) + ".aria2").write_bytes(b"control")
    assert verify_source(item, tmp_path)["status"] == "partial"
    Path(str(target) + ".aria2").unlink()
    item.pop("git_blob_sha1")
    assert verify_source(item, tmp_path)["status"] == "unverifiable"


def test_alias_paths_cannot_inflate_complete_count(tmp_path):
    data = b"one physical file"
    item = source(data)
    land(tmp_path, item, data)
    alias = {**item, "id": "alias", "path": "data/../data/item.bin"}
    report = audit_registry({"source_count": 2, "sources": [item, alias]}, tmp_path)
    assert not report["summary"]["registry_valid"]
    assert not report["summary"]["complete"]


def test_complete_requires_registered_count_and_reuses_verified_bytes(tmp_path):
    data = b"dataset"
    item = source(data)
    land(tmp_path, item, data)
    config = {"sources": [item], "source_count": 1}
    report = audit_registry(config, tmp_path)
    assert report["summary"]["complete"]
    cached = audit_registry(config, tmp_path, report)
    assert cached["summary"]["reused_count"] == 1
    assert cached["records"][0]["sha256"] == report["records"][0]["sha256"]
    config["source_count"] = 2
    assert not audit_registry(config, tmp_path)["summary"]["complete"]
    config["source_count"] = 1
    config["total_bytes"] = len(data) + 1
    assert not audit_registry(config, tmp_path)["summary"]["complete"]
