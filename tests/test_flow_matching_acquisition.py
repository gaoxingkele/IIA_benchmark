"""Raw-file preservation and integrity checks for the acquisition bundle."""
import hashlib
from pathlib import Path
import zipfile

import pytest

pytest.importorskip("requests")
pytest.importorskip("pypdf")

from scripts.data_acquisition import download_flow_matching_bundle as bundle


def test_interrupted_queue_retries_sources_without_records(tmp_path, monkeypatch):
    from scripts.data_acquisition import continue_flow_matching_downloads as continuation
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    (tmp_path / "complete.csv").write_bytes(b"valid preserved file")
    sources = [{"id": "complete", "path": "complete.csv"},
               {"id": "interrupted", "path": "missing.csv"},
               {"id": "gate", "path": "private.zip"},
               {"id": "lost", "path": "lost.csv"}]
    records = {"complete": {"status": "available"}, "gate": {"status": "quota_exceeded"},
               "lost": {"status": "available"}}
    assert [s["id"] for s in continuation.pending_sources(sources, records)] == ["interrupted", "lost"]


def test_cffi_range_quota_preserves_transport_prefix(tmp_path, monkeypatch):
    pytest.importorskip("curl_cffi")
    from scripts.data_acquisition import download_flow_matching_alternatives as alternative
    target = tmp_path / "range.segment"
    target.write_bytes(b"preserved prefix")
    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        def raise_for_status(self): pass
        def iter_content(self, size):
            yield b"<html><title>Google Drive - Quota exceeded</title></html>"
        def close(self): pass
    class Session:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, *args, **kwargs): return Response()
    monkeypatch.setattr(alternative.cffi, "Session", Session)
    with pytest.raises(alternative.PublicAccessGate, match="quota exceeded"):
        alternative.cffi_transfer("https://example.invalid/public", target, None, (16, 31))
    assert target.read_bytes() == b"preserved prefix"


def test_interrupted_cffi_ranges_resume_received_prefixes(tmp_path, monkeypatch):
    pytest.importorskip("curl_cffi")
    from scripts.data_acquisition import download_flow_matching_alternatives as alternative
    payload = b"a,b\n1,2\n3,4\n"
    starts = []
    class Response:
        status_code = 206
        def __init__(self, start, end):
            self.start, self.end = start, end
            self.headers = {"content-range": f"bytes {start}-{end}/{len(payload)}"}
        def raise_for_status(self): pass
        def iter_content(self, size):
            yield payload[self.start:min(self.start + 2, self.end + 1)]
            raise alternative.cffi.RequestsError("interrupted transfer")
        def close(self): pass
    class Session:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, *, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            starts.append(start)
            return Response(start, end)
    monkeypatch.setattr(alternative.cffi, "Session", Session)
    target = tmp_path / "data.csv"
    alternative.segmented({"url": "https://example.invalid/raw", "format": "csv",
                           "size_bytes": len(payload), "range_parts": 1, "range_workers": 1,
                           "range_transport": "curl_cffi", "segment_bytes": 4,
                           "checksum": "sha256:" + hashlib.sha256(payload).hexdigest()}, target, None)
    assert target.read_bytes() == payload
    assert starts == list(range(0, len(payload), 2))


def test_alternate_ranges_resume_preserved_chunks_and_check_hash(tmp_path, monkeypatch):
    pytest.importorskip("curl_cffi")
    from scripts.data_acquisition import download_flow_matching_alternatives as alternative
    payload = b"a,b\n1,2\n3,4\n"
    target = tmp_path / "data.csv"
    folder = tmp_path / "data.csv.chunks"
    folder.mkdir()
    (folder / "0000.part").write_bytes(payload[:2])
    starts = []
    def fake_curl(url, output, proxy, byte_range=None):
        lo, hi = byte_range
        starts.append(lo)
        output.write_bytes(payload[lo:hi + 1])
        return 0, 206, {"content-range": f"bytes {lo}-{hi}/{len(payload)}"}, ""
    monkeypatch.setattr(alternative, "curl_transfer", fake_curl)
    item = {"url": "https://example.invalid/raw", "format": "csv", "size_bytes": len(payload),
            "range_parts": 2, "range_workers": 1,
            "checksum": "sha256:" + hashlib.sha256(payload).hexdigest()}
    alternative.segmented(item, target, None)
    assert target.read_bytes() == payload
    assert starts == [2, 6]


def test_wrong_alternate_range_never_appends_to_preserved_partial(tmp_path, monkeypatch):
    pytest.importorskip("curl_cffi")
    from scripts.data_acquisition import download_flow_matching_alternatives as alternative
    target = tmp_path / "data.csv"
    folder = tmp_path / "data.csv.chunks"
    folder.mkdir()
    partial = folder / "0000.part"
    partial.write_bytes(b"ab")
    def wrong_curl(url, output, proxy, byte_range=None):
        output.write_bytes(b"incorrect")
        return 0, 206, {"content-range": "bytes 0-8/9"}, ""
    monkeypatch.setattr(alternative, "curl_transfer", wrong_curl)
    monkeypatch.setattr(alternative.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="exact requested range"):
        alternative.segmented({"url": "https://example.invalid/raw", "size_bytes": 8,
                               "range_parts": 1}, target, None)
    assert partial.read_bytes() == b"ab"
    assert not target.exists()


def test_explicit_drive_quota_stops_alternate_retries(tmp_path, monkeypatch):
    pytest.importorskip("curl_cffi")
    from scripts.data_acquisition import download_flow_matching_alternatives as alternative
    monkeypatch.setattr(alternative, "ROOT", tmp_path)
    calls = []
    def quota(url, output, proxy, byte_range=None):
        calls.append(url)
        output.write_bytes(b"<!DOCTYPE html><html><title>Google Drive - Quota exceeded</title></html>")
        return 0, 200, {}, ""
    monkeypatch.setattr(alternative, "curl_transfer", quota)
    record = alternative.acquire({"id": "quota", "path": "data.zip", "format": "zip",
                                  "url": "https://example.invalid/public",
                                  "checksum": "md5:" + "0" * 32}, None)
    assert record["status"] == "quota_exceeded"
    assert len(calls) == 1
    assert not (tmp_path / "data.zip").exists()


def test_existing_raw_file_is_preserved_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    target = tmp_path / "raw.csv"
    target.write_bytes(b"time,value\n0,1\n")
    monkeypatch.setattr(bundle.requests, "get", lambda *a, **kw: pytest.fail("network called"))
    item = {"id": "raw", "path": "raw.csv", "url": "https://example.invalid/raw"}
    record = bundle.acquire(item, None)
    assert record["status"] == "available"
    assert record["backend"] == "existing_preserved"
    assert record["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_bad_existing_checksum_is_preserved_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    target = tmp_path / "raw.csv"
    target.write_bytes(b"original user data")
    record = bundle.acquire({"id": "raw", "path": "raw.csv", "url": "https://example.invalid/raw",
                             "checksum": "sha256:" + "0" * 64}, None)
    assert record["status"] == "failed"
    assert target.read_bytes() == b"original user data"


def test_html_does_not_count_as_dataset(tmp_path):
    target = tmp_path / "data.csv"
    target.write_bytes(b"<!DOCTYPE html><html>Login</html>")
    with pytest.raises(ValueError, match="HTML"):
        bundle.validate(target, {})


def test_zip_crc_and_members_are_audited(tmp_path):
    target = tmp_path / "data.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("data.csv", "a,b\n1,2")
    assert bundle.validate(target, {"format": "zip"}) == {"archive_members": 1}


def test_missing_restricted_payload_retains_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    record = bundle.acquire({"id": "private", "path": "private.csv", "url": None,
                             "access": "requires_author_permission"}, None)
    assert record["status"] == "requires_author_permission"
    assert not (tmp_path / "private.csv").exists()


def test_outside_workspace_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside repository"):
        bundle.acquire({"id": "escape", "path": "../escape.csv"}, None)


def test_ranged_download_resumes_chunks_and_verifies_final_hash(tmp_path, monkeypatch):
    payload = b"time,value\n0,1\n1,2\n"
    class Response:
        status_code = 206
        def __init__(self, lo, hi):
            self.headers = {"Content-Range": f"bytes {lo}-{hi}/{len(payload)}"}
            self.body = payload[lo:hi + 1]
        def raise_for_status(self):
            pass
        def iter_content(self, size):
            yield self.body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    requests_seen = []
    class Session:
        def get(self, url, *, headers, **kwargs):
            lo, hi = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            requests_seen.append((lo, hi))
            return Response(lo, hi)
    monkeypatch.setattr(bundle, "http_session", lambda: Session())
    target = tmp_path / "raw.csv"
    chunks = tmp_path / "raw.csv.chunks"
    chunks.mkdir()
    (chunks / "0000.part").write_bytes(payload[:2])
    item = {"id": "range", "url": "https://example.invalid/file", "size_bytes": len(payload),
            "range_parts": 3, "range_workers": 2,
            "checksum": "sha256:" + hashlib.sha256(payload).hexdigest()}
    bundle.ranged_download(item, target, None)
    assert target.read_bytes() == payload
    assert min(lo for lo, hi in requests_seen) == 2
    assert len(list(chunks.glob("*.bin"))) == 3


def test_numpy_response_requires_numpy_magic(tmp_path):
    target = tmp_path / "data.npy"
    target.write_bytes(b'{"error":"quota"}')
    with pytest.raises(ValueError, match="signature"):
        bundle.validate(target, {"format": "npy"})


def test_lfs_pointer_cannot_be_registered_as_successful_npz(tmp_path):
    target = tmp_path / "data.npz"
    target.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 5000\n")
    with pytest.raises(ValueError, match="Git LFS"):
        bundle.validate(target, {"format": "npz"})


def test_existing_directory_has_file_checksums(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "ROOT", tmp_path)
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "raw.csv").write_bytes(b"a,b\n1,2")
    record = bundle.acquire({"id": "existing", "path": "data", "format": "directory"}, None)
    assert record["status"] == "available"
    assert record["files"][0]["sha256"] == hashlib.sha256(b"a,b\n1,2").hexdigest()
