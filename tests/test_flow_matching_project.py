import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "projects/flow_matching_research/tools/check_materials.py"
SPEC = importlib.util.spec_from_file_location("fm_project_materials", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_material_verification_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        MODULE.inspect(tmp_path, {"path": "../outside.csv"})


@pytest.mark.parametrize("content,expected", [(b"version https://git-lfs.github.com/spec/v1\n", "lfs_pointer"), (b"<html>download failed</html>", "html_instead_of_payload")])
def test_unusable_downloads_are_not_counted(tmp_path, content, expected):
    (tmp_path / "data.csv").write_bytes(content)
    assert MODULE.inspect(tmp_path, {"path": "data.csv", "format": "csv"}) == expected


def test_corrupted_registered_payload_is_detected_without_overwrite(tmp_path):
    target = tmp_path / "data.csv"
    target.write_bytes(b"changed raw data")
    assert MODULE.inspect(tmp_path, {"path": "data.csv", "sha256": "0" * 64}, True) == "checksum_mismatch"
    assert target.read_bytes() == b"changed raw data"


def test_missing_and_wrong_size_are_distinct(tmp_path):
    assert MODULE.inspect(tmp_path, {"path": "data.csv"}) == "missing"
    (tmp_path / "data.csv").write_bytes(b"x")
    assert MODULE.inspect(tmp_path, {"path": "data.csv", "bytes": 2}) == "invalid_size"
