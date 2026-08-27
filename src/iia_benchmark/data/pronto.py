from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile


def _unsafe_archive_member(info: zipfile.ZipInfo) -> str | None:
    """Return the reason a ZIP member cannot be safely materialized."""

    normalized = info.filename.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(info.filename)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return "absolute_or_drive_path"
    if any(part in {"", ".."} for part in posix.parts):
        return "empty_or_parent_path_component"
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        return "symbolic_link"
    if info.flag_bits & 0x1:
        return "encrypted_member"
    return None


def audit_pronto_archive(path: str | Path, *, verify_crc: bool = False) -> dict:
    """Inventory a PRONTO ZIP before extraction.

    The inventory is intentionally independent of PRONTO's internal folder
    names.  It rejects path traversal, drive paths, symbolic links, and
    encrypted members before any extraction is allowed.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffixes: Counter[str] = Counter()
    unsafe: list[dict[str, str]] = []
    total_compressed = 0
    total_uncompressed = 0
    file_count = 0
    directory_count = 0
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        for info in infos:
            reason = _unsafe_archive_member(info)
            if reason:
                unsafe.append({"path": info.filename, "reason": reason})
            if info.is_dir():
                directory_count += 1
                continue
            file_count += 1
            suffixes[PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() or "<none>"] += 1
            total_compressed += info.compress_size
            total_uncompressed += info.file_size
        crc_failure = archive.testzip() if verify_crc and not unsafe else None
    return {
        "archive": source.as_posix(),
        "members": len(infos),
        "files": file_count,
        "directories": directory_count,
        "compressed_bytes": total_compressed,
        "uncompressed_bytes": total_uncompressed,
        "compression_ratio": (
            total_uncompressed / total_compressed if total_compressed else 0.0
        ),
        "suffix_counts": dict(sorted(suffixes.items())),
        "unsafe_members": unsafe,
        "safe_to_extract": not unsafe,
        "crc_verified": bool(verify_crc and not unsafe and crc_failure is None),
        "crc_failure": crc_failure,
    }

