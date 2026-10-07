"""Validate logical repository paths through explicitly configured data mounts.

Keep experiment and acquisition registrations portable and byte-stable. A local
directory junction redirects their logical data paths to the configured disk.
Arbitrary symlinks or traversal do not become authorized by enabling a mount.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
import sys


STORAGE_CONFIG = 'configs/storage/data_storage.v1.json'


def _resolved(path: Path) -> Path:
    value = path.resolve()
    spelling = str(value)
    if sys.platform == 'win32' and spelling.startswith('\\\\?\\'):
        spelling = spelling[4:]
        if spelling.startswith('UNC\\'):
            spelling = '\\\\' + spelling[4:]
        value = Path(spelling)
    return value


def mounts(root: Path):
    configuration = root / STORAGE_CONFIG
    if not configuration.exists():
        return []
    config = json.loads(configuration.read_text(encoding='utf-8'))
    result = []
    for entry in config.get('mounts', []):
        if entry.get('platform') and entry['platform'] != sys.platform:
            continue
        relative = Path(entry['logical_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Storage mount must use a safe repository-relative path')
        physical = Path(entry['physical_path'])
        if not physical.is_absolute():
            raise ValueError('Storage mount physical path must be absolute')
        result.append((relative, _resolved(physical)))
    return result


def resolve_project_path(root: Path, value: str | Path, *, relative_only=False) -> Path:
    root = _resolved(root)
    path = Path(value)
    if relative_only and (path.is_absolute() or PureWindowsPath(str(value)).is_absolute()):
        raise ValueError('Registered paths must be relative to the workspace')
    # absolute() alone retains '..'; normalize lexically before containment.
    logical = Path(os.path.abspath(root / path))
    configured = mounts(root)
    if not logical.is_relative_to(root):
        for relative, physical in configured:
            if logical.is_relative_to(physical):
                logical = root / relative / logical.relative_to(physical)
                break
    if not logical.is_relative_to(root) or logical == root:
        raise ValueError('Registered path escapes the workspace')
    resolved = _resolved(logical)
    if resolved.is_relative_to(root):
        return logical
    for relative, physical in configured:
        logical_mount = root / relative
        if logical.is_relative_to(logical_mount) and _resolved(logical_mount) == physical:
            if resolved.is_relative_to(physical):
                return logical
    raise ValueError('Registered path escapes the workspace or configured data storage')


def project_relative_path(root: Path, value: str | Path) -> str:
    return resolve_project_path(root, value).relative_to(_resolved(root)).as_posix()


def public_dataset_root(root: Path) -> Path:
    return _resolved(resolve_project_path(root, 'data/public_datasets'))
