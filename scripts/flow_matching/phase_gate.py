"""Validate original-phase completion before industrial transfer can start."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_queue(queue_path, root=ROOT):
    settings = json.loads(Path(queue_path).read_text(encoding='utf-8'))
    pending = []
    for job in settings['jobs']:
        cfg_path = root / job['config']
        if hashlib.sha256(cfg_path.read_bytes()).hexdigest() != job['config_sha256']:
            raise ValueError('Prerequisite configuration changed')
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        directory = root / cfg['output_root']
        result_path = directory / 'result.json'
        if not result_path.exists():
            pending.append(cfg['id'])
            continue
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('status') != 'completed' or result.get('config_sha256') != job['config_sha256']:
            raise ValueError('Invalid prerequisite result')
        if result.get('config', {}).get('diagnostic') or result.get('config', {}).get('limit_test_cases'):
            raise ValueError('Diagnostic result cannot satisfy original-phase completion')
    return pending
