"""Bind exact-budget incident attempts to original jobs without adding seeds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verified_attempts(root, queue_paths):
    from scripts.flow_matching.run_tsad_queue import complete_result
    from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts
    attempts = []
    cache = {}
    def pinned(path, expected):
        actual = cache.setdefault(path, None)
        if actual is None:
            actual = cache[path] = sha(root / path)
        if actual != expected:
            raise ValueError('Recovery source changed: '+path)
    for queue_path in queue_paths:
        queue = json.loads((root/queue_path).read_text(encoding='utf-8'))
        for case in queue['jobs']:
            original, job = case['original_job'], case['job']
            pinned(case['original_queue'], case['original_queue_sha256'])
            parent = json.loads((root/case['original_queue']).read_text(encoding='utf-8'))
            if next(j for j in parent['jobs'] if j['id']==case['original_id']) != original:
                raise ValueError('Recovery original job is not in its frozen queue')
            if {k:v for k,v in job.items() if k not in ('id','output_directory')} != {
                k:v for k,v in original.items() if k not in ('id','output_directory')}:
                raise ValueError('Recovery changed the original experiment')
            if case.get('evaluation') != parent.get('evaluation') or case.get('settings') != parent.get('settings', {}):
                raise ValueError('Recovery changed evaluation or author settings')
            path = root/job['output_directory']/'result.json'
            entry = {'queue':queue_path,'original_id':case['original_id'],'original_job':original,
                     'recovery_job':job,'kind':case['kind'],'status':'not_completed',
                     'selection_policy':'Original completed attempt preferred; otherwise first valid completed recovery in registered queue order. No metric-based selection or extra independent seed.'}
            if path.exists():
                for source in job['frozen_files']:
                    pinned(source['path'], source['sha256'])
                receipt = json.loads(path.read_text(encoding='utf-8'))
                if receipt.get('experiment_sha256') != hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest():
                    raise ValueError('Recovery receipt belongs to another job')
                if case['kind'] in ('strict','industrial'):
                    if not complete_result(root,job): raise ValueError('Recovery result is incomplete')
                else:
                    verify_artifacts(receipt,root)
                entry.update(status='completed',result_path=path.relative_to(root).as_posix(),result_sha256=sha(path))
            attempts.append(entry)
    return attempts


def resolve(original, attempts):
    for attempt in attempts:
        if attempt['original_id']==original['id'] and attempt['status']=='completed':
            if attempt['original_job'] != original:
                raise ValueError('Recovery was bound to a different original experiment')
            return attempt
    return None
