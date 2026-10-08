"""Verify/re-register the immutable numerical repair jobs from their model configs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    base = json.loads((ROOT / 'configs/experiments/fm_tsad_cpu_queue.v1.json').read_text(encoding='utf-8'))
    repairs = {}
    for path in sorted((ROOT / 'configs/models').glob('fm_objective_*_stable_v2.json')):
        config = json.loads(path.read_text(encoding='utf-8'))
        original_path = config['numerical_repair']['original_config']
        original = json.loads((ROOT / original_path).read_text(encoding='utf-8'))
        if config['parameters'] != original['parameters']:
            raise ValueError('Numerical repair changed the declared training settings')
        repairs[original_path] = path.relative_to(ROOT).as_posix()
    if len(repairs) != 3:
        raise ValueError('Expected SB, SF2M and flow-only SF2M numerical repairs')
    source = 'src/iia_benchmark/models/fm_objective_stable.py'
    jobs = []
    for old in base['jobs']:
        if old['model_config'] not in repairs:
            continue
        job = copy.deepcopy(old)
        job['id'] += '__stable_numerics_v2'
        job['supersedes_failed_numerics_job'] = old['id']
        job['model_config'] = repairs[old['model_config']]
        job['output_directory'] = 'experiments/runs/fm_tsad_entropic_repair_v1/jobs/' + job['id']
        job['frozen_files'] = [item for item in job['frozen_files'] if item['path'] != old['model_config']]
        job['frozen_files'].extend([{'path': job['model_config'], 'sha256': sha(ROOT / job['model_config'])},
                                    {'path': source, 'sha256': sha(ROOT / source)}])
        jobs.append(job)
    queue = {**{key: value for key, value in base.items() if key != 'jobs'},
             'lane': 'cpu_repair', 'state_root': 'experiments/runs/fm_tsad_entropic_repair_v1', 'jobs': jobs,
             'numerical_repair_scope': 'Supersedes entropic coupling failures only; does not alter other original job records'}
    target = ROOT / 'configs/experiments/fm_tsad_entropic_repair_queue.v1.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != queue:
            raise ValueError('Frozen numerical queue changed; use a new registration version')
    else:
        write_json(target, queue)
    print(json.dumps({'verified_jobs': len(jobs), 'queue_sha256': sha(target)}))


if __name__ == '__main__':
    main()
