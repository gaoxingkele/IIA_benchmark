"""Register the exact original TimesNet SMD seed blocked by a preserved empty folder."""
import argparse
import copy
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest


def build_case(root, queue_path, original, identity, state_root):
    job = next(j for j in original['jobs'] if j['id'] == identity)
    output = root / job['output_directory']
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError('Original empty output evidence differs')
    replacement = copy.deepcopy(job)
    replacement['id'] += '__empty_output_recovery_v1'
    replacement['output_directory'] = state_root + '/jobs/' + replacement['id']
    if (root / replacement['output_directory']).exists():
        raise FileExistsError('Prior exact recovery output must remain intact')
    for entry in job['frozen_files']:
        if digest(root / entry['path']) != entry['sha256']:
            raise ValueError('Original frozen full source differs')
    return dict(kind='strict', original_queue=queue_path, original_queue_sha256=digest(root / queue_path),
        original_id=identity, original_job=job, job=replacement, settings=original.get('settings', {}),
        evaluation=original.get('evaluation'), python=original['python'],
        failure_record=dict(status='empty_output_blocked_before_training'),
        evidence=dict(original_empty_output_directory=job['output_directory'], old_directory_preserved=True,
            cause='Original worker rejects an existing directory. Empty directory does not prove training, OOM, or a completed seed.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    registration_path = ROOT / args.config
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    queue_path = ROOT / registration['original_queue']
    if digest(queue_path) != registration['original_queue_sha256']:
        raise ValueError('Original strict full queue differs')
    original = json.loads(queue_path.read_text(encoding='utf-8'))
    cases = [build_case(ROOT, registration['original_queue'], original, identity, registration['state_root'])
             for identity in registration['original_ids']]
    sources = [registration_path, Path(__file__), ROOT/'tests/test_fm_strict_empty_output_recovery_v1.py',
        ROOT/'scripts/flow_matching/run_resource_recovery.py',
        ROOT/'scripts/flow_matching/run_resource_recovery_job.py',
        ROOT/'scripts/flow_matching/run_tsad_experiment.py']
    recovery = dict(schema_version=3, settings=dict(registration['settings'],output_root=registration['state_root']),
        jobs=cases, frozen_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in sources],
        boundary='Exact original registered full strict TimesNet SMD seed1106. Only attempt ID/output differ; full model/data/evaluation/seed/budget/source stay identical. Original empty folder is preserved. This is not a new independent seed or an author-equivalence claim.')
    output = ROOT / registration['output_queue']
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(recovery,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(full_original_jobs=len(cases),queue=registration['output_queue'],sha256=digest(output))))


if __name__ == '__main__':
    main()
