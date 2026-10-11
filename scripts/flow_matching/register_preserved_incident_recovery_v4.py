"""Retain all incident cases and register exact full-budget stopped Pi attempts."""
import argparse
import copy
import json

from scripts.flow_matching.run_light_controller import ROOT, digest


NONSCIENCE_KEYS = ('id', 'output_directory')


def scientific(job):
    return {k: v for k, v in job.items() if k not in NONSCIENCE_KEYS}


def validate_case(root, case):
    path = root / case['original_queue']
    if digest(path) != case['original_queue_sha256']:
        raise ValueError('Original full scientific queue changed')
    original = json.loads(path.read_text(encoding='utf-8'))
    if next(j for j in original['jobs'] if j['id'] == case['original_id']) != case['original_job']:
        raise ValueError('Original canonical job differs')
    if (scientific(case['job']) != scientific(case['original_job'])
            or case['settings'] != original.get('settings', {})
            or case['evaluation'] != original.get('evaluation')):
        raise ValueError('Only attempt ID and output may change; all original scientific budgets must remain')
    for entry in case['job']['frozen_files']:
        if digest(root / entry['path']) != entry['sha256']:
            raise ValueError('Frozen original model/data/implementation differs')


def build_cases(root, parent, original_pi, original_pi_path, state_root, extra_ids):
    cases = copy.deepcopy(parent['jobs'])
    renewed = []
    for case in cases:
        validate_case(root, case)
        output = root / case['job']['output_directory']
        if (output / 'result.json').exists() or not output.exists():
            continue
        failure_path = output / 'failure.json'
        if not failure_path.exists():
            raise ValueError('Existing incident partial must have an explicit preserved terminal failure')
        failure = json.loads(failure_path.read_text(encoding='utf-8'))
        if not failure.get('resources', {}).get('resource_guard_aborted'):
            raise ValueError('Only the original memory-guard-aborted incident attempts may receive a fresh attempt')
        old = copy.deepcopy(case['job'])
        case['job'] = copy.deepcopy(case['original_job'])
        case['job']['id'] += '__preserved_incident_recovery_v4'
        case['job']['output_directory'] = state_root + '/jobs/' + case['job']['id']
        case['superseded_preserved_attempt'] = dict(job=old, failure_path=failure_path.relative_to(root).as_posix(),
                                                  failure_sha256=digest(failure_path), failure_record=failure)
        renewed.append(case['original_id'])
    existing = {c['original_id'] for c in cases}
    for identity in extra_ids:
        if identity in existing:
            raise ValueError('A canonical seed may appear only once in this recovery queue')
        original = next(j for j in original_pi['jobs'] if j['id'] == identity)
        output = root / original['output_directory']
        if (output / 'result.json').exists() or not (output / 'checkpoints').is_dir():
            raise ValueError('Additional original Pi job must be stopped, incomplete and preserved')
        evidence = [dict(path=p.relative_to(root).as_posix(), sha256=digest(p), bytes=p.stat().st_size)
                    for p in sorted(output.rglob('*')) if p.is_file()]
        job = copy.deepcopy(original)
        job['id'] += '__preserved_incident_recovery_v4'
        job['output_directory'] = state_root + '/jobs/' + job['id']
        failure_path = output / 'failure.json'
        failure = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {
            'status': 'preserved_incomplete_checkpoint_without_final_result',
            'cause': 'Terminal cause is not inferred from an incomplete checkpoint; full fresh attempt required.'}
        cases.append(dict(kind='pi', original_queue=original_pi_path, original_queue_sha256=digest(root / original_pi_path),
            original_id=identity, original_job=original, job=job, settings=original_pi['settings'], evaluation=None,
            python=original_pi['settings']['python'], failure_record=failure, evidence=evidence))
    if len({c['original_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate canonical training seed')
    for case in cases:
        validate_case(root, case)
        if (root / case['job']['output_directory']).exists() and case['original_id'] in [*renewed, *extra_ids]:
            raise FileExistsError('Fresh attempt must not overwrite previous artifacts')
    return cases, renewed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = ROOT / args.config
    registration = json.loads(path.read_text(encoding='utf-8'))
    for entry in registration['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen full recovery registration source differs')
    parent = json.loads((ROOT / registration['parent_queue']).read_text(encoding='utf-8'))
    pi = json.loads((ROOT / registration['original_pi_queue']).read_text(encoding='utf-8'))
    cases, renewed = build_cases(ROOT, parent, pi, registration['original_pi_queue'],
                                 registration['state_root'], registration['extra_original_pi_ids'])
    if len(cases) != 13 or len(renewed) != 2:
        raise ValueError('All11 original incident cases plus2 stopped Pi seeds required')
    queue = dict(schema_version=4, settings=dict(registration['settings'], output_root=registration['state_root']),
        jobs=cases, frozen_files=registration['source_receipts'], parent_queue=registration['parent_queue'],
        parent_queue_sha256=digest(ROOT / registration['parent_queue']),
        boundary='All11 original incident cases retained plus2 stopped original Pi seeds.5 prior completions unchanged;2 memory-aborted recovery attempts receive fresh outputs;4 original pending attempts unchanged;2 additional original Pi attempts freshly registered. No extra independent seeds, altered model/data/budget/settings or test-based attempt selection. Original failures/checkpoints preserved; complete45paper/681method/273data-task scope remains unfinished.')
    target = ROOT / registration['output_queue']
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(queue, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(original_canonical_cases=len(cases), renewed_memory_aborted_attempts=renewed,
                          newly_registered_original_pi_cases=registration['extra_original_pi_ids'],
                          additional_training_seed_slots=0, queue_sha256=digest(target))))


if __name__ == '__main__':
    main()
