"""Full original Diffusion-TS fMRI43 retry after actual500-step commit abort.

All70 paired generation jobs/14cohorts remain registered. Original author/data
citations and the frozen numerical worker are inherited from the original queue.
No saved optimizer/RNG state exists; repeat all15000 source updates with no credit
for partial progress. Only fresh paths and stronger admission change.
"""
import argparse
import copy
import inspect
import sys

from scripts.flow_matching import run_spectral_resource_order_v4 as ordering
from scripts.flow_matching.run_spectral_energy42_preserved_full_retry_v1 import selected_ast
from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha

IDENTITY = 'spectral_original__diffusion_ts__fmri__released_author_data__seed43'


def exact_retry(original, queue, config):
    expected = copy.deepcopy(original); expected['state_root'] = config['state_root']
    old = next(j for j in original['jobs'] if j['id'] == IDENTITY)
    job = next(j for j in expected['jobs'] if j['id'] == IDENTITY)
    job.update(output_directory=config['selected_output_directory'], artifact_directory=config['selected_artifact_directory'])
    if (len(original['jobs']) != 70 or len({j['id'] for j in queue['jobs']}) != 70
            or expected != queue or old['training_updates'] != 15000
            or old['gradient_accumulate_every'] != 2 or old['data_shape'] != [9977, 24, 50]
            or old['expected_parameters'] != 1406098 or old['generation_batch_size'] != 2001
            or old['generation_field_chunk_size'] is not None
            or old['output_directory'] == job['output_directory']
            or old['artifact_directory'] == job['artifact_directory']):
        raise ValueError('Retain all70 full jobs and exact original fMRI15000 recipe; fresh paths only')
    return old, job


def admission_settings(queue, job, registered_settings):
    settings = ordering.previous.admission_settings(queue, job, registered_settings)
    if job['id'] != IDENTITY:
        raise ValueError('Only the registered failed canonical fMRI43 slot may be dispatched')
    settings['minimum_commit_headroom_bytes'] = max(settings['minimum_commit_headroom_bytes'], 40 * 1024**3)
    return settings


def guarded_api():
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api = ordering.previous.original.guarded_api()
    api['resource_snapshot'] = resource_snapshot
    return api


def verify(config, check_live=True):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen full fMRI retry source changed: ' + pin['path'])
    previous = read(ROOT / config['original_resource_order_configuration'])
    original = ordering.verify_registration(previous)
    queue = read(ROOT / config['queue']); old, job = exact_retry(original, queue, config)
    incident = read(ROOT / config['original_resource_incident'])
    terminal_path = ROOT / config['original_terminal_receipt']
    terminal = read(terminal_path)
    progress = read(ROOT / old['output_directory'] / 'progress.json')
    if (incident['canonical_job'] != old or sha(terminal_path) != incident['original_terminal_receipt_sha256']
            or terminal['exitcode'] != 1 or terminal['resource_guard_aborted'] is not True
            or terminal['full_result_validated'] is not False
            or progress['optimizer_updates'] != 500 or progress['required_updates'] != 15000
            or config['credited_previous_optimizer_updates'] != 0
            or config['exact_optimizer_RNG_resume_available'] is not False
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
            or config['full_original_jobs'] != 70 or config['full_original_cohorts'] != 14
            or config['selected_admission_settings'] != admission_settings(original, old, original['settings'])):
        raise ValueError('Actual non-resumable500-step failure and stronger admission only required')
    from pathlib import Path
    for base in [ROOT / old['output_directory'], Path(old['artifact_directory'])]:
        if any(p.is_file() and p.suffix.lower() in ('.pt', '.pth', '.ckpt') for p in base.rglob('*')):
            raise ValueError('Inspect an unexpected saved native state before a fresh retry')
    if (ROOT / old['output_directory'] / 'result.json').exists():
        raise ValueError('Never duplicate an actual completed original experiment')
    old_preflight = read(ROOT / previous['preflight']); new_preflight = read(ROOT / config['preflight'])
    equivalent = copy.deepcopy(new_preflight); equivalent['queue_sha256'] = old_preflight['queue_sha256']
    if equivalent != old_preflight or new_preflight['queue_sha256'] != sha(ROOT / config['queue']):
        raise ValueError('Retain every original14-model full diagnostic observation')
    if check_live:
        import psutil
        for p in psutil.process_iter(['name', 'cmdline']):
            command = p.info['cmdline'] or []
            if ('python' in (p.info['name'] or '').lower() and '--job-id' in command
                    and IDENTITY in command):
                raise ValueError('Canonical fMRI43 native process already live; do not duplicate')
    return queue, old, job


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True); parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config)
    queue, old, job = verify(config)
    if args.probe_only:
        print(dict(full_original_jobs=70, full_original_cohorts=14, selected_canonical_id=IDENTITY,
            required_fresh_optimizer_updates=15000, credited_previous_updates=0,
            original_full_generation_shape=[10005, 24, 50], original_generation_batch_size=2001,
            required_evaluator_kinds=4, required_repeats_each=5,
            minimum_commit_headroom_bytes=config['selected_admission_settings']['minimum_commit_headroom_bytes'],
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            actual_training_executed=False, all_original_experiments_complete=False)); return
    previous = ordering.previous
    def write_with_scope(path, value):
        if str(path).replace('\\', '/').endswith(config['state_root'] + '/status.json'):
            value = dict(value, required_original_jobs=70, required_original_cohorts=14,
                selected_canonical_id=IDENTITY, selected_case_only=True,
                credited_previous_optimizer_updates=0, additional_training_seed_slots=0,
                all_original_experiments_complete=False)
        previous.original.write(path, value)

    namespace = dict(previous.original.__dict__, guarded_api=guarded_api, admission_settings=admission_settings,
        scheduler_jobs=ordering.scheduler_jobs, registered_settings=copy.deepcopy(queue['settings']),
        selected_canonical_id=IDENTITY, write=write_with_scope)
    exec(compile(selected_ast(inspect.getsource(previous.original.main)),
        '<unchanged full native queue selecting only preserved fMRI43 retry>', 'exec'), namespace)
    sys.argv = ['full_original_fmri_retry', '--queue', config['queue'], '--preflight', config['preflight']]
    namespace['main']()


if __name__ == '__main__':
    main()
