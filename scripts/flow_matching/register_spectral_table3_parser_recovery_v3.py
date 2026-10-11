"""Recover missing config-parser dependencies while retaining all original jobs."""
import copy
import json
from pathlib import Path

from scripts.flow_matching.register_spectral_table3_original_v1 import write_new
from scripts.flow_matching.run_spectral_table3_original_v2 import ROOT, read, sha, verify


def require_science_unchanged(previous, current):
    ignored = {'id', 'reproduction_status', 'environment_dependency_recovery'}
    if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in current.items() if k not in ignored}:
        raise ValueError('Only execution identity and config-parser recovery metadata may change')


def main():
    previous_path = 'configs/experiments/fm_spectral_table3_original_queue.v2.json'
    previous = read(ROOT / previous_path)
    verify(previous)
    lock_path = 'configs/reproducibility/spectral_table3_author_py310.v2.lock.json'
    lock = read(ROOT / lock_path)
    if not lock['inherited_numerical_versions_identical'] or not lock['parent_environment_unchanged']:
        raise ValueError('Only a separately verified config-parser overlay is allowed')
    queue = copy.deepcopy(previous)
    queue.update(queue_path='configs/experiments/fm_spectral_table3_original_queue.v3.json',
        python='.venv/spectral-table3-author-py310-v2/Scripts/python.exe', environment_lock=lock_path,
        state_root='experiments/runs/fm_spectral_table3_original_v3',
        cpu_data_audit='experiments/runs/fm_spectral_table3_original_v3/preflight/cpu_data_v1/proof.json',
        previous_attempt_queue=previous_path, previous_attempt_queue_sha256=sha(ROOT / previous_path),
        attempt_scope='Same four original canonical seed42 jobs,1000epochs and all full generator/evaluator budgets. Missing OmegaConf/ANTLR installed offline in a separate overlay; no inherited numerical version or original source changed.')
    sources = {item['path'] for item in previous['source_receipts']}
    sources.update([previous_path, lock_path, 'configs/acquisition/spectral_table3_environment_additions.v2.json',
        'scripts/flow_matching/freeze_spectral_table3_environment_v2.py',
        'scripts/flow_matching/audit_spectral_table3_parser_v2.py',
        'scripts/flow_matching/register_spectral_table3_parser_recovery_v3.py',
        'tests/test_fm_spectral_table3_parser_overlay_v2.py',
        'tests/test_fm_spectral_table3_parser_recovery_v3.py'])
    jobs = []
    for job in previous['jobs']:
        old = read(ROOT / job['model_config'])
        identity = job['original_canonical_id'] + '__parser_env_fix_v3'
        config = dict(old, id=identity, reproduction_status='config_parser_recovered_full_native_gpu_preflight_and_original_training_pending',
            environment_dependency_recovery='Separate OmegaConf2.2.0/ANTLR4.8 overlay. Original YAML, source, data, methods, budgets, seeds, optimizer and evaluator unchanged. Historical parser versions remain unknown.')
        require_science_unchanged(old, config)
        path = 'configs/models/fm_' + identity + '.json'
        write_new(ROOT / path, config)
        sources.add(path)
        native_root = Path(job['artifact_directory']).parent.parent / 'spectral_table3_original_v3'
        jobs.append(dict(job, id=identity, model_config=path, model_config_sha256=sha(ROOT / path),
            artifact_directory=str(native_root / identity),
            preflight_artifact_directory=str(native_root / ('preflight_' + identity)),
            output_directory=queue['state_root'] + '/jobs/' + identity))
    queue['jobs'] = jobs
    queue['source_receipts'] = [dict(path=path, sha256=sha(ROOT / path)) for path in sorted(sources)]
    write_new(ROOT / queue['queue_path'], queue)
    runtime = read(ROOT / 'configs/runtime/fm_spectral_table3_gpu_memory.v3.json')
    runtime.update(queue=queue['queue_path'], queue_sha256=sha(ROOT / queue['queue_path']), full_job_count=4)
    paths = {item['path'] for item in runtime['source_receipts']} | sources | {queue['queue_path']}
    runtime['source_receipts'] = [dict(path=path, sha256=sha(ROOT / path)) for path in sorted(paths)]
    write_new(ROOT / 'configs/runtime/fm_spectral_table3_gpu_memory.v4.json', runtime)
    print(json.dumps(dict(full_original_jobs=len(jobs), additional_training_seed_slots=0,
        original_settings_unchanged=queue['settings'] == previous['settings'],
        queue=queue['queue_path'], full_gpu_memory_admission_unchanged=True)))


if __name__ == '__main__':
    main()
