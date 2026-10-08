"""Register all full-budget jobs on a new immutable optional-metric runtime."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.models.fm_crossad_evaluator_compat import prepare_runtime
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    parent_path = ROOT / 'configs/experiments/fm_crossad_complete_queue.v6.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    queue = copy.deepcopy(parent)
    queue['schema_version'] = 7
    settings = queue['settings']
    runtime = 'data/public_datasets/flow_matching/crossad_author_runtime_v2'
    prepare_runtime(ROOT / settings['runtime_root'], ROOT / runtime)
    report_path = ROOT / 'docs/reports/fm_crossad_evaluator_compat_2026-10-09.json'
    proof = json.loads(report_path.read_text(encoding='utf-8'))
    assert proof['all_requested_metric_dataframes_bit_exact'] and not proof['benchmark_performance']
    assert proof['patched_runtime_manifest_sha256'] == sha(ROOT / runtime / 'runtime_manifest.json')
    settings.update(runtime_root=runtime, output_root='experiments/runs/fm_crossad_author_v2',
                    queue_path='configs/experiments/fm_crossad_complete_queue.v7.json')
    settings['boundary'] += (' Evaluator optional PATE import/call made conditional in a new runtime; '
                             'the four requested author metric calculations are unchanged and bit-exact in a diagnostic comparison. '
                             'Original failed full GECCO inference artifacts retained in the previous output root. '
                             'PATE dependency compatibility when PATE is requested remains a separate obligation.')
    paths = [ROOT / 'src/iia_benchmark/models/fm_crossad_evaluator_compat.py',
             ROOT / 'scripts/flow_matching/register_crossad_evaluator_compat.py', report_path,
             ROOT / runtime / 'runtime_manifest.json', *sorted((ROOT / runtime).rglob('*.py'))]
    frozen = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths]
    for old, job in zip(parent['jobs'], queue['jobs']):
        job['output_directory'] = settings['output_root'] + '/jobs/' + job['id']
        assert not (ROOT / job['output_directory']).exists(), job['id']
        job['frozen_files'] += frozen
        assert {k: v for k, v in job.items() if k not in ('output_directory', 'frozen_files')} == {
            k: v for k, v in old.items() if k not in ('output_directory', 'frozen_files')}
    assert len(queue['jobs']) == 139
    queue['evaluator_parent_queue'] = parent_path.relative_to(ROOT).as_posix()
    queue['evaluator_parent_sha256'] = sha(parent_path)
    target = ROOT / settings['queue_path']
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != queue:
        raise ValueError('Compatibility queue already frozen differently')
    write_json(target, queue)
    registration = {'queue': target.relative_to(ROOT).as_posix(), 'queue_sha256': sha(target), 'registered_jobs': 139,
                    'parent_queue': queue['evaluator_parent_queue'], 'parent_sha256': sha(parent_path),
                    'runtime_root': runtime, 'runtime_manifest_sha256': sha(ROOT / runtime / 'runtime_manifest.json'),
                    'original_algorithm_budgets_seeds_and_data_preserved': True, 'new_output_root': settings['output_root'],
                    'optional_metric_removed': False, 'original_failure_artifacts_preserved': True,
                    'diagnostic_metrics_equivalent': True, 'all_experiments_complete': False}
    write_json(ROOT / 'docs/reports/fm_crossad_evaluator_compat_registration_2026-10-09.json', registration)
    print(json.dumps(registration, indent=2))


if __name__ == '__main__':
    main()
