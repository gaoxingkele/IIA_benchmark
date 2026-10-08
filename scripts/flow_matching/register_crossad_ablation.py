"""Register all missing Table 3 rows and bind full-row controls to original jobs."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from iia_benchmark.models.fm_crossad_ablation import COMPONENTS
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    config_path = ROOT/'configs/experiments/fm_crossad_ablation_execution.v1.json'
    setup = json.loads(config_path.read_text())
    parent = json.loads((ROOT/'configs/experiments/fm_crossad_author_queue.v1.json').read_text())
    settings = dict(parent['settings'], output_root=setup['output_root'], queue_path=setup['queue_path'], boundary=setup['boundary'])
    jobs, bindings = [], []
    common = [config_path, ROOT/'src/iia_benchmark/models/fm_crossad_ablation.py',
              ROOT/'scripts/flow_matching/register_crossad_ablation.py', ROOT/'scripts/flow_matching/run_crossad_ablation_job.py',
              ROOT/'scripts/flow_matching/run_crossad_ablation_queue.py', ROOT/'docs/reports/fm_crossad_protocol_audit_2026-10-09.json']
    frozen = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in common]
    for row in setup['rows']:
        model_path = ROOT/setup['model_config_root']/f'fm_crossad_table3_row{row}.json'
        config = {'id': f'crossad_table3_row{row}', 'task': 'time_series_anomaly_detection',
                  'implementation': 'paper_reconstructed_author_harness', 'callable': 'iia_benchmark.models.fm_crossad_ablation.build_ablation',
                  'parameters': {'row': row, 'components': list(COMPONENTS[row]), 'parent': setup['parent_settings']},
                  'reproduction_status': 'Callable paper reconstruction; exact unpublished author ablation architecture unverified',
                  'citations': ['https://arxiv.org/abs/2510.12489', parent['settings']['source_url']], 'boundary': setup['boundary']}
        if model_path.exists() and json.loads(model_path.read_text()) != config:
            raise ValueError('Ablation model config frozen')
        write_json(model_path, config)
        for original in parent['jobs']:
            if original['mode'] != 'fresh_train' or original['dataset'] not in setup['datasets']:
                continue
            job = copy.deepcopy(original)
            identifier = f"crossad_ablation__row{row}__{job['dataset'].lower()}__seed{job['seed']}"
            job.update(id=identifier, ablation_row=row, author_recipe=f'paper_reconstructed_table3_row{row}',
                       model_config=model_path.relative_to(ROOT).as_posix(), output_directory=settings['output_root']+'/jobs/'+identifier)
            job['frozen_files'] += frozen + [{'path': model_path.relative_to(ROOT).as_posix(), 'sha256': sha(model_path)}]
            jobs.append(job)
    for original in parent['jobs']:
        if original['mode'] == 'fresh_train' and original['dataset'] in setup['datasets']:
            bindings.append({'table3_row': 6, 'dataset': original['dataset'], 'seed': original['seed'], 'original_job_id': original['id'],
                             'original_output_directory': original['output_directory'], 'completed': (ROOT/original['output_directory']/'result.json').exists()})
    jobs.sort(key=lambda j: (settings['seeds'].index(j['seed']), setup['rows'].index(j['ablation_row']), setup['datasets'].index(j['dataset'])))
    queue = {'schema_version': 1, 'settings': settings, 'jobs': jobs, 'boundary': setup['boundary'],
             'worker_script': 'run_crossad_ablation_queue.py', 'child_script': 'run_crossad_ablation_job.py'}
    target = ROOT/setup['queue_path']
    if target.exists() and json.loads(target.read_text()) != queue:
        raise ValueError('CrossAD ablation queue frozen')
    write_json(target, queue)
    report = {'registered_jobs': len(jobs), 'registered_stages': len(jobs)*4, 'full_row6_control_bindings': bindings,
              'queue_sha256': sha(target), 'all_six_rows_covered_by_jobs_or_original_bindings': len(jobs) == 90 and len(bindings) == 18,
              'boundary': setup['boundary'], 'all_experiments_complete': False}
    write_json(ROOT/'docs/reports/fm_crossad_ablation_registration_2026-10-09.json', report)
    print(json.dumps({'registered_jobs': len(jobs), 'row6_control_bindings': len(bindings)}))


if __name__ == '__main__':
    main()
