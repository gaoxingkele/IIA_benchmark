"""Register actual MOMENT release sizes/contexts on native grouped full-data tracks."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Registered MOMENT artifact differs; use a new version')
    else:
        write_json(path, value)


def main():
    settings_path = ROOT / 'configs/experiments/fm_moment_pretrained_execution.v1.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    source_files = [p for p in (ROOT / settings['source_root']).rglob('*.py') if '__pycache__' not in p.parts]
    common = [settings_path, ROOT / 'src/iia_benchmark/models/fm_moment_pretrained.py',
              ROOT / 'src/iia_benchmark/models/fm_supplemental_author.py',
              ROOT / 'scripts/flow_matching/model_registry.py', ROOT / 'scripts/flow_matching/run_moment_pretrained_job.py',
              ROOT / 'scripts/flow_matching/run_moment_pretrained_queue.py', ROOT / 'scripts/flow_matching/register_moment_pretrained.py',
              ROOT / 'scripts/flow_matching/run_tsad_experiment.py', ROOT / 'scripts/flow_matching/prepare_tsad_execution.py',
              ROOT / 'src/iia_benchmark/evaluation/mtsad_strict.py', ROOT / 'src/iia_benchmark/evaluation/mtsad_metrics.py'] + source_files
    common_hashes = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in common]
    reference = json.loads((ROOT / settings['evaluation_reference_queue']).read_text(encoding='utf-8'))
    jobs, models = [], []
    for size in settings['sizes']:
        directory = settings['checkpoint_root'] + '/MOMENT-1-' + size
        weights_path, config_path = directory+'/model.safetensors', directory+'/config.json'
        for context in settings['contexts']:
            identifier = 'fm_moment_pretrained_' + size + '__' + context['track']
            model_path = settings['model_output_directory'] + '/' + identifier + '.json'
            model = {'schema_version': 1, 'id': identifier, 'task': 'time_series_anomaly_detection', 'kind': 'window',
                     'entrypoint': 'iia_benchmark.models.fm_moment_pretrained:MomentPretrainedWindow',
                     'parameters': {'source_root': settings['source_root'], 'checkpoint_directory': directory,
                                    'checkpoint_sha256': sha(ROOT / weights_path), 'config_sha256': sha(ROOT / config_path),
                                    'window': context['window'], 'batch_size': settings['batch_size'], 'epochs': 0,
                                    'padding_policy': context['padding_policy'], 'seed': settings['seed'], 'device': 'cpu'},
                     'citation': 'https://arxiv.org/abs/2402.03885', 'data_citations': ['https://github.com/decisionintelligence/TAB'],
                     'source_commit': settings['source_commit'], 'source_repository': settings['source_repository'],
                     'checkpoint_citation': 'https://huggingface.co/AutonLab/MOMENT-1-' + size,
                     'reproduction_status': 'actual_official_release_pretrained_zero_shot_not_historical_paper_checkpoint_equivalence',
                     'repeat_boundary': settings['repeat_boundary'], 'boundary': settings['boundary']}
            immutable(ROOT / model_path, model)
            models.append(model_path)
            extra = [{'path': p, 'sha256': sha(ROOT / p)} for p in [model_path, weights_path, config_path]]
            for kind, manifest_path, dataset_ids in [('standard', settings['shared_input_manifest'], settings['datasets']),
                                                    ('industrial', settings['industrial_input_manifest'], settings['industrial_datasets'])]:
                manifest = json.loads((ROOT / manifest_path).read_text(encoding='utf-8'))
                for dataset_id in dataset_ids:
                    dataset = next(d for d in manifest['datasets'] if d['id'] == dataset_id)
                    frozen = common_hashes + extra + [{'path': manifest_path, 'sha256': sha(ROOT / manifest_path)},
                                                      {'path': dataset['input_npz'], 'sha256': dataset['input_sha256']}]
                    frozen += [{'path': r['path'], 'sha256': r['sha256']} for r in dataset['source_files']]
                    job_id = identifier + '__' + dataset_id + '__seed' + str(settings['seed'])
                    jobs.append({'id': job_id, 'dataset': dataset_id, 'seed': settings['seed'], 'window': context['window'],
                                 'model_config': model_path, 'dataset_manifest_path': manifest_path, 'input_kind': kind,
                                 'parameter_overrides': {'device': 'cpu', 'seed': settings['seed']},
                                 'output_directory': settings['output_root'] + '/jobs/' + job_id, 'frozen_files': frozen})
    queue = {'schema_version': 1, 'lane': 'moment_pretrained', 'state_root': settings['output_root'],
             'jobs': jobs, 'python': str(ROOT / settings['python']),
             'evaluation': dict(reference['evaluation'], cpu_threads=settings['cpu_threads']),
             'worker_script': 'run_moment_pretrained_queue.py', 'boundary': settings['boundary']}
    immutable(ROOT / settings['queue_path'], queue)
    report = {'registered_jobs': len(jobs), 'model_configs': models, 'queue_path': settings['queue_path'],
              'queue_sha256': sha(ROOT / settings['queue_path']), 'boundary': settings['boundary'],
              'repeat_boundary': settings['repeat_boundary'], 'goal_achieved': False}
    write_json(ROOT / 'docs/reports/fm_moment_pretrained_registration_2026-10-09.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
