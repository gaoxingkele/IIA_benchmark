"""Freeze all seven official-data checkpoints and original-budget training runs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    settings_path = ROOT/'configs/experiments/fm_crossad_author_execution.v1.json'
    settings = json.loads(settings_path.read_text())
    source, runtime, cache = [ROOT/settings[k] for k in ['source', 'runtime_root', 'cache_root']]
    common = [settings_path, ROOT/settings['model_config'], ROOT/settings['paper'],
              ROOT/'src/iia_benchmark/models/fm_crossad_author_runtime.py',
              ROOT/'scripts/flow_matching/prepare_crossad_author.py', ROOT/'scripts/flow_matching/register_crossad_author.py',
              ROOT/'scripts/flow_matching/run_crossad_author_job.py', ROOT/'scripts/flow_matching/run_crossad_author_queue.py']
    common += [p for p in source.rglob('*.py') if '__pycache__' not in p.parts]
    common += [p for p in runtime.rglob('*.py') if '__pycache__' not in p.parts] + [runtime/'runtime_manifest.json']
    common += [ROOT/settings['raw_root']/'DETECT_META.csv']
    common_frozen = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in common]
    jobs = []
    for name in settings['datasets']:
        data = json.loads((cache/(name+'.json')).read_text())
        model_path, train_path = [source/'configs'/name/p for p in ['model_configs_0.json', 'train_configs.json']]
        checkpoint = source/'configs'/name/'checkpoints_0/checkpoint.pth'
        frozen = common_frozen + [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in
                 [ROOT/data['raw_path'], cache/(name+'.pkl'), cache/(name+'.json'), model_path, train_path, checkpoint]]
        for mode, seeds in [('release_checkpoint', [2025]), ('fresh_train', settings['seeds'])]:
            for seed in seeds:
                identifier = f'crossad_author__{mode}__{name.lower()}__seed{seed}'
                jobs.append({'id': identifier, 'dataset': name, 'seed': seed, 'mode': mode, 'author_recipe': mode,
                    'model_parameters': json.loads(model_path.read_text()), 'train_parameters': json.loads(train_path.read_text()),
                    'raw_sha256': data['raw_sha256'], 'train_length': data['train_length'],
                    'release_checkpoint': checkpoint.relative_to(ROOT).as_posix(),
                    'output_directory': settings['output_root']+'/jobs/'+identifier, 'frozen_files': frozen,
                    'stages': [{'id': 'weights'}, {'id': 'author_test'}, {'id': 'author_spot'}, {'id': 'native_control'}]})
    jobs.sort(key=lambda j: (j['mode'] != 'release_checkpoint', settings['seeds'].index(j['seed']), settings['datasets'].index(j['dataset'])))
    queue = {'schema_version': 1, 'settings': settings, 'jobs': jobs, 'boundary': settings['boundary'],
             'worker_script': 'run_crossad_author_queue.py', 'child_script': 'run_crossad_author_job.py'}
    path = ROOT/settings['queue_path']
    if path.exists() and json.loads(path.read_text()) != queue:
        raise ValueError('CrossAD queue frozen; create new version')
    write_json(path, queue)
    report = {'registered_jobs': len(jobs), 'release_checkpoint_jobs': 7, 'fresh_train_jobs': 42,
              'registered_stages': sum(len(j['stages']) for j in jobs), 'queue_sha256': sha(path),
              'unique_frozen_files': len({r['path'] for j in jobs for r in j['frozen_files']}),
              'boundary': settings['boundary'], 'all_original_paper_experiments_complete': False}
    write_json(ROOT/'docs/reports/fm_crossad_author_registration_2026-10-09.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
