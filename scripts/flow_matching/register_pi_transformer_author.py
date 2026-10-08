"""Freeze original Pi dataset hyperparameters and all directly specified training axes."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from iia_benchmark.models.fm_pi_transformer_runtime import prepare
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def recipes(defaults, settings):
    yield 'full', copy.deepcopy(defaults), {}
    for axis, values in settings['training_axes'].items():
        section, key = axis.split('.')
        for value in values:
            if value == defaults[section][key]:
                continue
            config = copy.deepcopy(defaults)
            config[section][key] = value
            yield axis.replace('.', '_') + '_' + str(value), config, {axis: value}
    for axis, values in settings['multiplier_axes'].items():
        section, key = axis.split('.')
        for multiplier in values:
            config = copy.deepcopy(defaults)
            config[section][key] *= multiplier
            yield axis.replace('.', '_') + '_x' + str(multiplier), config, {axis + '_multiplier': multiplier}


def main():
    import yaml
    settings_path = ROOT / 'configs/experiments/fm_pi_transformer_author_execution.v1.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    author = yaml.safe_load((ROOT / settings['source'] / 'config.yaml').read_text(encoding='utf-8'))
    common = [settings_path.relative_to(ROOT).as_posix(), settings['model_config'], settings['paper'],
              'src/iia_benchmark/models/fm_pi_transformer_runtime.py',
              'scripts/flow_matching/register_pi_transformer_author.py',
              'scripts/flow_matching/run_pi_transformer_job.py',
              'scripts/flow_matching/run_pi_transformer_author_queue.py']
    jobs = []
    for branch in settings['branches']:
        runtime = settings['runtime_root'] + '/' + branch
        manifest = prepare(ROOT / settings['source'], ROOT / runtime, branch)
        runtime_files = [runtime + '/' + r['path'] for r in manifest['files']] + [runtime + '/runtime_patch_manifest.json']
        original_files = [settings['source'] + '/' + r['path'] for r in manifest['source_files']]
        frozen = [{'path': p, 'sha256': sha(ROOT / p)} for p in common + runtime_files + original_files]
        for dataset in settings['datasets']:
            config_path = settings['dataset_config_root'] + '/mtsad_' + dataset.lower() + '.json'
            data = json.loads((ROOT / config_path).read_text(encoding='utf-8'))
            raw = settings['raw_root'] + '/' + data['directory']
            originals = [{'path': raw + '/' + p, 'sha256': sha(ROOT / raw / p)} for p in dict.fromkeys(data['files'].values())]
            base = author['datasets'][dataset]
            if base['model']['enc_in'] != data['features']:
                raise ValueError('Pi native feature count differs')
            for recipe, config, changes in recipes(base, settings):
                for seed in settings['seeds']:
                    identifier = f'pi_author__{branch}__{dataset.lower()}__{recipe}__seed{seed}'
                    jobs.append({'id': identifier, 'dataset': dataset, 'seed': seed,
                                 'author_recipe': branch + '/' + recipe, 'branch': branch,
                                 'recipe': recipe, 'changes': changes, 'runtime': runtime,
                                 'author_config': config, 'data_audit': data, 'raw_directory': raw,
                                 'stages': [{'id': 'train'}, {'id': 'test_and_score_controls'}],
                                 'output_directory': settings['output_root'] + '/jobs/' + identifier,
                                 'frozen_files': frozen + originals + [{'path': config_path, 'sha256': sha(ROOT / config_path)}]})
    # Full original and corrected baselines first, then every frozen axis.
    jobs.sort(key=lambda j: (j['recipe'] != 'full', settings['datasets'].index(j['dataset']),
                             settings['seeds'].index(j['seed']), settings['branches'].index(j['branch']), j['recipe']))
    queue = {'schema_version': 1, 'jobs': jobs, 'settings': settings, 'boundary': settings['boundary'],
             'worker_script': 'run_pi_transformer_author_queue.py', 'child_script': 'run_pi_transformer_job.py'}
    target = ROOT / settings['queue_path']
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != queue:
        raise ValueError('Pi queue already frozen; create a new version')
    if not target.exists():
        write_json(target, queue)
    report = {'registered_jobs': len(jobs), 'registered_stages': len(jobs) * 2,
              'queue': settings['queue_path'], 'queue_sha256': sha(target),
              'baseline_jobs': sum(j['recipe'] == 'full' for j in jobs),
              'dataset_counts': {d: sum(j['dataset'] == d for j in jobs) for d in settings['datasets']},
              'remaining_journal_axes': settings['remaining_journal_axes'], 'boundary': settings['boundary'],
              'strict_TAB_result': False, 'complete': False}
    write_json(ROOT / 'docs/reports/fm_pi_transformer_author_registration_2026-10-09.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
