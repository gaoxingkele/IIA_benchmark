"""Freeze all six author MaelNet shell recipes as Windows-safe argv stages."""
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from iia_benchmark.models.fm_maelnet_runtime import prepare
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def script_arguments(text):
    lines = text.replace('\\\n', ' ').splitlines()
    commands = [shlex.split(line) for line in lines if line.strip().startswith('python -u run_anomaly.py')]
    if len(commands) != 1:
        raise ValueError('Expected one author Python stage')
    return commands[0][3:]


def main():
    recipe_path = ROOT / 'configs/experiments/fm_maelnet_author_execution.v1.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    manifest = prepare(ROOT / recipe['source'], ROOT / recipe['corrected_source'])
    common = ['configs/experiments/fm_maelnet_author_execution.v1.json', recipe['model_config'],
              'scripts/flow_matching/register_maelnet_author.py', 'scripts/flow_matching/run_maelnet_author_queue.py',
              'src/iia_benchmark/models/fm_maelnet_runtime.py', recipe['preflight_report'],
              recipe['corrected_source'] + '/runtime_patch_manifest.json']
    common += [recipe['corrected_source'] + '/' + r['path'] for r in manifest['files']]
    hashes = [{'path': p, 'sha256': sha(ROOT / p)} for p in common]
    jobs = []
    from scripts.flow_matching.run_maelnet_author_queue import argument
    for dataset in recipe['datasets']:
        data_config_path = recipe['dataset_config_root'] + '/mtsad_' + dataset.lower() + '.json'
        config = json.loads((ROOT / data_config_path).read_text(encoding='utf-8'))
        raw_directory = recipe['raw_root'] + '/' + config['directory']
        data_files = [raw_directory + '/' + file for file in dict.fromkeys(config['files'].values())]
        data_hashes = [{'path': p, 'sha256': sha(ROOT / p)} for p in data_files]
        for author_recipe in recipe['script_recipes']:
            stages = []
            for stage in recipe['stages']:
                script_path = recipe['source'] + '/' + recipe['script_recipe_root'] + '/' + author_recipe + '/' + dataset + '_scripts/' + stage
                args = script_arguments((ROOT / script_path).read_text(encoding='utf-8'))
                for field in ['enc_in', 'dec_in', 'c_out']:
                    if int(argument(args, field, 25)) != config['features']:
                        raise ValueError('Author script feature count differs: ' + script_path + ':' + field)
                stages.append({'id': Path(stage).stem, 'author_arguments': args, 'script': script_path, 'script_sha256': sha(ROOT / script_path)})
            fields = {'model_id': 'MaelNetS1_AnomalyTransformer_DCDetector_RL_Coba', 'data': 'MSL',
                      'features': 'M', 'd_model': '512', 'n_heads': '8', 'e_layers': '3', 'd_layers': '1',
                      'd_ff': '2048', 'factor': '5', 'embed': 'timeF'}
            keys = [tuple(argument(stage['author_arguments'], k, default) for k, default in fields.items()) for stage in stages]
            if len(set(keys)) != 1:
                raise ValueError('Author stage checkpoint setting differs: ' + script_path)
            for seed in recipe['seeds']:
                identifier = f'maelnet_author__{author_recipe.lower()}__{dataset.lower()}__seed{seed}'
                jobs.append({'id': identifier, 'dataset': dataset, 'seed': seed, 'author_recipe': author_recipe,
                             'raw_directory': raw_directory, 'stages': stages,
                             'output_directory': recipe['output_root'] + '/jobs/' + identifier,
                             'data_audit': config, 'frozen_files': hashes + data_hashes + [{'path': data_config_path, 'sha256': sha(ROOT / data_config_path)}]})
    queue = {'schema_version': 1, 'jobs': jobs, 'settings': recipe, 'boundary': recipe['boundary']}
    target = ROOT / recipe['queue_path']
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != queue:
            raise ValueError('Frozen author queue differs; use a new version')
    else:
        write_json(target, queue)
    write_json(ROOT / 'docs/reports/fm_maelnet_author_registration_2026-10-09.json',
               {'registered_jobs': len(jobs), 'author_stages': sum(len(j['stages']) for j in jobs),
                'queue_path': recipe['queue_path'], 'queue_sha256': sha(target), 'runtime_repairs': manifest['repairs'],
                'boundary': recipe['boundary'], 'complete': False})
    print(json.dumps({'registered_jobs': len(jobs), 'stages': sum(len(j['stages']) for j in jobs)}))


if __name__ == '__main__':
    main()
