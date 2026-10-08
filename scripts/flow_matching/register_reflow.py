"""Freeze iterative reflow and total-epoch controls without changing live queues."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Frozen artifact differs: ' + str(path))
    else:
        write_json(path, value)


def main():
    recipe_path = ROOT / 'configs/experiments/fm_reflow_execution.v1.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    if sha(ROOT / recipe['literature_pdf']) != recipe['literature_pdf_sha256']:
        raise ValueError('Rectified Flow paper changed')
    template = json.loads((ROOT / recipe['template_model']).read_text(encoding='utf-8'))
    queue = json.loads((ROOT / recipe['template_queue']).read_text(encoding='utf-8'))
    originals = [j for j in queue['jobs'] if j['model_config'] == recipe['template_model']]
    if len(originals) != 35:
        raise ValueError('Expected seven full datasets and five seeds')
    model_paths, jobs = [], []
    receipts = ['configs/experiments/fm_reflow_execution.v1.json', 'scripts/flow_matching/register_reflow.py',
                'src/iia_benchmark/models/fm_reflow.py', 'src/iia_benchmark/models/fm_objective_detectors.py',
                recipe['literature_pdf'], recipe['author_source_root'] + '/ImageGeneration/losses.py',
                recipe['author_source_root'] + '/ImageGeneration/run_lib_reflow.py']
    hashes = [{'path': p, 'sha256': sha(ROOT / p)} for p in receipts]
    for variant in recipe['variants']:
        model = copy.deepcopy(template)
        model.update(id=variant['id'], family='rectified_flow_iterative_reflow_local_tsad',
                     source_repository='gnobitab/RectifiedFlow', source_commit=recipe['author_commit'],
                     reproduction_status='iterative_reflow_transcription_full_local_tsad_not_original_image_reproduction',
                     deviations=[recipe['boundary'], 'No EMA; warm-start shared MLP with a fresh Adam at each reflow stage. Initial rectification retains the registered local CFM time/RNG convention.',
                                 'Couplings use as many generated pairs as full nonoverlapping training windows. Pair data, teacher states and per-stage metadata persist in checkpoint buffers.'])
        model['parameters']['epochs'] = variant['initial_epochs']
        if variant['stages'] > 1:
            model['entrypoint'] = 'iia_benchmark.models.fm_reflow:ReflowCFMDetector'
            model['parameters'].update(reflow_stages=variant['stages'], reflow_epochs=variant['reflow_epochs'],
                                       reflow_ode_steps=recipe['ode_steps'], reflow_solver=recipe['ode_solver'])
        model['data_citations'] = ['https://github.com/decisionintelligence/TAB']
        model_path = 'configs/models/' + model['id'] + '.json'
        immutable(ROOT / model_path, model)
        model_paths.append(model_path)
        for original in originals:
            job = copy.deepcopy(original)
            job['id'] = f"{model['id']}__{job['dataset']}__seed{job['seed']}"
            job['model_config'] = model_path
            job['output_directory'] = recipe['output_root'] + '/jobs/' + job['id']
            job['frozen_files'] = [r for r in job['frozen_files'] if r['path'] != recipe['template_model']]
            job['frozen_files'] += hashes + [{'path': model_path, 'sha256': sha(ROOT / model_path)}]
            job['total_training_epochs'] = variant['initial_epochs'] + (variant['stages'] - 1) * variant.get('reflow_epochs', 0)
            job['additional_pair_generation_compute'] = variant['stages'] > 1
            jobs.append(job)
    registered = {**{k: v for k, v in queue.items() if k != 'jobs'}, 'lane': 'cpu_reflow', 'jobs': jobs,
                  'state_root': recipe['output_root'], 'evaluation': {**queue['evaluation'], 'cpu_threads': recipe['cpu_threads']},
                  'boundary': recipe['boundary']}
    immutable(ROOT / recipe['queue_path'], registered)
    write_json(ROOT / 'docs/reports/fm_reflow_registration_2026-10-09.json',
               {'jobs': len(jobs), 'model_configs': model_paths, 'queue': recipe['queue_path'],
                'queue_sha256': sha(ROOT / recipe['queue_path']), 'recipe_sha256': sha(recipe_path),
                'boundary': recipe['boundary'], 'all_original_experiments_complete': False})
    print(json.dumps({'jobs': len(jobs), 'model_configs': model_paths}))


if __name__ == '__main__':
    main()
