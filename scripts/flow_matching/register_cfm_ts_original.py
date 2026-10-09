"""Freeze all main-text/appendix CFM-TS budgets and paired original ODE cases."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import platform

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT, read, sha


def environment():
    import torch
    import torchdiffeq
    import numpy
    import scipy
    module = importlib.import_module('torchdiffeq._impl.odeint')
    adjoint = importlib.import_module('torchdiffeq._impl.adjoint')
    return {'python': platform.python_version(), 'torch': torch.__version__, 'torchdiffeq': torchdiffeq.__version__,
            'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'odeint_source_sha256': sha(module.__file__), 'adjoint_source_sha256': sha(adjoint.__file__)}


def write_new(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    config_path = 'configs/reproducibility/cfm_ts_original_protocol.v1.json'
    protocol = read(ROOT / config_path)
    if (ROOT / protocol['queue']).exists():
        raise FileExistsError('Queue already frozen')
    manifest = read(ROOT / protocol['manifest'])
    if manifest['protocol_sha256'] != sha(ROOT / config_path) or len(manifest['cases']) != 30:
        raise ValueError('Complete registered paired data required')
    if any(sha(ROOT / case['path']) != case['sha256'] for case in manifest['cases']):
        raise ValueError('Simulation file changed')
    write_new(protocol['environment_lock'], environment())
    models = {}
    settings = dict(protocol['network'])
    settings.pop('activation')
    settings.update(grid_points=50, device='cpu', process_noise=protocol['process_noise'],
                    observation_noise=protocol['observation_noise'], gp_max_iterations=protocol['gp_max_iterations'],
                    solver=protocol['learned_ode_solver']['method'],
                    **{k: v for k,v in protocol['learned_ode_solver'].items() if k != 'method'})
    variants = protocol['cfm_variants'] + [{'id': 'node', 'method': 'node', 'samples': 1, 'target_mode': 'not_applicable'}]
    for variant in variants:
        epochs = [10,400,600] if variant['method'] == 'node' else [400,600]
        for budget in epochs:
            name = f"fm_cfm_ts_original_{variant['id']}_e{budget}"
            path = 'configs/models/' + name + '.json'
            parameters = dict(settings, epochs=budget, batch_size=256, seed=1103,
                              **{k:v for k,v in variant.items() if k != 'id'})
            model = {'schema_version':1, 'id':name, 'family':'paper_derived_original_continuous_dynamics',
                     'task':'continuous_time_dynamics', 'kind':'trajectory',
                     'entrypoint':'iia_benchmark.models.fm_cfm_ts_original:CFMTSPaperTrajectory',
                     'parameters':parameters, 'citation':protocol['citation'], 'paper_id':protocol['paper_id'],
                     'paper_source_sha256':protocol['paper_sha256'],
                     'data_reference':[d['id'] for d in protocol['datasets']],
                     'reproduction_status':'executable_paper_derived_reconstruction_not_author_equivalent',
                     'protocol_configuration':config_path, 'design_choices':protocol['explicit_interpretations']}
            write_new(path, model)
            models[variant['id'], budget] = path
    jobs = []
    for track in protocol['tracks']:
        for case in [c for c in manifest['cases'] if c['track'] == track['id']]:
            for variant in variants:
                budgets = track['node_epochs'] if variant['method'] == 'node' else track['cfm_epochs']
                for budget in budgets:
                    name = f"cfm_ts_original__{track['id']}__{variant['id']}_e{budget}__{case['dataset']}__seed{case['seed']}"
                    config = models[variant['id'], budget]
                    training_items = case['train_trajectories'] if variant['method'] == 'node' else case['train_trajectories']*case['observation_points']*variant['samples']
                    jobs.append({'id':name, 'track':track['id'], 'dataset':case['dataset'], 'seed':case['seed'],
                                 'variant':variant['id'], 'method':variant['method'], 'epochs':budget,
                                 'batch_size':track['batch_size'], 'case_id':case['id'],
                                 'data_path':case['path'], 'data_sha256':case['sha256'],
                                 'model_config':config, 'model_config_sha256':sha(ROOT / config),
                                 'model_overrides':{'batch_size':track['batch_size'], 'grid_points':case['observation_points'], 'seed':case['seed']},
                                 'training_items':training_items, 'expected_optimizer_updates':budget*math.ceil(training_items/track['batch_size']),
                                 'test_trajectories':case['test_trajectories'], 'test_points_per_trajectory':case['test_points_per_trajectory'],
                                 'observed_dimensions':case['observed_dimensions'],
                                 'split':case['split'], 'output_directory':protocol['state_root']+'/jobs/'+name,
                                 'artifact_directory':protocol['artifact_root']+'/'+name})
    jobs.sort(key=lambda j: ({'pendulum':0,'sine':1,'lotka_volterra':2}[j['dataset']],
                            [v['id'] for v in variants].index(j['variant']), j['epochs'], j['track'], j['seed']))
    if len(jobs) != 285 or len({j['id'] for j in jobs}) != 285:
        raise AssertionError('Complete two-track/five-seed scope differs')
    paths = [config_path, protocol['manifest'], protocol['paper_pdf'], protocol['environment_lock'],
             'src/iia_benchmark/models/fm_cfm_ts_original.py', 'src/iia_benchmark/models/fm_native_trajectory.py',
             'scripts/flow_matching/prepare_cfm_ts_original_data.py', 'scripts/flow_matching/register_cfm_ts_original.py',
             'scripts/flow_matching/run_cfm_ts_original_job.py', 'scripts/flow_matching/run_cfm_ts_original_queue.py',
             'scripts/flow_matching/heavy_resources.py', 'tests/test_fm_cfm_ts_original.py']
    queue = {'schema_version':1, 'protocol':config_path, 'manifest':protocol['manifest'],
             'python':protocol['python'], 'environment_lock':protocol['environment_lock'],
             'state_root':protocol['state_root'], 'resource_policy':protocol['resource_policy'],
             'jobs':jobs, 'source_receipts':[{'path':p,'sha256':sha(ROOT / p)} for p in paths],
             'registered_jobs':285, 'registered_data_cohorts':30, 'registered_model_configs':len(models),
             'author_equivalence_certified':False, 'diagnostic_smoke':False,
             'boundary':'Full original ODE tasks with paper-derived data. Conflicting protocols and target interpretations are separate; no test-based budget selection. Original author raw data/code and unstated numerical details remain gaps.'}
    write_new(protocol['queue'], queue)
    print(json.dumps({'registered_jobs':len(jobs), 'models':len(models), 'data_cohorts':len(manifest['cases'])}))


if __name__ == '__main__':
    main()
