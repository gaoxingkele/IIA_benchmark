"""Freeze full released/corrected GiFlow training, prior branches and sensitivities."""
import argparse
import json
from pathlib import Path

from scripts.flow_matching.giflow_native_protocol import ROOT, sha


def build_jobs():
    defaults = dict(model_state='train', missing_rate=.2, missing_type='point', model_name='FlowMatching',
                    diffusion_type='graph_time', device='cuda:0', k_eig=50, tau_s=.1425, tau_t=.3931,
                    channel=1, hidden_dim=64, propagation_layers=5, spatial_layers=4, dropout=.2,
                    training_epoch=300, batch_size=128, lr=.001, patience=40, lr_decay=True,
                    lr_decay_factor=.3, lr_decay_patience=10, weight_decay=1e-4, ema_start_epoch=30,
                    cuda=True, gpu=0, window=24, stride=1, adj_threshold=.1, val_len=.1, test_len=.2)
    recipes = [('air36', 'airquality_small', 'main', {})]
    recipes += [('aqi', 'AirQuality', 'main', {})]
    recipes += [('air36', 'airquality_small', 'prior_' + mode, {'diffusion_type': mode})
                for mode in ('graph', 'time', 'none')]
    recipes += [('air36', 'airquality_small', 'point_' + str(rate), {'missing_rate': rate})
                for rate in (.3, .4, .5, .6)]
    recipes += [('air36', 'airquality_small', 'threshold_' + str(threshold), {'adj_threshold': threshold})
                for threshold in (.02, .05, .2, .4, .6)]
    jobs = []
    # Pair identical seed/mask settings across the untouched and reviewed sources.
    for dataset, native_name, recipe, overrides in recipes:
        for seed in range(393, 398):
            for track in ('released_mirror', 'reviewed_corrected'):
                identity = f'giflow_native__{track}__{dataset}__{recipe}__seed{seed}'.replace('.', 'p')
                jobs.append({'id': identity, 'dataset': dataset, 'track': track, 'recipe': recipe, 'seed': seed,
                             'model_config': 'configs/models/fm_giflow_native_' + ('release' if track == 'released_mirror' else 'corrected') + '_v1.json',
                             'arguments': dict(defaults, dataset_name=native_name, seed=seed, **overrides),
                             'author_source': 'experiments/runs/flow_matching_campaign/sources/giflow/' +
                                              ('original' if track == 'released_mirror' else 'corrected'),
                             'output_directory': 'experiments/runs/fm_giflow_native_v1/jobs/' + identity})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='configs/experiments/fm_giflow_native_queue.v1.json')
    target = ROOT / parser.parse_args().output
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Already frozen; use the verifier or a reviewed successor')
    sources = [p for variant in ('original', 'corrected')
               for p in (ROOT / 'experiments/runs/flow_matching_campaign/sources/giflow' / variant).rglob('*.py')
               if '__pycache__' not in p.parts]
    files = ['scripts/flow_matching/giflow_native_protocol.py', 'scripts/flow_matching/register_giflow_native.py',
             'scripts/flow_matching/run_giflow_native_job.py', 'scripts/flow_matching/run_giflow_native_queue.py',
             'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/heavy_resources.py',
             'scripts/flow_matching/heavy_scheduler_v2.py', 'scripts/flow_matching/phase_gate.py',
             'scripts/flow_matching/run_tsad_queue.py', 'scripts/flow_matching/prepare_tsad_execution.py',
             'configs/runtime/fm_light_controllers.v1.json', 'configs/data/fm_graph_author_air.v1.json',
             'configs/experiments/fm_grin_original_queue.v1.json',
             'configs/reproducibility/graph_author_environments.v1.json',
             'configs/reproducibility/giflow_author_py310_requirements.lock.txt',
             'configs/models/fm_giflow_native_release_v1.json', 'configs/models/fm_giflow_native_corrected_v1.json',
             'experiments/runs/flow_matching_campaign/sources/giflow/patch_manifest.json',
             'papers/literature/flow_matching/pdfs/giflow.pdf']
    sources += [ROOT / p for p in files]
    runtime = json.loads((ROOT / 'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    sources += [ROOT / r['path'] for r in runtime['source_receipts']]
    data = json.loads((ROOT / 'configs/data/fm_graph_author_air.v1.json').read_text(encoding='utf-8'))
    staged = ROOT / data['output_root']
    sources += [staged / 'data_audit.json'] + [staged / p for p in data['members']]
    queue = {'schema_version': 1, 'id': 'giflow_native_v1', 'python': '.venv/giflow-author-py310/Scripts/python.exe',
             'state_root': 'experiments/runs/fm_giflow_native_v1',
             'prediction_artifact_root': 'F:/aicoding/IIA_Data/experiments/giflow_native_v1',
             'data_config': 'configs/data/fm_graph_author_air.v1.json',
             'environment_lock': 'configs/reproducibility/giflow_author_py310_requirements.lock.txt',
             'runtime_config': 'configs/runtime/fm_light_controllers.v1.json',
             'requires_complete_queue': 'configs/experiments/fm_grin_original_queue.v1.json',
             'settings': {'minimum_available_memory_bytes': 12 * 1024**3,
                          'minimum_commit_headroom_bytes': 16 * 1024**3,
                          'emergency_commit_headroom_bytes': 8 * 1024**3, 'cpu_threads': 4},
             'jobs': build_jobs(),
             'citations': ['https://arxiv.org/abs/2606.06682', 'https://github.com/zepengzhang/GiFlow',
                           'https://github.com/Graph-Machine-Learning-Group/grin#datasets'],
             'boundary': 'Full released-source training, 300 maximum epochs, native patience40, batch128, window24/stride1, 20 Euler steps, five paired seeds. Reviewed correction fixes validation test contamination, EMA resetting, best-checkpoint evaluation and data/device routing only. Released graph_time overwrites spatial prior with temporal; none is observed/zero prior, not paper FM-Gauss. Fixed taus, AdamW and chronological split retained. No paper-best or paper-ablation equivalence certification.',
             'remaining_paper_obligations': ['Paper Adam/patience10/random70-10-20 versus released AdamW/patience40/TemporalSplitter (validation10% of non-test).',
                  'Learned filtering-factor Problem5 and composed spatial-temporal Eq4 absent in release.',
                  'Block masks, PeMS08, attention/propagation ablations, Gaussian prior, Euler-step comparison and paper hyperparameter search remain required.',
                  'Air36 release has 8759 timestamps versus paper8760; full AQI batch128 must actually fit; no reduced batch fallback.'],
             'source_receipts': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sorted(set(sources))]}
    target.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'jobs': len(queue['jobs']), 'config_sha256': sha(target), 'source_files': len(queue['source_receipts'])}))


if __name__ == '__main__':
    main()
