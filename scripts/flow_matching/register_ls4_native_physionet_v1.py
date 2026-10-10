"""Freeze both released native LS4 patient tasks, source bindings and full budgets."""
import argparse
from pathlib import Path

import yaml

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true', required=True)
    parser.parse_args()
    queue_path = 'configs/experiments/fm_ls4_native_physionet_queue.v1.json'
    if (ROOT/queue_path).exists():
        raise FileExistsError('Preserve frozen native PhysioNet queue')
    loader = read(ROOT/'configs/reproducibility/fm_ls4_native_physionet_loader.2026-10-11_v2.json')
    old = read(ROOT/'configs/experiments/fm_ls4_cauchy_corrected_fred_nn5_queue.v3.json')
    artifact_root = 'F:/aicoding/IIA_Data/experiments/ls4_native_physionet_v1'
    profiles = ['interpolation','extrapolation']
    jobs = []
    for profile, source_config in zip(profiles, loader['native_configs']):
        native = yaml.safe_load((ROOT/source_config).read_text(encoding='utf-8'))
        for seed in old['seeds']:
            identity = f'ls4_native_physionet__{profile}__cauchy_corrected__seed{seed}'
            config_path = 'configs/models/'+identity+'.v1.json'
            config = dict(implementation='scripts.flow_matching.run_ls4_native_physionet_v1',
                reproduction_status='downloaded_author_full_native_pipeline_with_disclosed_Cauchy_correction',
                author_equivalence_certified=False, dataset='physionet2012', profile=profile, seed=seed,
                source_config=source_config, source_config_sha256=sha(ROOT/source_config),
                epochs=native['optim']['epochs'], original_optim=native['optim'], original_model=native['model'],
                batch_size=native['optim']['batch_size'], extrapolation=bool(native['data']['extrap']),
                train_patients=6400, test_patients=1600, variables=41,
                citations=['https://proceedings.mlr.press/v202/zhou23i.html',
                    'https://github.com/alexzhou907/ls4/tree/2b5cb76f89f6b935bbd697771433f3c74126ffca',
                    'https://physionet.org/content/challenge-2012/1.0.0/'])
            write(ROOT/config_path, config)
            jobs.append(dict(id=identity, profile=profile, seed=seed, model_config=config_path,
                output_directory='experiments/runs/fm_ls4_native_physionet_v1/jobs/'+identity,
                artifact_directory=artifact_root+'/jobs/'+identity))
    paths = {r['path'] for r in loader['source_receipts']}
    paths.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/loader['source_root']).rglob('*.py') if '__pycache__' not in p.parts)
    paths.update(j['model_config'] for j in jobs)
    paths.update(['scripts/flow_matching/run_ls4_native_physionet_v1.py',
        'scripts/flow_matching/run_ls4_native_physionet_queue_v1.py',
        'scripts/flow_matching/register_ls4_native_physionet_v1.py',
        'tests/test_fm_ls4_native_physionet_execution_v1.py',
        'scripts/flow_matching/ls4_cauchy_fallback_repair_v1.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_grasp_protocol_queue.py',
        'configs/reproducibility/ls4_cauchy_fallback_repair.v1.json',
        'configs/reproducibility/fm_ls4_native_physionet_loader.2026-10-11_v2.json',
        loader['output_directory']+'/original_loader_audit.json',loader['output_directory']+'/validation.json'])
    repair = read(ROOT/'configs/reproducibility/ls4_cauchy_fallback_repair.v1.json')
    paths.add(repair['official_cauchy_source'])
    queue = dict(queue_path=queue_path, source_root=loader['source_root'], python=loader['python'],
        environment_lock=loader['environment_lock'], prepared_root=loader['prepared_root'],
        loader_registration='configs/reproducibility/fm_ls4_native_physionet_loader.2026-10-11_v2.json',
        loader_proof=loader['output_directory']+'/original_loader_audit.json',
        cauchy_correction_config='configs/reproducibility/ls4_cauchy_fallback_repair.v1.json',
        cpu_threads=2, profiles=profiles, seeds=old['seeds'], jobs=jobs,
        state_root='experiments/runs/fm_ls4_native_physionet_v1',
        resource_lock=old['resource_lock'], settings=old['settings'],
        integration_lock='experiments/runs/fm_readonly_dataset_preprocessing_cpu.lock',
        integration_settings=dict(minimum_available_memory_bytes=17179869184, minimum_commit_headroom_bytes=30064771072,
            emergency_commit_headroom_bytes=17179869184, maximum_worker_rss_bytes=8589934592,
            maximum_worker_private_bytes=12884901888),
        integration_artifact=artifact_root+'/integration',
        integration_output='experiments/runs/fm_ls4_native_physionet_v1/preflight/integration',
        gpu_preflight_artifact=artifact_root+'/gpu_preflight',
        gpu_preflight_output='experiments/runs/fm_ls4_native_physionet_v1/preflight/gpu_preflight',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],
        expected_full_training_optimizer_updates=750000, all_original_experiments_complete=False,
        boundary='Actual downloaded train_interp_extrap.main/train/eval and VAE; original500epochs,41variables,6400/1600 fixed42 patient split,64/32 batches,50 whole-test evaluations. Official conjugate Cauchy runtime correction disclosed, raw source and data preserved. Original global train/test normalizer, interpolation without added heldout mask and test MSE checkpoint selection retained. Five local training seeds are a robustness extension; historical author equivalence unproved. CPU interface and full-GPU one-epoch preflight scores are diagnostics, not500epoch benchmark results. All45 papers/681 literature records/273 original task records remain in scope.')
    write(ROOT/queue_path, queue)
    print(dict(jobs=len(jobs), expected_optimizer_updates=750000, queue_sha256=sha(ROOT/queue_path)))


if __name__ == '__main__':
    main()
