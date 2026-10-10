"""Freeze full native USHCN training over both split branches/all five folds."""
import argparse

import yaml

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true', required=True)
    parser.parse_args()
    queue_path = 'configs/experiments/fm_ls4_native_ushcn_queue.v1.json'
    if (ROOT/queue_path).exists():
        raise FileExistsError('Preserve every frozen climate training registration')
    loader_path = 'configs/reproducibility/fm_ls4_native_ushcn_loader.2026-10-11_v1.json'
    loader = read(ROOT/loader_path)
    proof_path = loader['output_directory']+'/original_loader_audit.json'
    proof = read(ROOT/proof_path)
    if not proof['full_original_loader_passed'] or proof['full_profiles'] != 20:
        raise ValueError('Full native climate data audit required')
    old = read(ROOT/'configs/experiments/fm_ls4_native_physionet_queue.v2.json')
    artifact_root = 'F:/aicoding/IIA_Data/experiments/ls4_native_ushcn_v1'
    profiles = ['interpolation','extrapolation']
    branches = ['source_literal','id_corrected']
    jobs, cases = [], []
    for branch in branches:
        for fold in range(5):
            split_dir = loader['prepared_directory']+'/'+branch+f'/small_chunk_fold_idx_{fold}'
            split_hashes = [dict(path=split_dir+'/'+k+'_idx.npy',sha256=sha(ROOT/split_dir/(k+'_idx.npy')))
                            for k in ('train','val','test')]
            for profile, source_config in zip(profiles,loader['native_configs']):
                native = yaml.safe_load((ROOT/source_config).read_text(encoding='utf-8'))
                audited = next(r for r in proof['profiles'] if r['branch']==branch and r['fold']==fold
                               and r['source_config']==source_config)
                counts = audited['grouped_series_counts']
                case_id = f'{branch}__fold{fold}__{profile}'
                for seed in old['seeds']:
                    identity = f'ls4_native_ushcn__{case_id}__cauchy_corrected__seed{seed}'
                    config_path = 'configs/models/'+identity+'.v1.json'
                    if (ROOT/config_path).exists():
                        raise FileExistsError('Preserve every native model registration')
                    model = dict(implementation='scripts.flow_matching.run_ls4_native_ushcn_v1',
                        reproduction_status='downloaded_author_full_native_pipeline_with_explicit_split_branch_and_Cauchy_correction',
                        author_equivalence_certified=False,dataset='ushcn',split_branch=branch,fold=fold,
                        split_directory=split_dir,split_hashes=split_hashes,profile=profile,seed=seed,
                        source_config=source_config,source_config_sha256=sha(ROOT/source_config),
                        original_optim=native['optim'],original_model=native['model'],original_data=native['data'],
                        batch_size=native['optim']['batch_size'],epochs=native['optim']['epochs'],
                        variables=5,extrapolation=bool(native['data']['extrap']),
                        original_grouped_counts=counts,train_series=counts['train']+counts['val'],test_series=counts['test'],
                        independent_training_seeds_within_each_fold=5,folds_share_original_series=True,
                        native_source_future_samples_unused=True,interpolation_observed_target_overlap=True,
                        citations=['https://proceedings.mlr.press/v202/zhou23i.html',
                            'https://github.com/alexzhou907/ls4/tree/2b5cb76f89f6b935bbd697771433f3c74126ffca',
                            'https://github.com/edebrouwer/gru_ode_bayes/tree/ddd0b34e884dbee1c09b6a3927d1e9ab10443af8'])
                    write(ROOT/config_path,model)
                    jobs.append(dict(id=identity,split_branch=branch,fold=fold,profile=profile,seed=seed,
                        capacity_case=case_id,model_config=config_path,
                        output_directory='experiments/runs/fm_ls4_native_ushcn_v1/jobs/'+identity,
                        artifact_directory=artifact_root+'/jobs/'+identity))
                    if seed == 42:
                        cases.append(dict(id=case_id,split_branch=branch,fold=fold,profile=profile,model_config=config_path,
                                          full_train_batches=audited['full_train_batches'],full_test_batches=audited['full_test_batches']))
    paths = {r['path'] for r in loader['source_receipts']}
    paths.update(r['path'] for r in old['source_receipts'] if not r['path'].startswith('configs/models/'))
    paths.update(j['model_config'] for j in jobs)
    paths.update([loader_path,proof_path,loader['output_directory']+'/validation.json',
        'scripts/flow_matching/run_ls4_native_ushcn_v1.py','scripts/flow_matching/run_ls4_native_ushcn_queue_v1.py',
        'scripts/flow_matching/register_ls4_native_ushcn_v1.py','tests/test_fm_ls4_native_ushcn_execution_v1.py'])
    queue = dict(queue_path=queue_path,source_root=loader['source_root'],python=old['python'],
        environment_lock=loader['environment_lock'],raw_csv_path=loader['csv_path'],raw_csv_sha256=sha(ROOT/loader['csv_path']),
        loader_proof=proof_path,cauchy_correction_config=old['cauchy_correction_config'],cpu_threads=2,
        profiles=profiles,split_branches=branches,folds=list(range(5)),seeds=old['seeds'],jobs=jobs,capacity_cases=cases,
        state_root='experiments/runs/fm_ls4_native_ushcn_v1',resource_lock=old['resource_lock'],settings=old['settings'],
        integration_lock=old['integration_lock'],integration_settings=old['integration_settings'],
        integration_artifact=artifact_root+'/integration',integration_output='experiments/runs/fm_ls4_native_ushcn_v1/preflight/integration',
        gpu_preflight_artifact=artifact_root+'/gpu_preflight',gpu_preflight_output='experiments/runs/fm_ls4_native_ushcn_v1/preflight/gpu_preflight',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],
        expected_full_training_optimizer_updates=sum(500*c['full_train_batches']*len(old['seeds']) for c in cases),
        expected_full_model_training_jobs=100,expected_native_full_test_evaluations=5000,
        original_source_default_training_seed=42,original_yaml_default_fold=0,
        robustness_extension='Other4 training seeds and folds1-4 are explicitly local robustness coverage; five data folds overlap and are not25 independent training repetitions.',
        all_original_experiments_complete=False,
        boundary='Actual downloaded train_interp_extrap.main/train/eval and VAE. Original500epochs/batch64/all5 variables/full audited series; original source-literal and corrected-ID branches remain separate. Full GPU one-epoch/all-test diagnostics for all20 cases precede100 full500epoch jobs. Raw processed CSV and source never rewritten. Official conjugate Cauchy runtime correction disclosed. Original train-plus-validation-prefix normalization, ignored reserved future rows, interpolation target overlap and best-test MSE selection retained. Save actual best/final evaluated weights and pre-evaluation RNG for independent replay. Diagnostic results are not500epoch benchmark performance. All45 papers/681 literature records/273 original task records remain in scope.')
    write(ROOT/queue_path,queue)
    print(dict(jobs=len(jobs),capacity_cases=len(cases),optimizer_updates=queue['expected_full_training_optimizer_updates'],
               full_test_evaluations=5000,queue_sha256=sha(ROOT/queue_path)))


if __name__ == '__main__':
    main()
