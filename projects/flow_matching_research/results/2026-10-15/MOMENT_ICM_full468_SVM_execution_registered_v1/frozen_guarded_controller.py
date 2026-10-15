"""Guarded468 Tiny/ICM classifier cases after actual full PILE training.

All117 datasets x2 final trained models x2 explicit preprocessing tracks.
Periodic source AD callbacks and all forecasting tasks remain separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.flow_matching import moment_icm_svm_classification_v1 as science

ROOT, read, sha, write = science.ROOT, science.read, science.sha, science.write
MODULE = 'scripts.flow_matching.run_moment_icm_full_classification_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_full117_two_models_two_tracks_SVM.2026-10-15_v1.json'
MODEL = 'configs/models/fm_moment_ICM_full_PILE_SVM_classification.v1.json'
TEST = 'tests/test_fm_moment_icm_full_classification_v1.py'
POLICY = 'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'


def registration():
    source = read(ROOT/science.strict.PARENT)
    science.strict.source.verify(source)
    science.strict.verify(read(ROOT/science.strict.CONFIG))
    science.pretrain.verify(read(ROOT/science.pretrain.CONFIG))
    policy = read(ROOT/POLICY)
    jobs = [dict(id='moment_tiny_'+variant+'__'+track+'__'+j['dataset']+'__seed13', variant=variant,
                 track=track, dataset=j['dataset'], seed=13, checkpoint='full_final_post_scheduler_39479')
            for variant in ['independent', 'icm'] for track in ['source_TEST_fitted', 'TRAIN_fitted_control'] for j in source['jobs']]
    files = [MODULE.replace('.', '/')+'.py', 'scripts/flow_matching/moment_icm_svm_classification_v1.py', TEST,
             MODEL, POLICY, science.pretrain.CONFIG, science.strict.PARENT, science.strict.CONFIG, science.SVM_SOURCE,
             'src/iia_benchmark/models/fm_supplemental_moment.py', 'scripts/flow_matching/moment_icm_masked_training_v1.py']
    return dict(schema_version=1, paper_citation='https://arxiv.org/abs/2409.13530v1',
        related_source_citation='https://github.com/moment-timeseries-foundation-model/moment-research/tree/3ab637e413f35f2c317573c0ace280d825c558de',
        data_citation='https://www.timeseriesclassification.com/', model_configuration=MODEL,
        jobs=jobs, required_classifier_cases=468, original_classification_datasets=117,
        original_model_dataset_cases=234, separate_normalization_control_cases=234,
        original_paper_scalar_targets=702, original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        python=read(ROOT/science.pretrain.CONFIG)['python'], settings=policy['settings'], resource_lock=policy['resource_lock'],
        pooling_tolerance=dict(rtol=2e-6, atol=2e-6), SVM_source_sha256=sha(ROOT/science.SVM_SOURCE),
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_moment_icm_full117_two_models_two_tracks_SVM_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_full117_two_models_two_tracks_SVM_v1',
        boundary='Wait actual full2086source/80850189item/39479update final model plus guarded data proofs. '
            'CPUfloat32 casebatch1 retains all channels simultaneously;source patch-mask weighted temporal mean '
            'per channel then flatten Cx256 is a disclosed reconstruction,not unpublished author ICM certification. '
            'Original SVM grid/CV5/n_jobs10/10000 source subsample and positive-infinite C candidate retained, '
            'with any failed candidate warning recorded. No test-label tuning. Conditional Wilson intervals '
            'do not establish subject-group robustness or model-seed uncertainty;not strict leaderboard. '
            'Source periodic evaluations call AD,so this classifier never clears those pending obligations. '
            'All forecast/context baselines and remaining original tasks retained;no automatic retry or smaller substitutes.')


def verify(config):
    if config != registration():
        raise ValueError('All468 frozen cases,full trained weights and original numerical/resource policy required')
    return {j['id']: j for j in config['jobs']}


def prerequisites(config, job):
    trained = science.trained_checkpoint(config, job['variant'])
    if trained is None:
        return False
    return science.dataset_binding(config, job['dataset'], job['track']) is not None


def controller(config, jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); root = ROOT/config['state_root']; root.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](root/'controller.lock')
    if lane is None: raise RuntimeError('Full classification owner already live')
    completed, failed = [], {}
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_classifier_cases=468,
                 original_model_dataset_cases=234, separate_control_cases=234, all_original_experiments_complete=False)
    def update(value=None, **kw):
        state.update(value or {}); state.update(kw)
        state.update(completed_ids=list(completed), completed_count=len(completed), failed_preserved=dict(failed),
                     captured_utc=datetime.now(timezone.utc).isoformat())
        write(root/'status.json', state)
    def stage(job, phase):
        identity = job['id']; attempt = root/'attempts'/identity/phase
        output = Path(config['artifact_root'])/identity
        proof = output/('native_classification.json' if phase == 'native' else 'independent_review.json')
        receipt = attempt/'guard_receipt.json'
        if receipt.exists():
            previous = read(receipt)
            if previous['passed'] and previous['exit_code'] == 0 and not previous['resource_guard_aborted'] and sha(proof) == previous['proof_sha256']:
                science.verify_existing_evidence(config, job, output, phase)
                return True
            raise ValueError('Failed full classifier attempt preserved;no automatic retry')
        if attempt.exists() or proof.exists() or (phase == 'native' and output.exists()):
            raise ValueError('Partial classifier attempt preserved')
        attempt.mkdir(parents=True, exist_ok=False)
        command = [str(ROOT/config['python']), '-B', '-X', 'utf8', '-u', '-m', MODULE, '--stage', phase, '--job-id', identity]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='13', CUDA_VISIBLE_DEVICES='-1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'))
        with (attempt/'stdout.log').open('xb') as out, (attempt/'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=environment,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
            write(attempt/'launch.json', handle)
            update(state='running_full_classifier_'+phase, active_pid=child.pid, active_create_time=handle['create_time'], job=identity)
            code, usage = capped_wait(child, config, api, update)
        passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
        write(receipt, dict(passed=passed, exit_code=code, **usage, proof_sha256=sha(proof) if proof.exists() else None, exact_child_handle=handle))
        return passed
    try:
        while len(completed)+len(failed) < len(jobs):
            available_models = {v: science.trained_checkpoint(config, v) for v in ['independent', 'icm']}
            for identity, job in jobs.items():
                if identity in completed or identity in failed: continue
                try:
                    if available_models[job['variant']] is None: continue
                    if science.dataset_binding(config, job['dataset'], job['track']) is None: continue
                    def waiting(value): update(dict(value, active_pid=None, job=identity))
                    with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], waiting, poll_seconds=10):
                        verify(read(ROOT/CONFIG))
                        if not prerequisites(config, job): raise ValueError('Full actual scientific prerequisites disappeared')
                        if not stage(job, 'native') or not stage(job, 'independent'):
                            raise ValueError('Full guarded classifier stage failed;preserve')
                    completed.append(identity)
                except Exception as error: failed[identity] = str(error)
                update(state='scanning_full_original_classifier_cases', active_pid=None, job=None)
            update(state='waiting_actual_full_PILE_models_and_complete_preprocessing', active_pid=None, job=None)
            time.sleep(10)
        update(state='all468_classifier_attempts_terminal_other_original_experiments_remain', active_pid=None, job=None)
    finally: lane.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--stage', choices=['native', 'independent']); parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists(): raise FileExistsError('Preserve frozen full classifier registration')
        config = registration(); verify(config)
        with (ROOT/CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(configuration=CONFIG, classifier_cases=468, original_datasets=117)))
        return
    config = read(ROOT/CONFIG); jobs = verify(config); config['_config_path'] = CONFIG
    if args.probe_only:
        models = {v: science.trained_checkpoint(config, v) is not None for v in ['independent', 'icm']}
        print(json.dumps(dict(passed=True, classifier_cases=468, actually_ready_full_models=models,
                             numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
                             classifier_cases_completed=0, all_original_experiments_complete=False)))
    elif args.stage:
        job = jobs[args.job_id]
        if not prerequisites(config, job): raise ValueError('Actual complete scientific prerequisites required before imports')
        output = Path(config['artifact_root'])/job['id']
        (science.run_case if args.stage == 'native' else science.review_case)(config, job, output)
    else: controller(config, jobs)


if __name__ == '__main__': main()
