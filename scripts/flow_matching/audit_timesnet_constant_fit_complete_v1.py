"""Audit every saved timestamp and replay a completed local TimesNet checkpoint.

Preserve the original data, capacity, precision, batch size and evaluation.
No fitting or threshold selection is added. This does not certify the author
pipeline or the TAB training harness; full range metrics remain separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import subprocess

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api

CANONICAL = 'fm_strict_timesnet__smap__registered__seed1103__constant_fit_v2'
SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)


def validate_identity(config, queue, job, result, model, guard, guard_template):
    if (config['original_scope'] != SCOPE or config['job_id'] != CANONICAL
            or len(queue['jobs']) != 35 or len({j['id'] for j in queue['jobs']}) != 35
            or job['id'] != CANONICAL or job['seed'] != 1103 or job['dataset'] != 'smap'
            or job['window'] != 100 or job['parameter_overrides'] != dict(seed=1103, device='cpu')
            or result['job'] != job or result['id'] != job['id'] or result['status'] != 'completed'
            or result['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
            or result['parameter_count'] != 28132633 or result['paper_equivalence']
            or model['entrypoint'] != 'iia_benchmark.models.fm_legacy_window_adapter:LegacyWindowAdapter'
            or model['parameters']['method'] != 'timesnet' or model['parameters']['epochs'] != 3
            or model['parameters']['parameters'] != dict(window=100,d_model=128,d_ff=128,
                e_layers=3,top_k=3,num_kernels=6,dropout=.1,learning_rate=.0001,batch_size=128)
            or model['parameters']['seed'] != 1103 or model['parameters']['device'] != 'cpu'
            or config['cpu_threads'] != queue['evaluation']['cpu_threads'] or config['cpu_threads'] != 2
            or config['mode'] not in ('point', 'native')
            or config['settings'] != guard_template['settings']
            or config['resource_lock'] != guard_template['resource_lock']
            or guard['job_id'] != job['id'] or guard['exit_code'] != 0
            or guard['resource_guard_aborted'] or not guard['full_registered_budget_retained']):
        raise ValueError('Exact complete original model, full scope, evaluation and guard required')


def binding(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen full TimesNet audit source differs: ' + pin['path'])
    queue = read(ROOT / config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    result = read(ROOT / job['output_directory'] / 'result.json')
    model = read(ROOT / job['model_config'])
    guard = read(ROOT / config['terminal_receipt'])
    validate_identity(config, queue, job, result, model, guard, read(ROOT / config['guard_template']))
    from scripts.flow_matching.run_tsad_queue import complete_result
    if not complete_result(ROOT, job) or sha(ROOT / config['queue']) != guard['queue_sha256']:
        raise ValueError('Original completed model/score/queue hashes differ')
    for pin in job['frozen_files']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Original frozen scientific source differs')
    return queue, job, result, model


def restore_model(checkpoint, job, model_config, features, expected_parameters):
    import torch
    from iia_benchmark.models.mtsad_detectors import TimesNetDetector
    if (checkpoint['configuration'] != model_config or checkpoint['overrides'] != job['parameter_overrides']
            or checkpoint['window'] != job['window'] or set(checkpoint['modules']) != {'_model'}
            or checkpoint['adjacency'] is not None):
        raise ValueError('Exact original full checkpoint configuration required')
    params = model_config['parameters']
    detector = TimesNetDetector(**params['parameters'], epochs=params['epochs'], seed=job['seed'], device='cpu')
    detector._device = torch.device('cpu')
    detector._model = detector._build(features).to(detector._device)
    weights = checkpoint['modules']['_model']
    expected = detector._model.state_dict()
    if (set(weights) != set(expected) or any(weights[k].dtype != expected[k].dtype
            or weights[k].shape != expected[k].shape or not torch.isfinite(weights[k]).all() for k in expected)):
        raise ValueError('Every original finite weight, shape and dtype required')
    detector._model.load_state_dict(weights, strict=True)
    if sum(p.numel() for p in detector._model.parameters()) != expected_parameters:
        raise ValueError('Original full parameter count differs')
    detector._model.eval()
    return detector


def require_exact_scores(observed, expected):
    import numpy as np
    if observed.shape != expected.shape or not np.isfinite(observed).all() or not np.array_equal(observed, expected):
        raise ValueError('Complete original timestamp scores must match bitwise')


def worker(config, config_path):
    import numpy as np
    import torch
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved, covering_windows, align_scores
    from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays
    queue, job, result, model_config = binding(config)
    environment = dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,device='cpu')
    if environment != result['environment']:
        raise ValueError('Exact original numerical environment required')
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    manifest = read(ROOT / job['dataset_manifest_path'])
    dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    if not dataset['grouped_splits_verified'] or sha(ROOT / dataset['input_npz']) != dataset['input_sha256']:
        raise ValueError('Full original grouped prepared data required')
    for pin in dataset['source_files']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Original raw data differs')
    source = ROOT / job['output_directory']
    target = ROOT / config['output_directory']
    target.mkdir(parents=True,exist_ok=False)
    checkpoint = torch.load(source / 'model.pt',map_location='cpu',weights_only=True)
    features = {e['features'] for e in dataset['entities']}
    if len(features) != 1:
        raise ValueError('Full original model feature geometry required')
    detector = restore_model(checkpoint,job,model_config,features.pop(),result['parameter_count'])
    payload, coverage = {}, []
    with np.load(ROOT / dataset['input_npz'],allow_pickle=False) as inputs, np.load(source / 'scores.npz',allow_pickle=False) as saved:
        entities,lengths,validation = check_arrays(inputs,saved,dataset)
        training = np.concatenate([saved[e['array_prefix']+'_train'] for e in dataset['entities']])
        metrics = evaluate_saved(entities,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],queue['evaluation'],job['seed'])
        if metrics != result['metrics']:
            raise ValueError('All original point/core/event/bootstrap metrics must match exactly')
        truth = np.concatenate([pair[0] for pair in entities.values()]).astype(bool)
        scores = np.concatenate([pair[1] for pair in entities.values()])
        for ratio, record in metrics['strict'].items():
            threshold = float(np.percentile(validation,100-float(ratio),method='linear'))
            if threshold != record['calibration']['threshold']:
                raise ValueError('Only full validation scores may calibrate the original threshold')
            p,r,f,_ = precision_recall_fscore_support(truth,scores>threshold,average='binary',zero_division=0)
            independent = dict(precision=float(p),recall=float(r),f1=float(f),
                auroc=float(roc_auc_score(truth,scores)),average_precision=float(average_precision_score(truth,scores)))
            if any(not math.isclose(independent[k],record['micro'][k],rel_tol=1e-12,abs_tol=1e-12) for k in independent):
                raise ValueError('Independent primary metric implementation disagrees')
        for entry in dataset['entities']:
            prefix = entry['array_prefix']
            for split in ('train','validation','test'):
                key = prefix+'_'+split
                values = inputs[key]
                if not np.isfinite(values).all():
                    raise ValueError('Full prepared inputs must be finite')
                windows,starts = covering_windows(values,job['window'])
                if config['mode'] == 'native':
                    observed = align_scores(detector.score(windows),starts,len(values),job['window'])
                    require_exact_scores(observed,saved[key])
                    payload[key] = observed
                coverage.append(dict(entity=entry['entity_id'],split=split,points=len(values),windows=len(windows)))
            if config['mode'] == 'native':
                payload[prefix+'_labels'] = inputs[prefix+'_labels']
            write(target/'progress.json',dict(full_entities_finished=len(coverage)//3,required=len(dataset['entities']),mode=config['mode']))
    if config['mode'] == 'native':
        np.savez_compressed(target/'replayed_scores.npz',**payload)
    proof = dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),canonical_id=job['id'],
        configuration=config_path,configuration_sha256=sha(ROOT/config_path),mode=config['mode'],
        original_scope=config['original_scope'],original_queue_jobs=35,original_data_control_jobs=291,
        configured_training_epochs=3,persisted_training_epoch_update_counter_available=False,
        original_full_registered_budget_receipt_passed=True,parameters=result['parameter_count'],
        full_entities=len(lengths),full_test_points=sum(lengths.values()),full_validation_points=len(validation),
        full_train_points=len(training),coverage=coverage,all_saved_streams_verified=True,
        independently_recomputed_primary_metrics=metrics['strict']['1']['micro'],
        conditional_entity_cluster_f1_ci95=metrics['strict']['1']['f1_ci95'],
        original_event_metrics=metrics['strict']['1']['event_metrics'],
        original_test_calibrated_TAB_core_diagnostic=metrics['tab_core_diagnostic'],
        all_original_point_core_event_bootstrap_metrics_exact=True,
        full_native_checkpoint_reinference_passed=config['mode']=='native',
        all_replayed_timestamp_scores_bitwise_equal=config['mode']=='native',
        replay_scores_sha256=sha(target/'replayed_scores.npz') if config['mode']=='native' else None,
        original_checkpoint_sha256=result['checkpoint_sha256'],original_scores_sha256=result['scores_sha256'],
        original_result_sha256=sha(source/'result.json'),additional_training_seed_slots=0,additional_optimizer_updates=0,
        historical_author_equivalence_certified=False,strict_TAB_training_harness_certified=False,
        full_TAB_range_metrics_completed=False,all_original_experiments_complete=False)
    write(target/'independent_full_saved_score_verification.json',proof)
    print(json.dumps(dict(passed=True,mode=config['mode'],entities=len(lengths),test_points=sum(lengths.values()))))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/args.config)
    if args.worker:
        worker(config,args.config)
        return
    binding(config)
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    api=resource_api();lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:
        raise RuntimeError('This full audit already has a live owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        if (base/'receipt.json').exists():
            raise FileExistsError('Preserve successful and failed attempts')
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.audit_timesnet_constant_fit_complete_v1','--config',args.config,'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=subprocess.CREATE_NO_WINDOW)
                p=psutil.Process(child.pid)
                update(dict(active_pid=p.pid,active_create_time=p.create_time(),actual_command=p.cmdline()))
                code,usage=capped_wait(child,config,api,update)
            proof_path=ROOT/config['output_directory']/'independent_full_saved_score_verification.json'
            passed=code==0 and not usage['resource_guard_aborted'] and proof_path.exists() and read(proof_path)['passed']
            write(base/'receipt.json',dict(exit_code=code,full_saved_score_review_passed=passed,mode=config['mode'],**usage))
            update(dict(state='completed' if passed else 'failed_preserved',active_pid=None))
            if not passed:
                raise RuntimeError('Full TimesNet audit failed; preserve the actual attempt')
    finally:
        lane.close()


if __name__=='__main__':
    main()
