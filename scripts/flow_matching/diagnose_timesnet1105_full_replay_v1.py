"""Complete all51 replay streams after the preserved strict bitwise failure.

This is a separate diagnostic, never a replacement/relaxation of the original
native certificate. Original checkpoint, score code, batches and gates stay.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from scripts.flow_matching import continue_timesnet_remaining4_cold_v3 as old
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic,capped_wait

ROOT,read,sha=old.ROOT,old.read,old.sha
MODULE='scripts.flow_matching.diagnose_timesnet1105_full_replay_v1'
CONFIG='configs/reproducibility/fm_timesnet1105_full_replay_diagnostic.2026-10-16_v1.json'
TEST='tests/test_fm_timesnet1105_full_replay_diagnostic_v1.py'
ERROR='ValueError: Complete original timestamp scores must match bitwise'


def preserved_failure(stage):
    attempt=ROOT/stage['state_root'];receipt=read(attempt/'receipt.json')
    error=(attempt/'stderr.log').read_text(encoding='utf-8')
    if (receipt['exit_code']!=1 or receipt['passed'] or receipt['resource_guard_aborted']
            or receipt['proof_sha256'] is not None or ERROR not in error
            or (Path(stage['output_directory'])/'independent_full_saved_score_verification.json').exists()):
        raise ValueError('Only the exact preserved terminal bitwise mismatch is admitted')
    return dict(receipt=str(attempt/'receipt.json'),receipt_sha256=sha(attempt/'receipt.json'),
        stderr=str(attempt/'stderr.log'),stderr_sha256=sha(attempt/'stderr.log'),
        strict_failure_remains=True)


def registration():
    c=read(ROOT/old.original.CONFIG);jobs=old.original.verify(c)
    job=jobs[old.original.identity(1105)];stage=old.original.stage_config(c,job,'native')
    point=old.checked_stage(c,job,'point');failure=preserved_failure(stage)
    source=ROOT/job['output_directory']
    pins={old.original.CONFIG,old.CONFIG,old.MODULE.replace('.','/')+'.py',MODULE.replace('.','/')+'.py',TEST}
    pins.update([str(source/name) for name in ['model.pt','scores.npz','result.json']])
    pins.update([failure['receipt'],failure['stderr'],point['proof']])
    pins.update(p['path'] for p in c['source_receipts'])
    return dict(original_configuration=old.original.CONFIG,original_stage=stage,
        canonical_id=job['id'],point_certificate=point,preserved_failure=failure,
        state_root='experiments/runs/fm_timesnet1105_full_replay_diagnostic_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/timesnet1105_full_replay_diagnostic_v1',
        settings=c['settings'],resource_lock=c['resource_lock'],python=c['python'],cpu_threads=2,
        original_scope=c['original_scope'],required_entities=51,
        required_train_points=101565,required_validation_points=26684,required_test_points=408160,
        full_original_batch_size=128,full_original_parameters=28132633,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        citation=c['paper_citation'],data_citation=c['data_citation'],
        diagnostic_completion_is_not_native_equivalence=True,original_tolerance_changed=False,
        new_training_updates=0,all_original_experiments_complete=False,
        boundary='Separate complete all51/train-validation-test checkpoint replay diagnostic. '
          'Preserve exact original failed receipt/error/empty output,successful point audit and all immutable sources. '
          'Same original28132633parameter CPU float32 model,2threads,batch128,window100,all full points/labels, '
          'original resource gates/caps/lock. Save every original-score comparison and all raw-window/aligned scores; '
          'compare every mismatch explicitly,not by a widened tolerance. Original bitwise failure never becomes a pass. '
          'Full original saved/replayed metrics plus fixed-original-validation-threshold controls;no training or retry credit. '
          'No backend-cause proof,historical deterministic replay,author or TAB certification;all research scope retained.')


def verify(config):
    if config!=registration():raise ValueError('Exact original full replay geometry/resources and preserved failure required')
    return config['original_stage']


def compare_scores(observed,expected):
    import numpy as np
    if observed.shape!=expected.shape or not np.isfinite(observed).all() or not np.isfinite(expected).all():
        raise ValueError('Every complete finite original score coordinate is required')
    difference=np.asarray(observed,dtype=np.float64)-np.asarray(expected,dtype=np.float64)
    unequal=observed!=expected
    return dict(points=int(observed.size),scores_exactly_equal=bool(np.array_equal(observed,expected)),
        changed_points=int(np.count_nonzero(unequal)),maximum_absolute_difference=float(np.max(np.abs(difference))),
        mean_absolute_difference=float(np.mean(np.abs(difference))),
        RMS_difference=float(np.sqrt(np.mean(difference*difference))))


def fixed_threshold_counts(labels,scores,threshold):
    import numpy as np
    if labels.shape!=scores.shape or not np.isin(labels,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Full aligned finite scores and binary labels required')
    truth=labels.astype(bool);prediction=scores>threshold
    tp=int(np.count_nonzero(truth&prediction));fp=int(np.count_nonzero(~truth&prediction));fn=int(np.count_nonzero(truth&~prediction))
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0,
        recall=tp/(tp+fn) if tp+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def worker(config):
    stage=verify(config)
    import numpy as np
    import torch
    from scripts.flow_matching.run_tsad_experiment import covering_windows,align_scores,evaluate_saved
    from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays
    queue,job,result,model_config=old.original.binding(stage)
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,device='cpu')
    if environment!=result['environment']:raise ValueError('Original numerical environment must match')
    torch.set_num_threads(config['cpu_threads']);torch.set_num_interop_threads(1)
    manifest=read(ROOT/job['dataset_manifest_path']);dataset=next(d for d in manifest['datasets'] if d['id']==job['dataset'])
    source=ROOT/job['output_directory'];target=Path(config['artifact_root']);target.mkdir(parents=True,exist_ok=False)
    cp=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    features={e['features'] for e in dataset['entities']};assert len(features)==1
    detector=old.original.original.restore_model(cp,job,model_config,features.pop(),result['parameter_count'])
    def state_digest():
        h=hashlib.sha256()
        for name,value in detector._model.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    weights_before=state_digest();payload={};raw_payload={};comparisons=[]
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs,np.load(source/'scores.npz',allow_pickle=False) as saved:
        entities,lengths,validation=check_arrays(inputs,saved,dataset)
        training=np.concatenate([saved[e['array_prefix']+'_train'] for e in dataset['entities']])
        original_metrics=evaluate_saved(entities,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],queue['evaluation'],job['seed'])
        if original_metrics!=result['metrics']:raise ValueError('Every original saved metric must remain exact')
        for entry in dataset['entities']:
            prefix=entry['array_prefix']
            for split in ['train','validation','test']:
                key=prefix+'_'+split;values=inputs[key]
                if not np.isfinite(values).all():raise ValueError('All original input points must be finite')
                windows,starts=covering_windows(values,job['window'])
                raw=detector.score(windows);aligned=align_scores(raw,starts,len(values),job['window'])
                comparison=compare_scores(aligned,saved[key])
                comparisons.append(dict(entity=entry['entity_id'],split=split,key=key,windows=len(windows),
                    original_input_strides=list(values.strides),original_input_C_contiguous=bool(values.flags.c_contiguous),**comparison))
                payload[key]=aligned;raw_payload[key+'__window_scores']=raw;raw_payload[key+'__window_starts']=starts
            payload[prefix+'_labels']=inputs[prefix+'_labels']
            atomic(target/'progress.json',dict(full_entities_finished=len(comparisons)//3,required_entities=51,
                changed_streams=sum(not x['scores_exactly_equal'] for x in comparisons),strict_failure_preserved=True))
        np.savez_compressed(target/'aligned_scores.npz',**payload)
        np.savez_compressed(target/'raw_window_scores_and_starts.npz',**raw_payload)
        with np.load(target/'aligned_scores.npz',allow_pickle=False) as replay:
            replay_entities,replay_lengths,replay_validation=check_arrays(inputs,replay,dataset)
            replay_training=np.concatenate([replay[e['array_prefix']+'_train'] for e in dataset['entities']])
            replay_metrics=evaluate_saved(replay_entities,replay_validation,replay_training,dataset['input_sha256'],result['checkpoint_sha256'],queue['evaluation'],job['seed'])
            y=np.concatenate([x[0] for x in entities.values()]);old_score=np.concatenate([x[1] for x in entities.values()])
            new_score=np.concatenate([x[1] for x in replay_entities.values()])
            controls={}
            for ratio,record in original_metrics['strict'].items():
                threshold=record['calibration']['threshold']
                controls[ratio]=dict(original_threshold=threshold,original=fixed_threshold_counts(y,old_score,threshold),
                    replay_at_same_original_threshold=fixed_threshold_counts(y,new_score,threshold),
                    prediction_disagreements=int(np.count_nonzero((old_score>threshold)!=(new_score>threshold))))
    assert len(comparisons)==153 and len(lengths)==51 and sum(lengths.values())==408160 and len(validation)==26684 and len(training)==101565
    assert state_digest()==weights_before
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Original immutable source or preserved failure changed')
    report=dict(diagnostic_execution_complete=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG,configuration_sha256=sha(ROOT/CONFIG),canonical_id=job['id'],environment=environment,
        full_entities=51,full_streams=153,full_train_points=101565,full_validation_points=26684,full_test_points=408160,
        streams=comparisons,all_this_replay_scores_bitwise_equal=all(x['scores_exactly_equal'] for x in comparisons),
        strict_historical_failure_preserved=True,strict_native_reproduction_certified=False,
        original_saved_metrics=original_metrics,replayed_metrics=replay_metrics,fixed_original_threshold_controls=controls,
        original_model_state_unchanged=True,model_state_sha256=weights_before,
        original_failure=config['preserved_failure'],
        arrays=[dict(path=str(target/name),sha256=sha(target/name)) for name in ['aligned_scores.npz','raw_window_scores_and_starts.npz']],
        new_optimizer_updates=0,new_training_seed_slots=0,numerical_backend_cause_proven=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
    atomic(target/'full_replay_diagnostic.json',report)
    print(json.dumps(dict(diagnostic_complete=True,all_streams=153,changed_streams=sum(not x['scores_exactly_equal'] for x in comparisons),strict_native_reproduction_certified=False)))


def controller(config):
    import psutil
    verify(config)
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('Diagnostic metadata already owned')
    if (base/'status.json').exists():lane.close();raise FileExistsError('Preserve every diagnostic attempt')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),active_pid=None,
        canonical_id=config['canonical_id'],all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {},**kw,captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    try:
        update(state='waiting_original_full_replay_resources_for_separate_diagnostic')
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            verify(config);command=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=subprocess.CREATE_NO_WINDOW)
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                atomic(base/'worker_launch.json',handle);update(state='full_original_replay_diagnostic_running',active_pid=child.pid,exact_child_handle=handle)
                code,usage=capped_wait(child,config,api,lambda value:update(value))
            report=Path(config['artifact_root'])/'full_replay_diagnostic.json'
            complete=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['diagnostic_execution_complete']
            atomic(base/'guard_receipt.json',dict(exit_code=code,diagnostic_execution_complete=complete,
                strict_native_reproduction_certified=False,**usage,exact_child_handle=handle,report_sha256=sha(report) if report.exists() else None))
            update(state='diagnostic_complete_original_strict_failure_preserved' if complete else 'diagnostic_failure_preserved',active_pid=None)
    finally:lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.register:old.exclusive(ROOT/CONFIG,registration());return
    config=read(ROOT/CONFIG)
    if args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
