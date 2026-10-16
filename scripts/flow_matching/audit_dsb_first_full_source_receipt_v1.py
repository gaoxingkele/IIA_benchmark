"""Verify the first completed native DSB training receipt and all moment data.

No neural imports/forwards or LP solves. Full stochastic checkpoint replay and
independent fullW1 remain prerequisites for the separate native70 completion.
"""
import argparse
import array
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching import run_dsb_singlecell_full70_v1 as dsb
from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as npy
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.audit_dsb_first_full_source_receipt_v1'
TEST='tests/test_fm_dsb_first_full_source_receipt_v1.py'
CONFIG='configs/reproducibility/fm_dsb_first_full_source_receipt.v1.json'
IDENTITY='dsb_singlecell_sourceE4__independent_public_donor13176__leave1__seed42'


def registration():
    q=read(ROOT/dsb.QUEUE);job=next(j for j in q['jobs'] if j['id']==IDENTITY);out=Path(job['artifact_directory'])
    guard=ROOT/q['state_root']/'attempts'/(out.name+'_train')/'guard_receipt.json'
    paths=[MODULE.replace('.','/')+'.py',TEST,dsb.QUEUE,dsb.MODEL,dsb.ADAPTER,dsb.COMPONENTS,npy.CONFIG,
        npy.MODULE.replace('.','/')+'.py',str(out/'source_result.json'),str(out/'phase_progress.json'),str(guard),
        'experiments/runs/flow_matching_campaign/sources/ot_cfm/original/runner/src/models/components/distribution_distances.py']
    return dict(canonical_id=IDENTITY,source_queue=dsb.QUEUE,source_job=job,source_guard=str(guard),
        state_root='experiments/runs/fm_dsb_first_full_source_receipt_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/dsb_first_full_source_receipt_v1',
        settings=read(ROOT/npy.CONFIG)['settings'],resource_lock=read(ROOT/npy.CONFIG)['resource_lock'],
        metric_tolerance=q['metric_tolerance'],original_DSB_models_required=70,available_models=60,deferred_models=10,
        original_scope=q['original_scope'],source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        citations=[read(ROOT/dsb.MODEL)['citation'],read(ROOT/dsb.MODEL)['experiment_citation'],job['data']['citation']],
        neural_forwards=0,optimizer_updates=0,independent_LP_solves=0,full_native_replay_complete=False,
        independent_full_W1_complete=False,historical_paper_equivalence=False,all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Original70-scope,full first source,unchanged moment tolerance and bounded audit guard required')
    dsb.verify(read(ROOT/dsb.QUEUE))


def verify_phases(progress):
    expected=[];counts=dict(f=0,b=0)
    for phase in range(1,21):
        for role in ['b','f']:
            counts[role]+=10001
            expected.append(dict(phase=phase,direction=role,actual_phase_updates=10001,cumulative_updates=dict(counts)))
    if (progress['phases']!=expected or progress['actual_updates']!=400040
            or progress['required_updates']!=400040 or progress['required_phases']!=40):
        raise ValueError('All40 original ordered inclusive10001-step passes and400040actual updates required')


def load_matrix(path):
    with Path(path).open('rb') as f:header,payload=npy.npy(f)
    if header['descr']!='<f4' or len(header['shape'])!=2 or header['shape'][1]!=5:
        raise ValueError('Complete original FP32 five-coordinate population required')
    data=array.array('f');data.frombytes(payload)
    if sys.byteorder!='little':data.byteswap()
    if not all(math.isfinite(x) for x in data):raise ValueError('All original prediction/truth coordinates must be finite')
    return header['shape'],data


def moments(shape,values):
    count,dimensions=shape
    if count<=0 or dimensions!=5 or len(values)!=count*dimensions:raise ValueError('Every complete population coordinate required')
    means=[];medians=[]
    for channel in range(dimensions):
        data=sorted(float(x) for x in values[channel::dimensions])
        means.append(math.fsum(data)/count)
        # Original torch.median chooses the lower middle value for evenN.
        medians.append(data[(count-1)//2])
    return means,medians


def independent_moments(prediction_shape,prediction,truth_shape,truth):
    left=moments(prediction_shape,prediction);right=moments(truth_shape,truth);values={}
    for prefix,a,b in [('Mean',left[0],right[0]),('Median',left[1],right[1])]:
        delta=[x-y for x,y in zip(a,b)];mse=math.fsum(x*x for x in delta)/5
        values.update({prefix+'_MSE':mse,prefix+'_L2':math.sqrt(mse),prefix+'_L1':math.fsum(abs(x) for x in delta)/5})
    return values


def worker(c):
    verify(c);q=read(ROOT/dsb.QUEUE);job=c['source_job'];out=Path(job['artifact_directory'])
    path=out/'source_result.json';result=read(path);guard=read(c['source_guard']);profile=read(ROOT/dsb.MODEL)
    if (not result['passed'] or not result['full_training_complete'] or result['job']!=job
            or result['queue_sha256']!=sha(ROOT/dsb.QUEUE) or result['actual_optimizer_updates']!=400040
            or result['parameters']!=28522 or not guard['passed'] or guard['exit_code']!=0
            or guard['resource_guard_aborted'] or guard['report_sha256']!=sha(path)):
        raise ValueError('Actual complete full-budget source and successful guard required')
    if Path(result['selected_checkpoint']).resolve()!= (out/'phase20_b_full_state.pt').resolve():
        raise ValueError('Only the predeclared final backward checkpoint may be selected')
    verify_phases(read(out/'phase_progress.json'))
    bindings={}
    for item in result['artifacts']:
        p=Path(item['path'])
        if not p.resolve().is_relative_to(out.resolve()) or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Every complete original trained artifact and byte must remain unchanged')
        if str(p) in bindings:raise ValueError('Duplicate trained artifact identity')
        bindings[str(p)]=item['sha256']
    for phase in range(1,21):
        for role in ['b','f']:
            if str(out/f'phase{phase}_{role}_full_state.pt') not in bindings:
                raise ValueError('All40 full model/optimizer/RNG checkpoints required')
    if sha(ROOT/job['data']['path'])!=job['data']['sha256']:raise ValueError('Complete original input bytes changed')
    prediction_shape,prediction=load_matrix(out/'prediction.npy');truth_shape,truth=load_matrix(out/'truth.npy')
    if list(prediction_shape)!=result['full_prediction_shape'] or list(truth_shape)!=result['full_truth_shape']:
        raise ValueError('Complete actual prediction and truth shapes required')
    independent=independent_moments(prediction_shape,prediction,truth_shape,truth)
    for key,value in independent.items():
        if not math.isclose(value,result['metrics'][key],rel_tol=c['metric_tolerance']['rtol'],abs_tol=c['metric_tolerance']['atol']):
            raise ValueError('Independent full-population moment metric differs: '+key)
    solves=result['LP_solves']
    if (len(solves)!=2 or [r['power'] for r in solves]!=[1,2]
            or any(r['LP_result_code']!=1 or r['full_cost_shape']!=[prediction_shape[0],truth_shape[0]] for r in solves)
            or solves[0]['objective']!=result['metrics']['1-Wasserstein']
            or math.sqrt(solves[1]['objective'])!=result['metrics']['2-Wasserstein']):
        raise ValueError('Both actual original full-population OPTIMAL LP receipts required')
    target=Path(c['artifact_root']);target.mkdir(parents=True,exist_ok=False)
    write(target/'report.json',dict(passed=True,canonical_id=IDENTITY,source_result_sha256=sha(path),source_guard_sha256=sha(c['source_guard']),
        original_input_sha256=job['data']['sha256'],all209_original_trained_artifacts_unchanged=len(bindings)==209,
        full_artifact_bindings=bindings,complete_original_IPF_passes=40,actual_source_optimizer_updates=400040,
        complete_prediction_shape=list(prediction_shape),complete_truth_shape=list(truth_shape),
        original_full_LP_cost_entries_each=prediction_shape[0]*truth_shape[0],original_source_LP_receipts=solves,
        full_original_source_metrics=result['metrics'],independent_six_moment_metrics=independent,
        independent_median_semantics='Original torch lower middle order statistic for even population size',
        mathematical_tolerance=c['metric_tolerance'],new_neural_forwards=0,new_optimizer_updates=0,new_LP_solves=0,
        full_trained_source_conditions=1,required_original_DSB_conditions=70,complete_five_seed_cohorts=0,
        full_stochastic_native_replay_complete=False,independent_full_W1_complete=False,
        historical_paper_equivalence_certified=False,strict_TAB_result=False,source_gamma_to_paper_sigma_mapping_certified=False,
        original_scope=c['original_scope'],all_original_experiments_complete=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);verify(c)
    if a.worker:worker(c);return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('First DSB full-source audit already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(v):
        v=dict(v)
        if v.get('state')=='replaying_full_checkpoints':v['state']='verifying_DSB_full_source_and_all_population_moments'
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        if (base/'worker.stdout.log').exists():raise RuntimeError('Preserve terminal audit attempt;no retry')
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
            command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='verifying_DSB_full_source',active_pid=child.pid));code,usage=capped_wait(child,c,api,update)
            proof=Path(c['artifact_root'])/'report.json';passed=code==0 and not usage['resource_guard_aborted'] and proof.is_file() and read(proof)['passed']
            write(base/'receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None))
            if not passed:raise RuntimeError('Terminal DSBsource audit failure preserved')
        update(dict(state='complete',active_pid=None,passed=True))
    finally:owner.close()


if __name__=='__main__':main()
