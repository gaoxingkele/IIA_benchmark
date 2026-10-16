"""Full released TrajectoryNet checkpoint replay with explicit W2/W1 semantics."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.replay_TrajectoryNet_author_fig8_full_v1'
CONFIG='configs/reproducibility/fm_TrajectoryNet_author_fig8_full_checkpoint_replay.v1.json'
MODEL='configs/models/fm_TrajectoryNet_author_fig8_EB_leave3_replay.v1.json'
HELPER='src/iia_benchmark/models/trajectorynet_author_replay_v1.py'
ACQUISITION='configs/acquisition/fm_TrajectoryNet_author_checkpoint_materials.v1.json'
CFM_REFERENCE='configs/reproducibility/fm_otcfm_EB100_full90_checkpoint_replay.v1.json'


def registration():
    acquisition=read(ROOT/ACQUISITION);profile=read(ROOT/MODEL)
    material=read(Path(acquisition['output_directory'])/'verified_acquisition.json')
    reference=read(ROOT/CFM_REFERENCE)
    identifier=reference['required_case_ids'][0]
    ref=Path(reference['artifact_root'])/hashlib.sha256(identifier.encode()).hexdigest()[:20]
    files=[MODULE.replace('.','/')+'.py',MODEL,HELPER,ACQUISITION,CFM_REFERENCE,
           'tests/test_fm_trajectorynet_author_replay_v1.py','tests/test_fm_trajectorynet_full_replay_contract_v1.py',
           str(Path(acquisition['output_directory'])/'verified_acquisition.json'),
           str(ref/'full_checkpoint_replay.json'),str(ref/'t3_truth.npy')]
    files += [(Path(acquisition['source_root'])/r['path']).as_posix() for r in material['source_files']]
    data=read(ROOT/'configs/acquisition/fm_TrajectoryNet_EB100_official.v1.json')
    original=Path(data['output_directory'])/data['filename']
    return dict(model=MODEL,source_root=acquisition['source_root'],material_receipt=str(Path(acquisition['output_directory'])/'verified_acquisition.json'),
        original_checkpoint=str(ROOT/acquisition['source_root']/acquisition['source_checkpoint']),
        author_reference_trajectory=str(ROOT/acquisition['source_root']/acquisition['source_reference_trajectory']),
        overlay_directory=material['isolated_import_directory'],original_data=str(original),
        original_data_sha256='17623d16dd4fe7b679aad80916d130cb2863a4cd4ee742f139a3c2eed91ba230',
        matched_CFM_truth_reference=str(ref/'t3_truth.npy'),matched_reference_role='data_identity_only_not_leaveout3_model_performance',
        full_population_counts={'0':2381,'1':4163,'2':3278,'3':3665,'4':3332},
        full_input_cells=16819,original_PC_dimensions=100,native_dimensions=5,leftout_timepoint=3,
        expected_parameters=9621,author_training_seed=None,local_training_runs_added=0,
        author_original_iterations_logged=10000,selected_checkpoint_step_certified=False,
        native_EMD_kind=profile['original_EMD_kind'],matched_metric=profile['matched_additional_metric'],
        fixed_primary_direction=profile['primary_direction'],historical_reference_tolerance=profile['historical_author_saved_trajectory_tolerance'],
        python=acquisition['python'],numerical_threads=reference['numerical_threads'],settings=reference['settings'],resource_lock=reference['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_TrajectoryNet_author_fig8_full_checkpoint_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/TrajectoryNet_author_fig8_full_checkpoint_replay_v1',
        original_scope=reference['original_scope'],paper_baseline_reproduced=False,all_original_experiments_complete=False,
        boundary=profile['boundary'])


def verify(c):
    if c!=registration():raise ValueError('Every full original source/checkpoint/input/metric/guard binding required')
    if sha(Path(c['original_data']))!=c['original_data_sha256']:raise ValueError('Complete original100PCdata changed')
    material=read(Path(c['material_receipt']))
    for row in material['installed_files']:
        if sha(Path(c['overlay_directory'])/row['path'])!=row['sha256']:raise ValueError('Original0.0.1solver dependency changed')


def worker(c):
    import numpy as np
    spec=importlib.util.spec_from_file_location('_author_TrajectoryNet_replay_helper',ROOT/HELPER)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    profile=read(ROOT/MODEL)
    torch,args,model=helper.native_model(ROOT/c['source_root'],c['overlay_directory'],c['original_checkpoint'],profile['source_parameters'])
    torch.set_num_threads(c['numerical_threads']);torch.manual_seed(profile['replay_rng_seed'])
    data=helper.native_EB_data(ROOT/c['source_root'],c['original_data'])
    labels=data.get_times();values=data.get_data()
    counts={str(t):int(np.sum(labels==t)) for t in data.get_unique_times()}
    if counts!=c['full_population_counts'] or values.shape!=(16819,5):raise ValueError('All16819cells/5PCs/fivetimepoints required')
    out=Path(c['artifact_root']);out.mkdir(parents=True,exist_ok=False)
    args.save=str(out);args.data=data;args.timepoints=data.get_unique_times()
    args.int_tps=(np.arange(max(args.timepoints)+1)+1.)*args.time_scale
    args.embedding_name='pcs'
    model.train(profile['native_module_training_flag'])
    evaluation=importlib.import_module('TrajectoryNet.eval')
    evaluation.args=args
    reference=np.load(c['author_reference_trajectory'],mmap_mode='r',allow_pickle=False)
    end=values[labels==4]
    if reference.shape!=(100,3332,5) or not np.array_equal(reference[0],end.astype(np.float32)):
        raise ValueError('Entire saved reference initial population/order/normalization must match')
    truth32=values[labels==3].astype(np.float32)
    if not np.array_equal(truth32,np.load(c['matched_CFM_truth_reference'],allow_pickle=False)):
        raise ValueError('All3665x5targetcoordinates must match the existingCFMinput exactly')
    evaluation.integrate_backwards(end,model,str(out),ntimes=100,device='cpu')
    replay=np.load(out/'backward_trajectories.npy',mmap_mode='r',allow_pickle=False)
    if replay.shape!=reference.shape or not np.isfinite(replay).all():raise ValueError('Full100frame3332cell native replay required')
    tolerance=c['historical_reference_tolerance']
    maximum=float(np.max(np.abs(replay-reference)))
    cached_close=bool(np.allclose(replay,reference,**tolerance))
    write(ROOT/c['state_root']/'progress.json',dict(state='full_saved100frame_replay_complete_computing_all_native_metrics',
        reference_allclose=cached_close,reference_maximum_absolute_difference=maximum,optimizer_updates=0))
    native_eval=importlib.import_module('TrajectoryNet.eval_utils')
    emd=importlib.import_module('TrajectoryNet.optimal_transport.emd')
    original_EMD=native_eval.earth_mover_distance;original_LP=emd.pot.emd2
    clouds=[];native_solves=[]
    def observed_LP(a,b,costs,**kwargs):
        if kwargs.get('log'):raise ValueError('Original native caller must remain unlogged')
        result,log=original_LP(a,b,costs,log=True,**kwargs)
        if int(log['result_code'])!=1 or log['warning'] is not None:raise ValueError('Full original squaredcostLP must reach OPTIMAL')
        native_solves.append(dict(full_cost_shape=list(costs.shape),dtype=str(costs.dtype),
            squared_cost_objective=float(result),reported_native_W2=math.sqrt(float(result)),LP_result_code=1))
        return (result,log) if kwargs.get('return_matrix') else result
    def observed_EMD(p,q,*a,**kw):
        clouds.append((np.array(p,copy=True),np.array(q,copy=True)))
        return original_EMD(p,q,*a,**kw)
    emd.pot.emd2=observed_LP;native_eval.earth_mover_distance=observed_EMD
    try:native_W2=native_eval.evaluate_kantorovich_v2(torch.device('cpu'),args,model)
    finally:emd.pot.emd2=original_LP;native_eval.earth_mover_distance=original_EMD
    if len(clouds)!=2 or len(native_solves)!=2 or native_W2.shape!=(2,):raise ValueError('Bothcomplete native backward/forward evaluations required')
    from scipy.spatial.distance import cdist
    matched=[]
    for index,(prediction,truth64) in enumerate(clouds):
        expected=3332 if index==0 else 3278
        if prediction.shape!=(expected,5) or truth64.shape!=(3665,5) or not np.array_equal(truth64.astype(np.float32),truth32):
            raise ValueError('Full input populations and unchanged full leftouttruth required')
        costs=torch.cdist(torch.from_numpy(prediction),torch.from_numpy(truth32)).numpy()
        weights1=emd.pot.unif(len(prediction));weights2=emd.pot.unif(len(truth32))
        value,log=original_LP(weights1,weights2,costs,numItermax=1e7,log=True)
        if int(log['result_code'])!=1 or log['warning'] is not None:raise ValueError('CFM-style full rawEuclidean W1 must reach OPTIMAL')
        independent=cdist(prediction,truth32,metric='euclidean')
        error=max(float(np.max(np.abs(independent[i:i+256]-costs[i:i+256]))) for i in range(0,len(prediction),256))
        other,proof=original_LP(weights1,weights2,independent,numItermax=1e7,log=True)
        if int(proof['result_code'])!=1 or proof['warning'] is not None:raise ValueError('Independent fullfloat64W1 must reach OPTIMAL')
        difference=abs(float(value)-float(other));bound=error+1e-8*max(1.,abs(float(value)),abs(float(other)))
        if difference>bound:raise ValueError('Independent W1 differs beyond all-entry cost perturbation bound')
        np.save(out/('backward_prediction.npy' if index==0 else 'forward_prediction.npy'),prediction,allow_pickle=False)
        matched.append(dict(direction='from_future' if index==0 else 'from_past',full_cost_shape=list(costs.shape),
            target_dtype='float32_exact_CFM_truth_identity',W1=float(value),independent_float64_W1=float(other),
            native_LP_result_code=1,independent_LP_result_code=1,all_cost_entries_checked=int(costs.size),
            maximum_cost_entry_difference=error,objective_difference=difference,cost_perturbation_bound=bound))
        del costs,independent
    np.save(out/'matched_truth.npy',truth32,allow_pickle=False)
    exclusive(out/'full_author_checkpoint_replay.json',dict(passed=True,checkpoint_sha256=sha(Path(c['original_checkpoint'])),
        configuration_sha256=sha(ROOT/CONFIG),source_commit=profile['source_commit'],
        full_population_counts=counts,parameters=sum(p.numel() for p in model.parameters()),
        saved_reference_shape=list(reference.shape),saved_reference_allclose=cached_close,
        saved_reference_bitwise_equal=bool(np.array_equal(replay,reference)),saved_reference_maximum_absolute_difference=maximum,
        fixed_historical_reference_tolerance=tolerance,native_author_metric_name='W2_default_EMD',
        native_author_W2_backward_forward=list(map(float,native_W2)),native_author_LP_proofs=native_solves,
        additional_matched_W1=matched,fixed_primary_forward_W1=matched[1]['W1'],
        matched_CFM_truth_reference_role=c['matched_reference_role'],all_source_target_float32_coordinates_equal=True,
        raw_author_metric_target_dtype='float64',author_training_seed=None,local_training_runs_added=0,
        selected_checkpoint_training_iteration_certified=False,logged_variant='Base+D',
        reported_0_848_variant='Base+D+V_mean_over_three_leaveouts_not_this_checkpoint',
        source_evaluation_early_exit_bypassed_by_calling_unchanged_functions=True,
        original_source_files_modified=False,torch_version=torch.__version__,
        torchdiffeq_import_directory=c['overlay_directory'],torchdiffeq_runtime_version='0.0.1',
        historical_environment_equivalence_certified=False,paper_full_baseline_reproduced=False,
        strict_TAB_result=False,all_original_experiments_complete=False))


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Preserve everyoriginalauthor-checkpoint replay attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
               additional_training_seed_slots=0,all_original_experiments_complete=False)
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
            verify(c)
            command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(c['numerical_threads']),
                             MKL_NUM_THREADS=str(c['numerical_threads']),LOKY_MAX_CPU_COUNT=str(c['numerical_threads']),MPLBACKEND='Agg'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                update(dict(state='running_original_full_author_checkpoint_replay',active_pid=child.pid,exact_child_handle=handle))
                code,usage=capped_wait(child,c,api,update)
            report=Path(c['artifact_root'])/'full_author_checkpoint_replay.json'
            passed=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
            exclusive(base/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,
                exact_child_handle=handle,report_sha256=sha(report) if report.exists() else None))
            update(dict(state='complete_full_author_checkpoint_replay_paper_cohort_still_pending' if passed else 'failed_preserved',
                        active_pid=None,resource_guard_aborted=usage['resource_guard_aborted']))
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if a.worker:worker(c)
    else:controller(c)


if __name__=='__main__':main()
