"""One tiny full source/checkpoint/native integration fixture; no benchmark."""
import json
import numpy as np
from scripts.flow_matching import industrial_extended_fullcase_common_v4 as common
from scripts.flow_matching.timegan_native_common_v1 import write,read,sha


def test_complete_twenty_epoch_source_and_native_auxiliary_head_pipeline(tmp_path,monkeypatch):
    import torch
    from iia_benchmark.models.fm_objective_stable_extended import extended_plan
    from scripts.flow_matching.diagnose_industrial_remaining8_entropic_couplings_v1 import sequence
    monkeypatch.setattr(common,'ROOT',tmp_path);torch.set_num_threads(2)
    groups=dict(training_groups=[dict(array_prefix='t',group_id='train',entity_id='train',source_interval=[0,6],points=6,features=2)],
        validation_groups=[dict(array_prefix='v',group_id='val',entity_id='val',source_interval=[0,5],points=5,features=2)],
        entities=[dict(array_prefix='a',group_id='testA',entity_id='testA',source_interval=[0,7],points=7,test_points=7,features=2),
                  dict(array_prefix='b',group_id='testB',entity_id='testB',source_interval=[0,8],points=8,test_points=8,features=2)])
    inputs={};rng=np.random.default_rng(8)
    for role,key in [('train','training_groups'),('validation','validation_groups'),('test','entities')]:
        for g in groups[key]:
            p=g['array_prefix'];n=g['points'];inputs[p+('_test' if role=='test' else '_values')]=rng.normal(size=(n,2)).astype(np.float32)
            inputs[p+'_labels']=np.asarray([i%3!=0 for i in range(n)],dtype=np.uint8)
            inputs[p+'_timestamps']=np.arange(n,dtype=np.float64);inputs[p+'_source_rows']=np.arange(n,dtype=np.int64)
            inputs[p+'_fault_labels']=np.asarray(['normal' if i%3==0 else 'fault' for i in range(n)])
    np.savez_compressed(tmp_path/'input.npz',**inputs)
    dataset=dict(id='fixture',input_npz='input.npz',input_sha256=sha(tmp_path/'input.npz'),source_files=[],scaler=dict(fit_points=6),selection_audit=dict(unit_fixture_only=True),**groups)
    write(tmp_path/'manifest.json',dict(datasets=[dataset]))
    parameters=dict(objective='sf2m',sigma=1.,window=3,hidden_size=4,epochs=20,batch_size=3,score_weight=1.,score_times=[.5],
        monte_carlo_samples=1,sinkhorn_max_final_iterations=200000)
    configuration=dict(id='unit_fixture',entrypoint='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector',
        parameters=parameters,reproduction_status='unit_fixture_only_not_benchmark')
    write(tmp_path/'model.json',configuration)
    windows=np.stack([inputs['t_values'][:3],inputs['t_values'][3:]])
    def solve(a,b,e):
        audits=[];plan=extended_plan(a,b,e,audit=audits);return plan,audits[-1]
    problems=sequence(windows,parameters,7,solve);write(tmp_path/'couplings.json',dict(coupling_problems=problems))
    job=dict(id='unit_fixture',dataset='fixture',seed=7,window=3,model_config='model.json',dataset_manifest_path='manifest.json',
        parameter_overrides=dict(seed=7,device='cpu'),output_directory='output',frozen_files=[dict(path='model.json',sha256=sha(tmp_path/'model.json'))])
    evaluation=dict(cpu_threads=2,strict_false_alarm_percent=[.1,1,5],primary_false_alarm_percent=1,bootstrap_resamples=2,
        bootstrap_block_points=2,tab_threshold_ratios_percent=[1],tab_commit='fixture')
    case=dict(canonical_original_id='unit_fixture',job=job,evaluation=evaluation,coupling_report=str(tmp_path/'couplings.json'),fit_windows=2,optimizer_updates=20)
    common.train(case)
    audit=read(tmp_path/'output/complete_training_solver_audit.json')
    assert audit['passed'] and audit['optimizer_updates']==20 and len(audit['transport_problems'])==20
    guard=tmp_path/'source_guard.json';write(guard,dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=sha(tmp_path/'output/complete_training_solver_audit.json')))
    common.replay(case,guard,tmp_path/'native')
    proof=read(tmp_path/'native/report.json')
    assert proof['passed'] and proof['all_score_arrays_exact_bytes'] and proof['new_optimizer_updates']==0
    assert sum(r['points'] for r in proof['native_coverage'] if r['role']=='test')==15
    assert set(proof['independent_strict_metrics'])=={'0.1','1','5'}
