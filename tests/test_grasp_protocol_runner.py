import json
import numpy as np
import pytest

from scripts.flow_matching.grasp_protocol_runner import complete, fingerprint, metric_records, sha, write_json
from scripts.flow_matching.register_grasp_protocol import score_recipes
from scripts.flow_matching.summarize_grasp_protocol import aggregate


def test_full_inference_scope_deduplicates_default_without_cartesian_invention():
    config = {'inference_flow_times':[1,2,5,10,20,50],'inference_source_samples':[1,2,5,10,20]}
    recipes = score_recipes('main',config)
    assert len(recipes)==10 and recipes[0]=={'source_samples':5,'flow_evaluations':10}
    assert {(r['source_samples'],r['flow_evaluations']) for r in recipes} == {(5,n) for n in [1,2,5,10,20,50]}|{(n,10) for n in [1,2,5,10,20]}
    assert score_recipes('gnn',config)==[recipes[0]]


def test_actual_native_metrics_are_separate_from_validation_threshold():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[1]/'experiments/runs/fm_grasp_paper_protocol_v1/sources/mtsbench/original/Detectors/evaluation/basic_metrics.py'
    spec = importlib.util.spec_from_file_location('native_metric_test',source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.basic_metricor()
    labels,scores,validation = np.array([0,0,1,1]),np.array([0.,2.,3.,4.]),np.array([0.,1.,2.])
    a = metric_records(native,labels,scores,validation)
    b = metric_records(native,1-labels,scores,validation)
    assert a['validation_only_control']['threshold']==b['validation_only_control']['threshold']
    assert a['validation_only_control']['threshold']==np.quantile(validation,.99)
    assert a['paper_retrospective']['Best_F1']==pytest.approx(2/(2+.00001))
    assert a['paper_retrospective']['Best_F1']!=b['paper_retrospective']['Best_F1']
    assert a['validation_only_control']['test_labels_used_for_calibration'] is False
    assert a['paper_retrospective']['point_adjustment'] is False


def test_complete_requires_actual_full_budget_all_sweeps_and_artifact_hashes(tmp_path):
    job = {'id':'full','output_directory':'result','expected_optimizer_updates':4500,
           'test_points':120,'validation_points':100,'score_recipes':[{'source_samples':5,'flow_evaluations':10}]}
    checkpoint = tmp_path/'checkpoint'
    checkpoint.write_bytes(b'weights')
    history_path = tmp_path/'result/training_history.jsonl'
    history_path.parent.mkdir()
    history_path.write_text('\n'.join(json.dumps({'epoch':epoch,'optimizer_updates':3*epoch,
                    **({'validation_loss':.1} if epoch%50==0 else {})}) for epoch in range(1,1501)),encoding='utf-8')
    scores = tmp_path/'scores'
    scores.write_bytes(b'saved_scores')
    result = {'status':'completed','job_sha256':fingerprint(job),'diagnostic':False,'paper_equivalence_certified':False,
              'epochs_completed':1500,'optimizer_updates':4500,'scored_points':120,'validation_points':100,
              'score_recipes':job['score_recipes'],'metrics':[{'recipe':r} for r in job['score_recipes']],
              'artifacts':[{'path':p.relative_to(tmp_path).as_posix(),'sha256':sha(p)} for p in [checkpoint,history_path,scores]]}
    target = tmp_path/'result/result.json'
    write_json(target,result)
    assert complete(job,tmp_path)
    for field,bad in [('epochs_completed',1),('optimizer_updates',3000),('scored_points',100),('score_recipes',[]),('diagnostic',True)]:
        write_json(target,{**result,field:bad})
        with pytest.raises(ValueError):
            complete(job,tmp_path)
    write_json(target,result)
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        complete(job,tmp_path)


def test_seed_uncertainty_does_not_promote_missing_or_single_replicate():
    assert aggregate([])['mean'] is None
    assert aggregate([.5])['se'] is None
    result = aggregate([.4,.5,.6])
    assert result['mean']==pytest.approx(.5)
    assert result['std']==pytest.approx(.1)
    assert result['ci95_low']<.4 and result['ci95_high']>.6


def test_full_registration_declares_all_paper_main_architecture_and_tau_profiles():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/experiments/fm_grasp_protocol_registration.v2.json').read_text(encoding='utf-8'))
    profiles = {p['id']:{**config['base_parameters'],**p['overrides']} for p in config['profiles']}
    assert len(profiles)==12 and len(config['seeds'])==10
    assert sum(d['entities'] for d in config['datasets'])*len(profiles)*len(config['seeds'])==9120
    assert len(config['datasets'])*len(profiles)*len(config['seeds'])==480
    assert {profiles[x]['architecture'] for x in ['main','gnn','transformer']}=={'tsmixer','gnn','transformer'}
    assert {profiles[x]['tau'] for x in profiles if x.startswith('tau_')}|{profiles['main']['tau']}=={0,.25,.5,1,2,4,8}
    assert all(p['epochs']==1500 and p['batch_size']==256 and p['validation_interval']==50 for p in profiles.values())


def test_gpu_guard_checks_identity_and_aborts_only_its_owned_child(monkeypatch):
    from types import SimpleNamespace
    from scripts.flow_matching import run_grasp_protocol_queue as controller
    settings = {'gpu_index':0,'gpu_uuid':'GPU-registered','emergency_gpu_free_bytes':2*1024**3,
                'emergency_commit_headroom_bytes':10*1024**3}
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**k:'0, GPU-registered, 1024\n')
    assert controller.gpu_snapshot(settings)['gpu_free_bytes']==1024**3
    with pytest.raises(RuntimeError):
        controller.gpu_snapshot({**settings,'gpu_uuid':'GPU-other'})
    memory = SimpleNamespace(rss=100,private=200,vms=200)
    monkeypatch.setattr(controller.psutil,'Process',lambda pid:SimpleNamespace(children=lambda **k:[],memory_info=lambda:memory))
    child = SimpleNamespace(pid=123,poll=lambda:None,wait=lambda:-9)
    stopped, updates = [], []
    api = {'resource_snapshot':lambda:{'physical_available_bytes':16*1024**3,'commit_available_bytes':20*1024**3},
           'terminate_tree':lambda owned:stopped.append(owned)}
    code, usage = controller.guarded_gpu_wait(child,settings,api,updates.append)
    assert code==-9 and stopped==[child]
    assert usage['resource_guard_aborted'] is True
    assert usage['minimum_commit_available_bytes']==20*1024**3
    assert usage['minimum_gpu_free_bytes']==1024**3
    assert updates[-1]['state']=='resource_guard_aborted'


def test_transient_gpu_observation_timeout_keeps_polling_same_child(monkeypatch):
    from types import SimpleNamespace
    from scripts.flow_matching import run_grasp_protocol_queue as controller
    settings = {'emergency_gpu_free_bytes':2*1024**3,'emergency_commit_headroom_bytes':10*1024**3}
    observations = iter([None,None,0])
    child = SimpleNamespace(pid=123,poll=lambda:next(observations),wait=lambda:0)
    def observe(settings):
        raise controller.subprocess.TimeoutExpired('nvidia-smi',10)
    monkeypatch.setattr(controller,'gpu_snapshot',observe)
    monkeypatch.setattr(controller.time,'sleep',lambda _:None)
    memory = SimpleNamespace(rss=100,private=200,vms=200)
    monkeypatch.setattr(controller.psutil,'Process',lambda pid:SimpleNamespace(children=lambda **k:[],memory_info=lambda:memory))
    stopped=[]
    api = {'resource_snapshot':lambda:{'physical_available_bytes':16*1024**3,'commit_available_bytes':20*1024**3},
           'terminate_tree':lambda owned:stopped.append(owned)}
    code, usage = controller.guarded_gpu_wait(child,settings,api,lambda _:None)
    assert code==0 and stopped==[] and usage['gpu_observation_retries']==2
    assert usage['resource_guard_aborted'] is False
