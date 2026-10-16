import ast
from pathlib import Path
import pytest
from scripts.flow_matching import run_otcfm_singlecell_original_CUDA_full120_v1 as s


def test_all120_original_science_and_all24_full_probes_survive_without_new_seeds():
    c=s.registration();base=s.read(s.ROOT/s.original.QUEUE);old={j['id']:j for j in base['jobs']}
    assert len(c['jobs'])==120 and len(s.preflight.cases(c))==24
    assert c['settings']==base['settings'] and c['python']==base['python'] and c['environment_lock']==base['environment_lock']
    assert c['GPU']['minimum_free_bytes']==18*2**30 and c['native_metric_tolerance']==dict(rtol=2e-6,atol=2e-6)
    for job in c['jobs']:
        canonical=old[job['canonical_CPU_condition_reference']]
        for key in ['scientific_protocol','data','seed','method','net','sigma_min','ot_sampler','leaveout_timepoint','source_yaml']:
            assert job[key]==canonical[key]
        assert job['artifact_directory']!=canonical['artifact_directory']
        assert job['scientific_protocol']['max_epochs']==1000
    assert c['jobs'][0]['canonical_CPU_condition_reference']==s.FIRST
    assert not c['original_CPU_conditions_replaced'] and c['new_independent_training_seed_slots']==0


def test_only_exact_device_and_full_probe_observations_change_in_native_AST():
    text=Path(s.original.__file__).read_text();tree=s.transform_native(text)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    worker=ast.unparse(functions['worker']);runtime=ast.unparse(functions['source_runtime'])
    assert "accelerator='gpu'" in worker and "precision=32" in worker
    assert 'trainer.fit' in worker and 'trainer.test' in worker and 'contract.terminal_reason' in worker
    assert "os.environ['CUDA_VISIBLE_DEVICES'] = '0'" in runtime
    assert 'if not torch.cuda.is_initialized()' in worker
    assert worker.count('_observe_full_initial_velocity_field')==1 and worker.count('_GPU_batch')==2
    assert worker.count(".to('cuda:0')")==4
    with pytest.raises(ValueError):s.transform_native(text.replace("accelerator='cpu'", "accelerator='gpu'"))


def test_private_source_and_replay_keep_native_losses_and_all_independent_LP_checks():
    c=s.registration();native,inspect,audit=s.modules(c)
    assert native.QUEUE==s.CONFIG and inspect.native is native and audit.native is native
    assert audit.inspect is inspect.inspect and audit.CONFIG==s.CONFIG
    assert native.identity.__code__.co_code==s.original.identity.__code__.co_code
    assert inspect.validate_record.__code__.co_code==s.integrity.validate_record.__code__.co_code
    def names(code):
        value=set(code.co_names)
        for item in code.co_consts:
            if hasattr(item,'co_names'):value|=names(item)
        return value
    assert set(['inspect','Trainer','test','isclose','load_state_dict']).issubset(names(audit.worker.__code__))
    # Exact original whole-array W1/W2 LP implementation remains in the nested observer.
    nested=[v for v in audit.worker.__code__.co_consts if hasattr(v,'co_name') and v.co_name=='observe']
    assert len(nested)==1
    assert 'cdist' in nested[0].co_freevars or 'cdist' in nested[0].co_names


def test_GPU_emergency_monitor_keeps_all_original_CPU_resource_caps_and_own_child(monkeypatch):
    from scripts.flow_matching import audit_cfm_checkpoint_replay_v1 as capped
    class Child:pass
    child=Child();stopped=[];updates=[]
    def wait(actual,config,api,update):
        assert actual is child and config['settings']['maximum_worker_private_bytes']==8*2**30
        memory=api['resource_snapshot']();assert memory['commit_available_bytes']==40*2**30
        return 1,dict(resource_guard_aborted=False)
    monkeypatch.setattr(capped,'capped_wait',wait)
    monkeypatch.setattr(s.GPU_api,'gpu_snapshot',lambda:dict(free_bytes=1))
    c=dict(GPU=dict(emergency_free_bytes=2*2**30),settings=dict(maximum_worker_private_bytes=8*2**30))
    api=dict(resource_snapshot=lambda:dict(commit_available_bytes=40*2**30),terminate_tree=lambda p:stopped.append(p))
    code,usage=s.GPU_wait(child,c,api,updates.append)
    assert code==1 and usage['resource_guard_aborted'] and usage['GPU_resource_guard_aborted']
    assert stopped==[child]
