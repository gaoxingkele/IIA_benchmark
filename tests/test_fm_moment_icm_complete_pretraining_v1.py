import ast
import json
import os
import subprocess
import sys
from types import SimpleNamespace,CodeType
import pytest
from scripts.flow_matching import run_moment_icm_complete_pretraining_v1 as p


def test_both_full_models_preserve_all_data_and_original_shared_GPU_resources():
    c=p.registration();body=p.verify(c)
    assert [j['mixer'] for j in c['jobs']]==['independent','icm']
    assert all(j['expected_items']==80850189 and j['expected_batches']==39479 and j['sources']==2086
        and j['batch_size']==2048 and j['epochs']==1 and j['seed']==13 for j in c['jobs'])
    assert c['settings']['minimum_commit_headroom_bytes']==28*2**30
    assert c['settings']['minimum_gpu_free_bytes']==16*2**30 and c['settings']['emergency_gpu_free_bytes']==4*2**30
    assert c['settings']==p.read(p.ROOT/p.POLICY)['settings']
    assert p.read(p.ROOT/p.masks.CONFIG)['settings']['maximum_worker_rss_bytes']==3*2**30
    assert c['python']==body['python'] and c['resource_lock']==p.read(p.ROOT/p.POLICY)['resource_lock']


@pytest.mark.parametrize('change',['omit_model','drop322','reduce_steps','batch','precision_env','lock','guard'])
def test_smaller_or_bypassed_scientific_runs_cannot_be_registered(change):
    c=p.registration()
    if change=='omit_model':c['jobs']=c['jobs'][:1]
    elif change=='drop322':c['jobs'][0]['sources']=1764
    elif change=='reduce_steps':c['jobs'][0]['expected_batches']=2
    elif change=='batch':c['jobs'][0]['batch_size']=64
    elif change=='precision_env':c['python']='another-python.exe'
    elif change=='lock':c['resource_lock']='bypass.lock'
    else:c['settings']['minimum_gpu_free_bytes']=1
    with pytest.raises(ValueError):p.verify(c)


def test_worker_refuses_incomplete_actual_prerequisites_before_any_model_import(monkeypatch):
    c=p.registration();monkeypatch.setattr(p,'ready',lambda:None)
    with pytest.raises(ValueError,match='before numerical imports'):p.worker(c,c['jobs'][0]['id'])


@pytest.mark.parametrize('excess',['commit','gpu'])
def test_unchanged_original_neural_host_and_gpu_guards_abort_without_real_processes(excess):
    settings=p.registration()['settings'];snapshot,wait=p.guarded_gpu_functions({})
    namespace=wait.__globals__
    usage=SimpleNamespace(rss=2*2**30,private=5*2**30,vms=1)
    proc=SimpleNamespace(memory_info=lambda:usage,children=lambda **kwargs:[])
    namespace['psutil']=SimpleNamespace(Process=lambda pid:proc,NoSuchProcess=RuntimeError,AccessDenied=PermissionError)
    namespace['gpu_snapshot']=lambda s:dict(gpu_free_bytes=(4*2**30-1 if excess=='gpu' else 20*2**30),gpu_index=0,gpu_uuid=s['gpu_uuid'])
    child=SimpleNamespace(pid=123,code=None)
    child.poll=lambda:child.code;child.wait=lambda:child.code
    def terminate(item):item.code=-9
    api=dict(resource_snapshot=lambda:dict(commit_available_bytes=(16*2**30-1 if excess=='commit' else 32*2**30)),terminate_tree=terminate)
    code,peaks=wait(child,settings,api,lambda value:None)
    assert code==-9 and peaks['resource_guard_aborted']
    # Admission and original GPU emergency limits were never relaxed.
    assert settings['emergency_gpu_free_bytes']==4*2**30 and settings['emergency_commit_headroom_bytes']==16*2**30


def test_cold_registration_and_guard_binding_do_not_import_torch_or_claim_prerequisite_success():
    code='''
import sys,json
from scripts.flow_matching import run_moment_icm_complete_pretraining_v1 as p
from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
c=p.registration();p.verify(c);api=install_facades();a,b=p.guarded_gpu_functions(api)
assert 'torch' not in sys.modules and callable(a) and callable(b)
print(json.dumps(dict(passed=True,models=2,torch_imported=False)))
'''
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['torch_imported'] is False


def test_registered_GPU_identity_is_checked_without_initializing_CUDA():
    settings=p.registration()['settings'];snapshot,_=p.guarded_gpu_functions({})
    namespace=snapshot.__globals__
    namespace['subprocess']=SimpleNamespace(check_output=lambda *args,**kwargs:'0, '+settings['gpu_uuid']+', 20000')
    assert snapshot(settings)['gpu_free_bytes']==20000*2**20
    namespace['subprocess']=SimpleNamespace(check_output=lambda *args,**kwargs:'0, wrong-GPU, 20000')
    with pytest.raises(RuntimeError,match='GPU identity'):snapshot(settings)


def test_neural_guard_is_loaded_without_changing_the_original_source_AST():
    snapshot,wait=p.guarded_gpu_functions({})
    # No CPU data-worker cap is inserted into the neural policy. All source
    # host/GPU emergency checks, retry semantics and process-tree tracking stay.
    source=p.ROOT/'scripts/flow_matching/run_grasp_protocol_queue.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    def semantics(code):
        return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
            tuple(semantics(v) if isinstance(v,CodeType) else v for v in code.co_consts))
    for name,function in [('gpu_snapshot',snapshot),('guarded_gpu_wait',wait)]:
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        namespace={}
        exec(compile(ast.Module(body=[original],type_ignores=[]),str(source),'exec'),namespace)
        assert semantics(function.__code__)==semantics(namespace[name].__code__)
        assert function.__code__.co_filename=='<unchanged-original-neural-host-and-gpu-guard>'
