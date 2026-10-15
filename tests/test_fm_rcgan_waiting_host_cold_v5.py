import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import continue_rcgan_waiting_host_cold_v5 as s


def source():return Path(s.entry.original.previous.original.__file__).read_text(encoding='utf-8')


def guard(tree):
    reference=ast.dump(ast.parse('binding_path.exists()',mode='eval').body)
    return next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.dump(n.test)==reference)


def test_real_full238_plus262_native_registration_and_every_guard_retained():
    c=s.registration();native,q,state=s.verify(c);production=s.read(s.ROOT/s.PRODUCER)
    assert len(q['jobs'])==c['required_original_seed_slots']==5
    assert (c['original_full_epochs'],c['verified_prior_epochs'],c['remaining_full_epochs'])==(500,238,262)
    assert c['native_python']==q['python'] and c['settings']==native['settings']==production['settings']
    assert c['resource_lock']==production['resource_lock'] and c['stronger_admission_commit_bytes']==40*2**30
    assert s.sha(s.ROOT/c['immutable_proof_binding'])==c['immutable_proof_binding_sha256']
    assert s.read(s.ROOT/c['immutable_proof_binding'])==s.expected_binding(native,q,state)


def test_reverse_only_metadata_reuse_change_recovers_entire_original_host_ast():
    c=s.registration();expected=s.entry.host_dispatch_ast(source());actual=s.host_ast(source(),c['immutable_proof_binding_sha256'])
    old=guard(expected);new=guard(actual);write=new.orelse[0]
    for parent in ast.walk(actual):
        body=getattr(parent,'body',None)
        if isinstance(body,list) and new in body:
            index=body.index(new);new.body=deepcopy(old.body);new.orelse=[];body.insert(index+1,write);break
    assert ast.dump(actual)==ast.dump(expected)


@pytest.mark.parametrize('condition',['identical','changed_dict','changed_hash','absent'])
def test_metadata_binding_reused_only_when_original_bytes_and_values_match(tmp_path,condition):
    c=s.registration();path=tmp_path/'metadata_proof_only.json';binding=deepcopy(c['proof_binding'])
    if condition!='absent':path.write_bytes((s.ROOT/c['immutable_proof_binding']).read_bytes())
    expected_sha=c['immutable_proof_binding_sha256'];before=path.read_bytes() if path.exists() else None;writes=[]
    if condition=='changed_dict':binding['configuration_sha256']='changed'
    if condition=='changed_hash':expected_sha='0'*64
    node=guard(s.host_ast(source(),expected_sha))
    def write(p,value):writes.append(p);p.write_text(json.dumps(value))
    values=dict(binding_path=path,binding=binding,sha=s.sha,read=s.read,write=write)
    code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<unit-metadata-only>','exec')
    if condition in ['changed_dict','changed_hash']:
        with pytest.raises(ValueError):exec(code,values)
    else:exec(code,values)
    if condition=='absent':assert writes==[path] and s.read(path)==binding
    else:assert not writes and path.read_bytes()==before


@pytest.mark.parametrize('fault',['budget','epochs','proof','python','guard'])
def test_no_training_budget_checkpoint_or_resource_shortcut(fault):
    c=s.registration()
    if fault=='budget':c['remaining_full_epochs']=1
    elif fault=='epochs':c['verified_prior_epochs']=499
    elif fault=='proof':c['immutable_proof_binding_sha256']='0'*64
    elif fault=='python':c['native_python']=sys.executable
    else:c['settings']['emergency_commit_headroom_bytes']=1
    with pytest.raises(ValueError):s.verify(c)


def test_only_exact_idle_parent_can_be_handed_off(monkeypatch):
    c=s.registration();h=c['previous_owner'];state=dict(pid=h['pid'],create_time=h['create_time'],state='waiting_for_memory',active_pid=None,shared_lock_owned=False)
    assert s.idle_allowed(c,state,h,[])
    assert not s.idle_allowed(c,dict(state,active_pid=123),h,[])
    assert not s.idle_allowed(c,state,h,[dict(name='python.exe',exe='C:/Python/python.exe')])


def test_actual_full_registration_and_host_ast_are_cold_without_tensorflow_numpy():
    code='from scripts.flow_matching import continue_rcgan_waiting_host_cold_v5 as s;from pathlib import Path;import sys,psutil,json;c=s.registration();s.verify(c);compile(s.host_ast(Path(s.entry.original.previous.original.__file__).read_text(),c["immutable_proof_binding_sha256"]),"<host>","exec");assert "numpy" not in sys.modules and "tensorflow" not in sys.modules;print(json.dumps(dict(private=psutil.Process().memory_info().private)))'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    assert json.loads(p.stdout)['private']<128*2**20
