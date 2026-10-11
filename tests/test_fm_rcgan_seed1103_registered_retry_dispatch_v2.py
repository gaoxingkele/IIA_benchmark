import ast
import copy
from pathlib import Path
import pytest

from scripts.flow_matching import rcgan_seed1103_registered_retry_runtime_v2 as runtime
from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v2 as dispatch


@pytest.mark.parametrize('name,path,expected',[
    ('conhost.exe','C:/Windows/System32/conhost.exe',True),
    ('python.exe','C:/Windows/System32/conhost.exe',False),
    ('conhost.exe','C:/untrusted/conhost.exe',False),
    ('python.exe','F:/author/python.exe',False),
])
def test_only_the_actual_windows_console_host_is_excluded_from_native_children(name,path,expected):
    assert dispatch.is_windows_console_host(name,path) is expected


def test_native_runtime_changes_only_the_exact_retry_identity_comparison():
    source = Path(runtime.original.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run']
    matches = [node for node in ast.walk(tree) if isinstance(node,ast.Constant) and node.value=='__state_recovery_v1']
    assert len(matches)==1
    matches[0].value='__state_recovery_v2'
    assert ast.dump(tree)==ast.dump(runtime.native_retry_ast(source))


@pytest.mark.parametrize('change',['seed','canonical','retry_id','epochs','prior','remaining'])
def test_native_runtime_rejects_other_seeds_ids_or_narrowed_budgets(change):
    config=dict(canonical_job=dict(id='rcgan_mnistfull__seed1103',seed=1103),
        recovery_job=dict(id='rcgan_mnistfull__seed1103__state_recovery_v2',seed=1103),
        full_original_epochs=500,verified_completed_epochs=238,remaining_full_epochs=262)
    if change=='seed':config['recovery_job']['seed']=1104
    elif change=='canonical':config['canonical_job']['id']='other'
    elif change=='retry_id':config['recovery_job']['id']='rcgan_mnistfull__seed1103__state_recovery_v1'
    elif change=='epochs':config['full_original_epochs']=1
    elif change=='prior':config['verified_completed_epochs']=84
    elif change=='remaining':config['remaining_full_epochs']=1
    with pytest.raises(ValueError):runtime.run(config,{}, {})


def test_controller_only_changes_fresh_state_and_exact_worker_adapter_dispatch():
    source=Path(dispatch.original.__file__).read_text(encoding='utf-8')
    expected=dispatch.controller_ast(source,'rcgan')
    counts=dict(state=0,imports=0,command=0)
    for node in ast.walk(expected):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id=='base':
                node.value=ast.parse("ROOT/_dispatch_configuration['state_root']",mode='eval').body;counts['state']+=1
            elif node.targets[0].id=='command':
                position=next(i for i,n in enumerate(node.value.elts) if isinstance(n,ast.Constant)
                    and n.value=='scripts.flow_matching.run_rcgan_native_state_recovery_v1')
                node.value.elts[position].value='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v2'
                node.value.elts[position+1:position+1]=[ast.Constant(value='--runtime-config'),
                    ast.Name(id='_dispatch_configuration_path',ctx=ast.Load())];counts['command']+=1
        if isinstance(node,ast.ImportFrom) and node.module=='scripts.flow_matching.rcgan_native_training_resume_runtime_v1':
            node.module='scripts.flow_matching.rcgan_seed1103_registered_retry_runtime_v2';counts['imports']+=1
    assert counts==dict(state=1,imports=2,command=1)
    assert ast.dump(expected)==ast.dump(dispatch.dispatch_ast(source))


@pytest.mark.parametrize('broken',['native_identity_missing','multiple_native_identities'])
def test_unexpected_native_source_structure_rejected(broken):
    source=Path(runtime.original.__file__).read_text(encoding='utf-8')
    if broken=='native_identity_missing':source=source.replace('__state_recovery_v1','__different')
    else:source=source.replace("job['id']!=canonical['id']+'__state_recovery_v1'",
        "(job['id']!=canonical['id']+'__state_recovery_v1' or job['id']!=canonical['id']+'__state_recovery_v1')")
    with pytest.raises(ValueError):runtime.native_retry_ast(source)


@pytest.mark.parametrize('broken',['native_worker','state_path','extra_import'])
def test_unexpected_controller_dispatch_structure_rejected(broken):
    source=Path(dispatch.original.__file__).read_text(encoding='utf-8')
    if broken=='native_worker':source=source.replace('scripts.flow_matching.run_rcgan_native_state_recovery_v1','unexpected.worker')
    elif broken=='state_path':source=source.replace("base=ROOT/config['state_root']","base=ROOT/config['other_state_root']")
    else:source=source.replace('import verify_result','import verify_result, run')
    with pytest.raises(ValueError):dispatch.dispatch_ast(source)
