import ast
import copy
import inspect

import pytest

from scripts.flow_matching import audit_full401_image_checkpoint_CPU_v1 as audit
from scripts.flow_matching import run_shcl_training_rng_queue_v3 as gpu


def cpu_config(): return audit.read(audit.ROOT/'configs/runtime/fm_full401_image_checkpoint_CPU_audit.v1.json')
def gpu_config(): return audit.read(audit.ROOT/'configs/runtime/fm_shcl_all49_training_rng_replay.v3.json')


def test_saved_full401_audit_keeps_entire25_job_2000epoch_scope_and_original_resource_policy():
    config=cpu_config(); job=audit.verify(config)
    assert job['training']['epochs']==2000 and job['training']['batch_size']==32
    assert config['completed_epochs']==401 and config['required_original_epochs']==2000
    reference=audit.read(audit.ROOT/config['reference_readonly_resource_configuration'])
    assert config['settings']==reference['settings']


@pytest.mark.parametrize('field',['completed_epochs','required_original_epochs','settings'])
def test_CPU_audit_cannot_shorten_saved_history_or_weaken_original_guards(field):
    config=copy.deepcopy(cpu_config())
    if field=='settings': config['settings']['maximum_worker_private_bytes']+=1
    else: config[field]-=1
    with pytest.raises(ValueError): audit.verify(config)


def test_all49_existing_GPU_checkpoint_gates_preserve_budget_and_add_original_training_GPU_mutex():
    config=gpu_config(); original=gpu.previous.original.job_api.verify(config)[0]
    gpu.verify_gpu_mutex(config)
    assert len([j for j in original['jobs'] if j['mode']=='train'])==49
    assert config['GPU_resource_lock']=='experiments/runs/fm_grasp_protocol_gpu.lock'
    assert config['GPU_resource_lock']!=config['resource_lock']


def test_new_GPU_admission_changes_one_context_entry_and_two_original_host_IO_bindings_only():
    tree=gpu.main_ast(); original=ast.parse(inspect.getsource(gpu.previous.original.main))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='combined_slot':
            replacement=ast.parse('slot(fixture)',mode='eval').body
            node.func=replacement.func;node.args=replacement.args;node.keywords=[]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='write':
            node.func=ast.parse('native.write',mode='eval').body
    assert ast.dump(tree)==ast.dump(original)


def test_GPU_handoff_refuses_current_native_worker_any_verified_result_or_PID_reuse():
    config=gpu_config();old=config['handoff']['previous_owner']
    state=dict(pid=old['pid'],create_time=old['create_time'],state='waiting_for_full_original_training_checkpoints',active_pid=None,
        required_count=49,verified_count=0,failures={},configuration_sha256=config['handoff']['previous_configuration_sha256'])
    assert gpu.safe_phase(config,state,[])
    for mutation in ('pid','time','active','verified'):
        changed=dict(state)
        if mutation=='pid':changed['pid']+=1
        elif mutation=='time':changed['create_time']+=1
        elif mutation=='active':changed['active_pid']=123
        else:changed['verified_count']=1
        assert not gpu.safe_phase(config,changed,[])
    assert not gpu.safe_phase(config,state,[dict(name='python.exe',exe='F:/native/python.exe')])
