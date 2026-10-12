import ast
import inspect

from scripts.flow_matching import run_shcl_full413_completion_v2 as replay


def test_only_host_admission_changes_all_original_worker_guards_and_dispatch_remain_identical():
    tree=replay.main_ast();old=ast.parse(inspect.getsource(replay.original.main))
    old_slot=next(node for node in old.body[0].body if isinstance(node,ast.FunctionDef) and node.name=='ready_slot')
    for index,node in enumerate(tree.body[0].body):
        if isinstance(node,ast.FunctionDef) and node.name=='ready_slot':tree.body[0].body[index]=old_slot
    assert ast.dump(tree)==ast.dump(old)


def test_complete49_train364_test_scope_and_every_existing_limit_retained():
    native=replay.original.replay.native
    old=native.read(native.ROOT/'configs/runtime/fm_shcl_full413_completion_replay.v1.json')
    new=native.read(native.ROOT/'configs/runtime/fm_shcl_full413_completion_replay.v2.json')
    replay.original.verify(new);replay.verify_gpu_mutex(new)
    for key in ('canonical_job_ids','full_training_commands','full_test_commands','settings','gpu_settings',
            'python','environment','resource_lock','bitwise_loss_score_label_and_RNG_gate'):
        assert new[key]==old[key]
    assert len(new['canonical_job_ids'])==413
    assert new['GPU_resource_lock']=='experiments/runs/fm_grasp_protocol_gpu.lock'
