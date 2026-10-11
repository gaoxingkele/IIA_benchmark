import ast
from copy import deepcopy

import pytest

from scripts.flow_matching import audit_saits_physio_patient_discrepancies_v2 as cpu
from scripts.flow_matching import verify_saits_physio_patient_discrepancies_v2 as cpu_review
from scripts.flow_matching.audit_saits_physio_original_cuda_replay_v3 import worker_ast,abort_reason
from scripts.flow_matching.verify_saits_physio_original_cuda_replay_v3 import review_ast,check_terminal


def test_full_original_checkpoint_loader_every_grid_batch_metric_statistic_and_tolerance_preserved():
    original=next(n for n in cpu.replay_ast().body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    class Undo(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.dump(node)==ast.dump(ast.parse('inference = dict(native)').body[0]):
                return ast.parse("inference = dict(native, device='cpu')").body[0]
            return self.generic_visit(node)
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and node.args:
                if ast.dump(node.args[0])==ast.dump(ast.parse("native['device']",mode='eval').body):
                    node.args[0]=ast.Constant(value='cpu')
            if isinstance(node.func,ast.Name) and node.func.id=='dict':
                keys={k.arg for k in node.keywords}
                if 'full_checkpoint_contents_and_re_inference_passed' in keys:
                    original_dict=next(n for n in ast.walk(original) if isinstance(n,ast.Call)
                        and isinstance(n.func,ast.Name) and n.func.id=='dict'
                        and 'full_checkpoint_contents_and_re_inference_passed' in {k.arg for k in n.keywords})
                    old={k.arg:k.value for k in original_dict.keywords}
                    node.keywords=[k for k in node.keywords if k.arg!='replay_inference_device']
                    for k in node.keywords:
                        if k.arg in ['cuda_context_initialized','boundary']:k.value=deepcopy(old[k.arg])
            return node
        def visit_If(self,node):
            if ast.dump(node.test)==ast.dump(ast.parse('not torch.cuda.is_initialized()',mode='eval').body):
                return ast.parse("if torch.cuda.is_initialized():\n    raise ValueError('Readonly CPU inference must not initialize CUDA')").body[0]
            return self.generic_visit(node)
    reverted=Undo().visit(worker_ast()).body[0]
    # Original v2 begins with diagnostic-list initialization, then visibility.
    reverted.body.insert(1,ast.parse("os.environ['CUDA_VISIBLE_DEVICES'] = '-1'").body[0])
    assert ast.dump(reverted)==ast.dump(original)


def test_complete_independent_truth_mask_patient_checkpoint_optimizer_rng_review_ast_preserved():
    class Undo(ast.NodeTransformer):
        def visit_Constant(self,node):
            substitutions={
                'scripts.flow_matching.verify_saits_physio_original_cuda_replay_v3':'scripts.flow_matching.verify_saits_physio_patient_discrepancies_v2',
                'audit_saits_physio_original_cuda_replay_v3':'audit_saits_physio_patient_discrepancies_v2'}
            if isinstance(node.value,str) and node.value in substitutions:node.value=substitutions[node.value]
            return node
    assert ast.dump(Undo().visit(review_ast()))==ast.dump(cpu_review.review_ast())


@pytest.mark.parametrize('violation',['commit','rss','private','gpu'])
def test_all_frozen_resource_abort_gates_enforced(violation):
    settings=dict(emergency_commit_headroom_bytes=100,maximum_worker_rss_bytes=50,
        maximum_worker_private_bytes=70,emergency_gpu_free_bytes=20)
    memory=dict(commit_available_bytes=100);rss=50;private=70;gpu=dict(gpu_free_bytes=20)
    assert abort_reason(memory,rss,private,gpu,settings) is None
    if violation=='commit':memory['commit_available_bytes']=99
    elif violation=='rss':rss=51
    elif violation=='private':private=71
    else:gpu['gpu_free_bytes']=19
    assert abort_reason(memory,rss,private,gpu,settings)==violation


def test_transient_gpu_observation_is_not_a_terminal_failure_and_host_caps_still_apply():
    settings=dict(emergency_commit_headroom_bytes=100,maximum_worker_rss_bytes=50,
        maximum_worker_private_bytes=70,emergency_gpu_free_bytes=20)
    assert abort_reason(dict(commit_available_bytes=100),50,70,None,settings) is None
    assert abort_reason(dict(commit_available_bytes=99),50,70,None,settings)=='commit'


def terminal_fixture(passed=False):
    ids=['a','b']; reviews=[dict(id='a',passed=True),dict(id='b',passed=passed)]
    producer=dict(completed_jobs=[dict(id=x) for x in ids],full_model_count=2)
    replay=dict(records=[dict(id=x,original_patient_statistics_match=r['passed']) for x,r in zip(ids,reviews)],
        original_patient_statistics_reviews=reviews,all_registered_native_models_reinferred=True,
        deferred_numerical_discrepancies_never_certify_reproduction=True,
        full_checkpoint_contents_and_re_inference_passed=passed,full_models_replayed=2,
        additional_training_seed_slots=0,cuda_context_initialized=True,replay_inference_device='cuda')
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_native_model_replay_passed=passed)
    return replay,receipt,producer


@pytest.mark.parametrize('passed',[True,False])
def test_actual_complete_cuda_replay_can_be_audited_without_relabeling_strict_failure(passed):
    check_terminal(*terminal_fixture(passed))


@pytest.mark.parametrize('violation',['cpu','wrong_device','partial','false_pass','abort','failed_exit'])
def test_gpu_full_coordinate_and_original_strict_checks_remain_fatal(violation):
    replay,receipt,producer=terminal_fixture(False)
    if violation=='cpu':replay['cuda_context_initialized']=False
    elif violation=='wrong_device':replay['replay_inference_device']='cpu'
    elif violation=='partial':replay['records'].pop()
    elif violation=='false_pass':replay['full_checkpoint_contents_and_re_inference_passed']=True
    elif violation=='abort':receipt['resource_guard_aborted']=True
    else:receipt['exit_code']=1
    with pytest.raises(ValueError):check_terminal(replay,receipt,producer)
