"""Full native recovery fidelity and actual68-epoch resource incident evidence."""
import ast
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.flow_matching.rcgan_seed1106_resume_binding_v1 import MODULES, recovery_ast, stronger_admission
from scripts.flow_matching.run_rcgan_seed1106_recovery_dispatch_v1 import DISPATCH_MODULES, dispatch_ast, bind
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha

PREFIX = 'configs/reproducibility/fm_rcgan_seed1106_'


@pytest.mark.parametrize('name', ['audit_rcgan_random_order_chain_v1.py',
    'audit_rcgan_checkpoint_resume_state_v1.py', 'rcgan_native_training_resume_runtime_v1.py',
    'run_rcgan_native_state_recovery_v1.py'])
def test_every_scientific_statement_and_complete_gate_preserved(name):
    source = ROOT.joinpath('scripts/flow_matching', name).read_text(encoding='utf-8')
    expected = ast.parse(source)
    expected.body = [node for node in expected.body if isinstance(node, ast.FunctionDef)]
    actual = recovery_ast(source)

    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int:
                node.value = {68: 238, 61: 231, 432: 262}.get(node.value, node.value)
            elif isinstance(node.value, str):
                node.value = node.value.replace('68', '238').replace('61', '231').replace('432', '262').replace('epoch_067', 'epoch_237')
                for old, new in MODULES.items():
                    node.value = node.value.replace(new, old)
            return node

        def visit_ImportFrom(self, node):
            node.module = {v: k for k, v in MODULES.items()}.get(node.module, node.module)
            return node

        def visit_Assign(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == '_cached_resource_api':
                return ast.ImportFrom(module='scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                    names=[ast.alias(name='resource_api')], level=0)
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_retry_admission_settings':
                return self.visit(node.args[1])
            return self.generic_visit(node)

    assert ast.dump(Reverse().visit(actual)) == ast.dump(expected)


def test_actual_failed68_epoch_snapshot_is_complete_and_not_fabricated():
    config = read(ROOT / (PREFIX + 'full68_random_order_chain.2026-10-13_v1.json'))
    from scripts.flow_matching.audit_rcgan_seed1106_chain_v1 import verify_registration
    queue, job, output, evaluations = verify_registration(config)
    assert job['seed'] == 1106 and len(queue['jobs']) == 5
    assert len(evaluations) == 68
    assert [x['epoch'] for x in evaluations] == list(range(68))
    assert len(list(output.joinpath('native_checkpoints').iterdir())) == 68
    receipt = read(ROOT / config['failed_terminal_receipt'])
    assert receipt['exit_code'] == 1 and receipt['resource_guard_aborted'] is True
    assert receipt['abort_snapshot']['commit_available_bytes'] < queue['settings']['emergency_commit_headroom_bytes']


def test_dispatch_keeps_original500_and_requires_both_complete_gates():
    config = read(ROOT / (PREFIX + 'full68_recovery_dispatch.2026-10-13_v1.json'))
    bind()['verify_dispatch_registration'](config)
    resume = read(ROOT / config['resume_configuration'])
    assert resume['verified_completed_epochs'] == 68
    assert resume['remaining_full_epochs'] == 432
    assert resume['full_original_epochs'] == 500
    assert resume['recovery_job']['seed'] == resume['canonical_job']['seed'] == 1106
    assert [x['name'] for x in config['stages']] == ['random_order_chain', 'checkpoint_state']
    assert resume['original_scope'] == dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
    for item in resume['source_receipts']:
        assert sha(ROOT / item['path']) == item['sha256']


def test_only_admission_is_stronger_while_every_emergency_cap_stays():
    resume = read(ROOT / (PREFIX + 'full432_state_recovery.2026-10-13_v1.json'))
    before = copy.deepcopy(resume['settings'])
    after = stronger_admission(resume, before)
    assert after['minimum_commit_headroom_bytes'] == 40 * 1024 ** 3
    assert before == resume['settings']
    assert {k:v for k,v in after.items() if k != 'minimum_commit_headroom_bytes'} == {
        k:v for k,v in before.items() if k != 'minimum_commit_headroom_bytes'}
    assert after['emergency_commit_headroom_bytes'] == 16 * 1024 ** 3


@pytest.mark.parametrize('value', [28 * 1024 ** 3, 32 * 1024 ** 3, 1, None])
def test_incident_admission_cannot_be_lowered_or_omitted(value):
    resume = read(ROOT / (PREFIX + 'full432_state_recovery.2026-10-13_v1.json'))
    resume['retry_admission_commit_headroom_bytes'] = value
    with pytest.raises(ValueError):
        stronger_admission(resume, resume['settings'])


def test_registered_full_recovery_is_not_a_claim_of_completed_proof_or_training():
    evidence = read(ROOT / 'projects/flow_matching_research/results/2026-10-13/rcgan_seed1106_full68_recovery_v1/actual_resource_abort_and_registered_full_recovery.json')
    assert evidence['complete_saved_checkpoints'] == evidence['complete_saved_evaluators'] == 68
    assert evidence['persisted_trace_epochs'] == 61
    assert evidence['missing_original_elapsed_times'] == list(range(61, 68))
    assert evidence['native_state_recovery_proved'] is False
    assert evidence['full500_result_verified'] is False
    assert evidence['identical_retry_performed'] is False
    assert evidence['all_original_experiments_complete'] is False


def test_all_original_binding_functions_compile_without_scientific_imports():
    code = '''import sys
from scripts.flow_matching.audit_rcgan_seed1106_chain_v1 import bind as chain
from scripts.flow_matching.audit_rcgan_seed1106_state_v1 import bind as state
from scripts.flow_matching.rcgan_seed1106_resume_runtime_v1 import bind as runtime
from scripts.flow_matching.run_rcgan_seed1106_recovery_v1 import bind as resume
from scripts.flow_matching.run_rcgan_seed1106_recovery_dispatch_v1 import bind as dispatch
for fn in (chain,state,runtime,resume,dispatch): fn()
assert 'numpy' not in sys.modules and 'tensorflow' not in sys.modules
'''
    subprocess.run([sys.executable, '-X', 'utf8', '-c', code], check=True)
