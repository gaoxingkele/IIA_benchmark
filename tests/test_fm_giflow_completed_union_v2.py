import ast
from scripts.flow_matching import run_giflow_completed_union_v2 as union


def test_all_original_full_model_array_and_tolerance_gates_remain_identical():
    source=ast.parse((union.ROOT/'scripts/flow_matching/audit_giflow_complete_replay_union_v1.py').read_text(encoding='utf8'))
    original=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
    actual=union.verify_ast().body[0]
    old_return=next(n.value for n in original.body if isinstance(n,ast.Return))
    actual_return=next(n.value for n in actual.body if isinstance(n,ast.Return))
    old_keywords={n.arg:n for n in old_return.keywords}
    report_keywords={id(n) for n in actual_return.keywords}
    changed=0
    for node in ast.walk(actual):
        if isinstance(node,ast.Compare) and ast.unparse(node)=="len(completed) != config['expected_completed_models']":
            node.comparators[0]=ast.Constant(value=3);changed+=1
        if isinstance(node,ast.keyword) and id(node) in report_keywords and node.arg in ('completed_canonical_models','full_scoring_batches',
                'full_test_window_evaluations','full_native_forward_calls','released_training_seeds_completed','corrected_training_seeds_completed'):
            node.value=old_keywords[node.arg].value;changed+=1
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            if node.value=='Frozen four completed models within the unchanged140 original slots required':
                node.value=node.value.replace('four','three');changed+=1
            if node.value.startswith('All four previously completed original canonical models;56'):
                node.value=node.value.replace('All four previously completed original canonical models;56',
                    'All three previously completed original canonical models;42');changed+=1
    assert changed==9 and ast.dump(actual)==ast.dump(original)


def test_cold_parent_import_does_not_import_numerical_or_GPU_backend():
    import subprocess,sys
    code="import scripts.flow_matching.run_giflow_completed_union_v2,sys;assert not {'numpy','scipy','torch'}&set(sys.modules)"
    assert subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True).returncode==0
