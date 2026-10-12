import ast

from scripts.flow_matching import run_timegan_full45_completion_replay_v4 as current
from scripts.flow_matching import run_timegan_full45_completion_replay_v3 as previous


def test_all_native_array_validation_and_original_full_replays_remain_in_worker():
    before, after = previous.runtime_namespace(), current.runtime_namespace()
    for name in ('worker','complete_metadata','check_report','completed_native_result','restore_full_checkpoint',
                 'independent_full_loader','bitwise_array','bitwise_score','numpy_state_matches'):
        assert before[name].__code__ is after[name].__code__
    worker_names = after['worker'].__code__.co_names
    assert 'completed_native_result' in worker_names and 'complete_metadata' in worker_names
    assert 'restore_full_checkpoint' in worker_names and 'bitwise_array' in worker_names


def test_parent_only_defers_duplicate_numerical_check_and_keeps_every_other_gate():
    before, after = previous.main_ast(), current.main_ast()
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value,str):
                n.value = n.value.replace('scripts.flow_matching.run_timegan_full45_completion_replay_v4',
                                          'scripts.flow_matching.run_timegan_full45_completion_replay_v3')
            return n
        def visit_Call(self, n):
            if isinstance(n.func,ast.Name) and n.func.id == '_read_candidate_metadata':
                n.func.id = 'completed_native_result';n.args.append(ast.Name(id='queue',ctx=ast.Load()))
            return self.generic_visit(n)
    assert ast.dump(Reverse().visit(after)) == ast.dump(before)


def test_parent_candidate_loader_does_not_call_any_numerical_verifier(monkeypatch):
    calls=[]
    base=current.original.original.base
    monkeypatch.setattr(base,'read',lambda path:calls.append(path) or dict(status='candidate'))
    monkeypatch.setattr(current.original,'completed_native_result',lambda *args:(_ for _ in ()).throw(AssertionError('heavy verifier in parent')))
    assert current.read_candidate_metadata(dict(result_path='candidate.json')) == dict(status='candidate')
    assert len(calls) == 1
