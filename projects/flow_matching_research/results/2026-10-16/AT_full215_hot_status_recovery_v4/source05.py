import ast
import inspect
from types import FunctionType
import pytest
from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v3 as s


def test_only_the_proven_missing_handoff_argument_changes():
    old=ast.parse(inspect.getsource(s.previous.controller));new=s.controller_tree()
    class Reverse(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='transfer':
                assert len(n.args)==3 and n.args[-1].value==10;n.args.pop()
            return n
    assert ast.dump(old)==ast.dump(Reverse().visit(new))


def test_functiontype_really_loses_defaults_and_explicit_poll_binds():
    fn=s.previous.handoff.handoff
    stripped=FunctionType(fn.__code__,fn.__globals__)
    with pytest.raises(TypeError):inspect.signature(stripped).bind({},None)
    inspect.signature(stripped).bind({},None,10)


def test_actual_private_controller_dispatches_transfer_and_retains_original_worker(monkeypatch,tmp_path):
    # No OS process is signalled: replace the transfer code and worker with pure fixtures.
    import scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 as cold
    captured=[]
    class Handle:
        def close(self):pass
    monkeypatch.setattr(cold,'install_facades',lambda:dict(exclusive_lock=lambda p:Handle()))
    namespace={}
    exec('def transfer_fixture(role,base,poll_seconds):\n    capture(role,base,poll_seconds)\n',namespace)
    monkeypatch.setattr(s.previous.handoff,'capture',lambda *a:captured.append(a),raising=False)
    monkeypatch.setattr(s.previous.handoff,'handoff',namespace['transfer_fixture'])
    monkeypatch.setattr(s.previous,'no_original_attempts',lambda c:True)
    monkeypatch.setattr(s.previous.original,'controller',lambda runtime,source,selected:None)
    runtime=dict(state_root=str(tmp_path/'isolated'),predecessor_owner=dict(pid=123,create_time=1,command=['fixture']))
    s.private_controller()(runtime,{}, {})
    assert len(captured)==1 and captured[0][2]==10


def test_all_original215_tasks_and_every_resource_rule_survive_both_metadata_repairs():
    old=s.read(s.ROOT/s.previous.OLD_CONFIG);new=s.registration()
    assert new['tasks']==old['tasks'] and len(new['tasks'])==215
    assert new['settings']==old['settings'] and new['resource_lock']==old['resource_lock']
    assert new['required_models']==500 and new['required_tasks']==215
    assert new['preserved_pre_handoff_failure']['previous_handoff_performed'] is False
    assert not new['preserved_pre_handoff_failure']['previous_numerical_worker_started']
