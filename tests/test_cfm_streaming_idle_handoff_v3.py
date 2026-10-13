import ast
import hashlib
import inspect
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as current


def test_streamed_hash_includes_every_byte_and_final_short_block(tmp_path):
    data=bytes(range(256))*501+bytes(range(53));p=tmp_path/'data';p.write_bytes(data)
    assert current.streaming_sha(p)==hashlib.sha256(data).hexdigest()


def test_native_registration_only_changes_import_to_same_pure_function_binding():
    actual=current.native_registration_ast();body=actual.body[0].body
    for index,node in enumerate(body):
        if isinstance(node,ast.Assign) and getattr(node.value,'id',None)=='_fingerprint':
            body[index]=ast.parse('from scripts.flow_matching.run_cfm_ts_original_job import fingerprint').body[0]
    assert ast.dump(actual)==ast.dump(ast.parse(inspect.getsource(current.original.native_registration)))


def test_active_child_or_lost_proof_disallows_handoff():
    state=dict(active_pid=None,state='waiting_actual_original_training_completion',verified_original_ids=list(range(128)),failed_preserved={})
    assert current.idle_allowed('original',state,False,[])
    assert not current.idle_allowed('original',dict(state,active_pid=9),False,[])
    assert not current.idle_allowed('original',dict(state,verified_original_ids=list(range(127))),False,[])
    assert not current.idle_allowed('original',state,False,[dict(name='python.exe',exe='native-worker.exe')])


def test_lazy_metadata_fingerprint_matches_actual_completed_original_result():
    values=current.namespace({'preserved_verified_records':'unused','expected_verified_models':128})
    result=current.original.read(current.ROOT/'experiments/runs/fm_cfm_ts_original_v1/jobs/cfm_ts_original__appendix_d__gp_paper_literal_e400__sine__seed1103/result.json')
    assert values['_fingerprint'](result['job'])==result['job_sha256']
    assert values['controller'].__code__ is current.original.controller.__code__
