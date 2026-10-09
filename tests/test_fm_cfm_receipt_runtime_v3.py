import json
from scripts.flow_matching.run_cfm_ts_receipt_runtime_v3 import write_with_parents


def test_receipt_directory_is_created_and_atomic_payload_is_retained(tmp_path):
    path=tmp_path/'new_lane/receipts/job.json'
    payload={'complete':True,'resource_observations':{'resource_guard_aborted':False},'exit_code':0}
    write_with_parents(path,payload)
    assert json.loads(path.read_text(encoding='utf-8'))==payload
    assert not path.with_suffix('.json.tmp').exists()
    write_with_parents(path,{'complete':False})
    assert json.loads(path.read_text(encoding='utf-8'))=={'complete':False}
