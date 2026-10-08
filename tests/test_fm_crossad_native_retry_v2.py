import json
from pathlib import Path

import pytest
from scripts.flow_matching.register_crossad_native_retry_v2 import build


def fixture(root, aborted=True):
    own=root/'scripts/flow_matching/register_crossad_native_retry_v2.py'
    own.parent.mkdir(parents=True);own.write_text('source')
    settings={'minimum_commit_headroom_bytes':16*1024**3,'emergency_commit_headroom_bytes':8*1024**3,
              'minimum_available_memory_bytes':24*1024**3,'python':'original-python'}
    predecessor={'settings':settings,'cases':[{'name':'SMD'}],'source_receipts':[],
                 'parent_queue':'original.json','parent_queue_sha256':'original-hash'}
    parent=root/'configs/experiments/fm_crossad_preflight_resource_retry.v1.json'
    parent.parent.mkdir(parents=True);parent.write_text(json.dumps(predecessor))
    base=root/'experiments/runs/fm_crossad_complete_preflight_v4'
    for name in ('SMD','PSM_row4'):
        failure=base/name/'failure.json';failure.parent.mkdir(parents=True)
        failure.write_text(json.dumps({'resource_usage':{'resource_guard_aborted':aborted,
                                                        'peak_tree_private_bytes':16466898944}}))
    return predecessor,base


def test_only_uncovered_failure_reserved_without_changing_emergency_or_parent(tmp_path):
    predecessor,base=fixture(tmp_path)
    before={p:p.read_bytes() for p in base.glob('*/failure.json')}
    result=build(tmp_path)
    assert [c['name'] for c in result['cases']]==['PSM_row4']
    assert result['cases'][0]['row']==4 and result['cases'][0]['dataset']=='PSM'
    assert result['settings']==dict(predecessor['settings'],minimum_commit_headroom_bytes=26*1024**3)
    assert result['parent_queue_sha256']==predecessor['parent_queue_sha256']
    assert all(p.read_bytes()==b for p,b in before.items())


def test_source_error_not_retried_as_resource_failure(tmp_path):
    fixture(tmp_path,aborted=False)
    with pytest.raises(ValueError,match='source repair'):build(tmp_path)


def test_complete_proof_excludes_failure(tmp_path):
    fixture(tmp_path)
    proof=tmp_path/'docs/reports/fm_crossad_complete_preflight_v4_PSM_row4_2026-10-09.json'
    proof.parent.mkdir(parents=True);proof.write_text('{}')
    with pytest.raises(ValueError,match='No uncovered'):build(tmp_path)
