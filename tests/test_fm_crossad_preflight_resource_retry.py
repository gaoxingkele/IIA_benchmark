import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching.run_crossad_preflight_resource_retry import ROOT, validate_proof, verify


def settings():
    return json.loads((ROOT/'configs/experiments/fm_crossad_preflight_resource_retry.v1.json').read_text(encoding='utf-8'))


def test_all_retry_cases_are_preserved_resource_incidents_at_unchanged_full_budgets():
    config=settings()
    verify(config)
    parent=json.loads((ROOT/config['parent_queue']).read_text(encoding='utf-8'))
    assert config['settings']==parent['settings']
    assert {c['name'] for c in config['cases']}=={'GECCO_row4','PSM_row3','SMD','SWAN','SWAT'}
    for case in config['cases']:
        original=json.loads((ROOT/case['original_failure_path']).read_text(encoding='utf-8'))
        assert original['resource_usage']['resource_guard_aborted']


def test_native_proof_rejects_reduced_batch_missing_backward_and_wrong_dataset(tmp_path):
    config=settings()
    source=ROOT/'docs/reports/fm_crossad_complete_preflight_v4_GECCO_2026-10-09.json'
    case={'dataset':'GECCO'}
    assert validate_proof(source,config,case)['native_batch_shape'][:2]==[128,128]
    value=json.loads(source.read_text(encoding='utf-8'))
    for key,replacement in [('native_batch_shape',[1,128,9]),('finite_backward',False),('dataset','PSM')]:
        changed=copy.deepcopy(value)
        changed['records'][0][key]=replacement
        # Proofs must remain under the registered root for their relative provenance.
        path=ROOT/'tmp/native_retry_invalid_proof.json'
        path.write_text(json.dumps(changed))
        with pytest.raises(AssertionError):
            validate_proof(path,config,case)
