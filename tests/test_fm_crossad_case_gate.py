import json
from pathlib import Path
import pytest
from scripts.flow_matching.run_crossad_case_gated_queue import native_proof
from scripts.flow_matching.prepare_tsad_execution import sha
ROOT=Path(__file__).resolve().parents[1]


def test_scheduler_keeps_all_full_budget_jobs_and_changes_only_execution_metadata():
    parent=json.loads((ROOT/'configs/experiments/fm_crossad_complete_queue.v4.json').read_text())
    child=json.loads((ROOT/'configs/experiments/fm_crossad_complete_queue.v5.json').read_text())
    assert len(child['jobs'])==139 and child['settings']['preflight_parent_sha256']==sha(ROOT/child['settings']['preflight_parent_queue'])
    for a,b in zip(parent['jobs'],child['jobs']):
        assert {k:v for k,v in a.items() if k!='frozen_files'}=={k:v for k,v in b.items() if k!='frozen_files'}
        assert b['frozen_files'][:len(a['frozen_files'])]==a['frozen_files']


def test_native_case_proof_never_accepts_reduced_batch_or_changed_model(tmp_path):
    job={'id':'one','dataset':'PSM','train_parameters':{'batch_size':128},'model_parameters':{'seq_len':192},
         'activation_checkpointing':'pure_layer_v2','raw_sha256':'raw'}
    parent=tmp_path/'parent.json';parent.write_text(json.dumps({'jobs':[job]}))
    settings={'preflight_parent_queue':'parent.json','preflight_parent_sha256':sha(parent)}
    report=tmp_path/'docs/reports/fm_crossad_complete_preflight_v4_PSM_2026-10-09.json';report.parent.mkdir(parents=True)
    values={'queue_sha256':sha(parent),'records':[{'dataset':'PSM','finite_backward':True,'deterministic_release_inference':True,'native_batch_shape':[1,192,25]}]}
    report.write_text(json.dumps(values))
    with pytest.raises(AssertionError):native_proof(tmp_path,job,settings)
    values['records'][0]['native_batch_shape'][0]=128;report.write_text(json.dumps(values))
    assert native_proof(tmp_path,job,settings)['native_batch_shape']==[128,192,25]
    changed=dict(job,model_parameters={'seq_len':96})
    with pytest.raises(AssertionError):native_proof(tmp_path,changed,settings)
    report.unlink()
    assert native_proof(tmp_path,job,settings) is None
