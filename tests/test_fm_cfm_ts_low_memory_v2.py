from copy import deepcopy
import json
import subprocess
import sys

import pytest

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT,read
from scripts.flow_matching.register_cfm_ts_exact_recovery_v1 import validate_recovery


def recovery_fixture():
    original=read(ROOT/'configs/experiments/fm_cfm_ts_original_queue.v1.json')
    recovery=deepcopy(original);job=recovery['jobs'][0]
    job.update(recovery_original_job_id=job['id'],id=job['id']+'__retry',
        output_directory='experiments/runs/unit_only_recovery',artifact_directory='F:/unit_only_recovery')
    return original,recovery


def test_exact_recovery_rejects_budget_data_seed_or_model_changes():
    original,recovery=recovery_fixture();validate_recovery(original,recovery)
    for field in ['epochs','seed','data_sha256','model_overrides']:
        wrong=deepcopy(recovery);wrong['jobs'][0][field]='changed'
        with pytest.raises(ValueError,match='scientific protocol'):validate_recovery(original,wrong)
    wrong=deepcopy(recovery);wrong['resource_policy']['cpu_threads']=99
    with pytest.raises(ValueError,match='safeguards'):validate_recovery(original,wrong)


def test_exact_recovery_cannot_overwrite_original_or_duplicate_seed_slot():
    original,recovery=recovery_fixture()
    recovery['jobs'][0]['artifact_directory']=original['jobs'][0]['artifact_directory']
    with pytest.raises(ValueError,match='overwrite'):validate_recovery(original,recovery)
    original,recovery=recovery_fixture();recovery['jobs'][1]=deepcopy(recovery['jobs'][0])
    with pytest.raises(ValueError):validate_recovery(original,recovery)


def test_low_memory_parent_stays_light_after_real_full_artifact_validation(tmp_path):
    # A real existing full experiment is verified in a separate interpreter.
    from scripts.flow_matching.prepare_cfm_ts_original_data import sha
    original=read(ROOT/'configs/experiments/fm_cfm_ts_original_queue.v1.json')
    done=next(j for j in original['jobs'] if (ROOT/j['output_directory']/'result.json').exists())
    runtime={'controllers':{'original':{'queue':'configs/experiments/fm_cfm_ts_original_queue.v1.json',
        'queue_sha256':sha(ROOT/'configs/experiments/fm_cfm_ts_original_queue.v1.json')}},'source_receipts':[]}
    path=tmp_path/'runtime.json';path.write_text(json.dumps(runtime),encoding='utf-8')
    result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
        '--runtime-config',str(path),'--probe-only','--probe-completed-id',done['id']],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
    probe=json.loads(result.stdout)
    assert probe['full_completed_artifact_verified']
    assert not probe['numpy_imported'] and not probe['torch_imported']
    assert probe['private_bytes']<100*2**20
