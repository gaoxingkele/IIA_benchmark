import json

import pytest

from scripts.flow_matching import run_gtgan_energy_native_queue_v1 as controller


def test_gpu_execution_does_not_start_with_missing_full_data_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    queue = dict(construction_proof='proof.json', construction_audit_state_root='audit',
        construction_audit_config_sha256='frozen-config')
    assert controller.construction_ready(queue) is False


def test_a_proof_file_without_successful_worker_exit_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    (tmp_path / 'proof.json').write_text(json.dumps(dict(passed=True)))
    queue = dict(construction_proof='proof.json', construction_audit_state_root='audit',
        construction_audit_config_sha256='frozen-config')
    assert controller.construction_ready(queue) is False


def test_failed_native_audit_cannot_be_promoted_to_gpu_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    (tmp_path / 'audit').mkdir()
    (tmp_path / 'audit/receipt.json').write_text(json.dumps(dict(exit_code=1, full_native_energy_audit_passed=False)))
    (tmp_path / 'proof.json').write_text(json.dumps(dict(passed=True)))
    queue = dict(construction_proof='proof.json', construction_audit_state_root='audit',
        construction_audit_config_sha256='frozen-config')
    with pytest.raises(ValueError, match='successful actual worker exit'):
        controller.construction_ready(queue)
