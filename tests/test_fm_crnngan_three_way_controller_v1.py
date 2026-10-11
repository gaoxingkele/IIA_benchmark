import json
from types import SimpleNamespace
import pytest

from scripts.flow_matching import run_crnngan_three_way_repeatability_v1 as controller
from tests.test_fm_crnngan_three_way_repeatability_v1 import proof


def test_live_original_diagnostic_is_waited_for_and_missing_handle_is_not_restarted(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    config = dict(dependency_output_directory='original')
    directory = tmp_path / 'original'
    directory.mkdir()
    assert not controller.dependency_ready(config, dict(pid=1))
    with pytest.raises(RuntimeError, match='missing'):
        controller.dependency_ready(config, None)
    (directory / 'guard_receipt.json').write_text(json.dumps(dict(exit_code=0, passed=True, resource_guard_aborted=False)))
    (directory / 'original_four_epoch_repeatability.json').write_text(json.dumps(proof()))
    assert not controller.dependency_ready(config, dict(pid=1))
    assert controller.dependency_ready(config, None)
    (directory / 'guard_receipt.json').write_text(json.dumps(dict(exit_code=1, passed=False, resource_guard_aborted=True)))
    with pytest.raises(RuntimeError, match='unsuccessfully'):
        controller.dependency_ready(config, None)


def test_controller_rejects_reused_pid_or_changed_command(monkeypatch):
    process = SimpleNamespace(pid=123, create_time=lambda: 42.0,
        cmdline=lambda: ['python', 'original'], cpu_times=lambda: (2.0, 3.0))
    monkeypatch.setattr(controller.psutil, 'Process', lambda pid: process)
    assert controller.owned_process(dict(pid=123, create_time=42.0, command_token='original'))['cpu_seconds'] == 5.0
    with pytest.raises(ValueError, match='reused'):
        controller.owned_process(dict(pid=123, create_time=41.0, command_token='original'))
    with pytest.raises(ValueError, match='command'):
        controller.owned_process(dict(pid=123, create_time=42.0, command_token='different'))
