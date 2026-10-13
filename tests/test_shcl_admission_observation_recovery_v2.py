import ast
import copy
import inspect
import subprocess
from unittest.mock import patch

import pytest
from scripts.flow_matching import run_shcl_admission_observation_recovery_v2 as current

CONFIG = 'configs/runtime/fm_shcl_native_admission_observation_recovery.v2.json'


def test_exact_original_source_control_flow_except_two_admission_queries_and_host_state():
    actual = current.main_ast()
    function = actual.body[0]
    function.body = [n for n in function.body if not (isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and getattr(n.value.func, 'id', None) == '_host_queue')]
    for node in ast.walk(actual):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == '_admission_gpu_snapshot':
            node.func.id = 'gpu_snapshot'
    assert ast.dump(actual) == ast.dump(ast.parse(inspect.getsource(current.original.main)))


@pytest.mark.parametrize('error', [subprocess.CalledProcessError(255, ['nvidia-smi']),
                                 subprocess.TimeoutExpired(['nvidia-smi'], 5)])
def test_unavailable_observation_cannot_admit_or_report_available_capacity(error):
    settings = {'gpu_uuid': 'original-uuid', 'minimum_gpu_free_bytes': 8 * 2**30}
    with patch.object(current.original, 'gpu_snapshot', side_effect=error):
        result = current.admission_gpu_snapshot(settings)
    assert result['gpu_uuid'] == settings['gpu_uuid'] and result['gpu_free_bytes'] == 0
    assert result['gpu_admission_observation_valid'] is False
    assert result['gpu_free_bytes'] < settings['minimum_gpu_free_bytes']


def test_wrong_uuid_or_invalid_observation_is_not_hidden_as_transient():
    with patch.object(current.original, 'gpu_snapshot', side_effect=ValueError('Exact registered CUDA UUID required')):
        with pytest.raises(ValueError, match='UUID'):
            current.admission_gpu_snapshot({})


def test_successful_observation_preserves_original_exact_capacity():
    result = {'gpu_uuid': 'original-uuid', 'gpu_free_bytes': 12000 * 2**20}
    with patch.object(current.original, 'gpu_snapshot', return_value=result):
        actual = current.admission_gpu_snapshot({})
    assert actual == dict(result, gpu_admission_observation_valid=True)


def test_full413_exact_scientific_job_and_settings_registration():
    config = current.NATIVE.read(current.ROOT / CONFIG)
    queue, incident = current.verify(config)
    assert len(queue['jobs']) == 413 and sum(j['mode'] == 'train' for j in queue['jobs']) == 49
    isolated = current.host_queue(queue, config)
    assert {k:v for k,v in isolated.items() if k != 'state_root'} == {k:v for k,v in queue.items() if k != 'state_root'}
    assert isolated['settings'] == queue['settings']
    assert all(j['native_DataLoader_workers'] == 8 for j in queue['jobs'])
    assert incident['original_jobs_dispatched'] == 0


@pytest.mark.parametrize('field,value', [('required_training_commands', 48), ('required_test_commands', 363)])
def test_no_original_command_can_be_removed(field, value):
    config = copy.deepcopy(current.NATIVE.read(current.ROOT / CONFIG));config[field] = value
    with pytest.raises(ValueError, match='All413'):
        current.verify(config)


def test_live_exact_predecessor_prevents_duplicate_controller():
    config = current.NATIVE.read(current.ROOT / CONFIG)
    incident = current.NATIVE.read(current.ROOT / config['preserved_incident'])
    with patch('psutil.Process') as process:
        process.return_value.create_time.return_value = incident['previous_owner']['create_time']
        process.return_value.cmdline.return_value = incident['previous_owner']['command']
        with pytest.raises(RuntimeError, match='still exists'):
            current.retired_owner_absent(incident)


def test_native_registration_probe_never_requires_host_process_dependency():
    with patch('sys.argv', ['probe', '--config', CONFIG, '--probe-only']), \
         patch.object(current, 'retired_owner_absent', side_effect=ImportError('host-only psutil missing')) as host:
        current.main()
    host.assert_not_called()
