import json
import subprocess
import sys

from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as s


def idle():
    return dict(pid=37280, create_time=1792129142.1064477, active_pid=None,
                state='waiting_actual_full_source_and_checkpoint_arrays')


def test_only_exact_idle_owner_without_numerical_descendants_is_eligible():
    state = idle()
    expected = s.EXPECTED_OWNER
    assert s.safe_idle(state, expected, [])
    assert s.safe_idle(state, expected, [dict(name='conhost.exe', exe='C:/Windows/System32/conhost.exe')])
    for field, value in [('pid', 37281), ('create_time', 1792129142.1064),
                         ('active_pid', 1), ('state', 'running_author')]:
        assert not s.safe_idle(dict(state, **{field: value}), expected, [])
    assert not s.safe_idle(state, expected, [dict(name='python.exe', exe='C:/Python/python.exe')])
    assert not s.safe_idle(state, expected, [dict(name='conhost.exe', exe='D:/other/conhost.exe')])


def test_identity_requires_pid_creation_time_and_full_command():
    class Process:
        pid = 37280
        def create_time(self): return s.EXPECTED_OWNER['create_time']
        def cmdline(self): return s.EXPECTED_OWNER['command']
    p = Process()
    assert s.exact_identity(p, s.EXPECTED_OWNER)
    for field, value in [('pid', 37281), ('create_time', 1), ('command', ['other'])]:
        assert not s.exact_identity(p, dict(s.EXPECTED_OWNER, **{field: value}))


def test_cold_lock_still_excludes_second_owner(tmp_path):
    path = tmp_path / 'controller.lock'
    first = s.cold_lock(path)
    assert first is not None
    try:
        assert s.cold_lock(path) is None
    finally:
        first.close()
    second = s.cold_lock(path)
    assert second is not None
    second.close()


def test_collector_import_keeps_original_numerical_validators_without_numerical_libraries():
    script = ('import json,sys,psutil;from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as s;'
              'assert not any(x in sys.modules for x in ["numpy","torch","pandas"]);'
              'print(json.dumps({"private_bytes":psutil.Process().memory_info().private}))')
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=s.ROOT,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['private_bytes'] < 64 * 2**20
