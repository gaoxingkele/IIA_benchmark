from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import continue_three_gan_idle_hosts_cold_v1 as s


@pytest.fixture(scope='module')
def config():
    return s.registration()


def test_all_original_jobs_native_budgets_and_every_resource_policy_remain(config):
    assert set(config['roles']) == set(s.ROLES)
    for name, role in config['roles'].items():
        module, original, queue = s.original(role)
        assert role['required_original_slots'] == len(queue['jobs']) == (4 if role['native_role'] is None else 45)
        expected = original['settings'] if role['native_role'] is None else original['roles'][role['native_role']]['settings']
        assert role['settings'] == expected
        assert role['resource_lock'] == original['resource_lock'] and role['cpu_threads'] == original['cpu_threads']
        assert role['previous_owner']['command'][6:] == role['arguments']
        if role['native_role'] is None:
            assert (role['required_periodic_generations'], role['required_trained_evaluators'], role['required_native_network_checkpoints']) == (29, 580, 116)
        else:
            assert role['full_native_budget'] == dict(E0=50000, GS=50000, G=100000, E=100000, D_maximum=50000,
                discriminative_repeats=10, discriminative_updates_per_repeat=2000,
                predictive_repeats=10, predictive_updates_per_repeat=5000)
    assert config['roles']['timegan_original45']['stronger_admission_commit_bytes'] == 40*2**30


@pytest.mark.parametrize('name', s.ROLES)
def test_entire_original_main_code_is_identical_only_metadata_callback_binding_differs(config, name, tmp_path):
    role = config['roles'][name]
    module, _, _ = s.original(role)
    original = module.runtime_namespace()['main'] if role['entry'].endswith('_v4') else module.main
    continued = s.continued_main(role, tmp_path)
    assert continued.__code__ == original.__code__
    callback = 'handoff' if role['native_role'] is None else 'safe_handoff'
    assert continued.__globals__[callback] is not original.__globals__[callback]
    for key in original.__code__.co_names:
        if key != callback and key in original.__globals__:
            assert continued.__globals__[key] is original.__globals__[key] or key == 'worker'


@pytest.mark.parametrize('name', s.ROLES)
def test_metadata_callback_accepts_only_original_route_after_new_verified_handoff(config, name, tmp_path, monkeypatch):
    role = config['roles'][name]
    module, original, queue = s.original(role)
    called = []
    monkeypatch.setattr(s, 'already_handed_off', lambda r, b: called.append((r, b)))
    function = s.continued_main(role, tmp_path)
    callback = function.__globals__['handoff' if role['native_role'] is None else 'safe_handoff']
    if role['native_role'] is None:
        args = (original, queue, s.ROOT/role['previous_state_root'], lambda v: None)
    else:
        args = (role['native_role'], original['roles'][role['native_role']], s.ROOT/role['previous_state_root'], lambda v: None, original['poll_seconds'])
    callback(*args)
    assert called == [(role, tmp_path)]
    with pytest.raises(ValueError):
        callback(*args[:-1])
    bad = list(args)
    bad[0] = 'other-original-route'
    with pytest.raises(ValueError):
        callback(*bad)


@pytest.mark.parametrize('name', s.ROLES)
def test_idle_handoff_rejects_any_native_child_active_pid_or_owned_lock(config, name):
    role = config['roles'][name]
    owner = role['previous_owner']
    state = dict(pid=owner['pid'], create_time=owner['create_time'], active_pid=None, shared_lock_owned=False,
                 state='waiting_full_gpu_resources' if role['native_role'] is None else 'waiting_for_memory')
    assert s.idle_allowed(role, state, owner, [])
    assert not s.idle_allowed(role, dict(state, active_pid=1234), owner, [])
    assert not s.idle_allowed(role, dict(state, shared_lock_owned=True), owner, [])
    assert not s.idle_allowed(role, state, owner, [dict(name='python.exe', exe='C:/Python/python.exe')])


@pytest.mark.parametrize('fault', ['missing_handoff', 'wrong_owner', 'native_stopped', 'old_still_live', 'changed_prior_metadata'])
def test_metadata_reuse_requires_original_bytes_and_actual_new_safe_handoff(config, fault, tmp_path, monkeypatch):
    role = config['roles']['timegan_energy']
    proof = dict(previous_owner=role['previous_owner'], original_native_child_stopped=False, waiting_controller_only_stopped=True)
    if fault == 'wrong_owner':
        proof['previous_owner'] = dict(role['previous_owner'], create_time=0)
    if fault == 'native_stopped':
        proof['original_native_child_stopped'] = True
    if fault != 'missing_handoff':
        s.common.exclusive(tmp_path/'handoff.json', proof)
    import psutil
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: fault == 'old_still_live')
    if fault == 'changed_prior_metadata':
        original_sha = s.sha
        monkeypatch.setattr(s, 'sha', lambda path: '0'*64 if Path(path) == s.ROOT/role['prior_metadata'][0]['path'] else original_sha(path))
    with pytest.raises((ValueError, FileNotFoundError)):
        s.already_handed_off(role, tmp_path)


@pytest.mark.parametrize('fault', ['queue_count', 'source', 'cpu_guard', 'gpu_guard', 'old_identity', 'proof'])
def test_registration_rejects_scope_and_safeguard_shortcuts(config, fault):
    changed = deepcopy(config)
    if fault == 'queue_count': changed['roles']['timegan_original45']['required_original_slots'] = 1
    elif fault == 'source': changed['source_receipts'][0]['sha256'] = '0'*64
    elif fault == 'cpu_guard': changed['roles']['timegan_energy']['settings']['emergency_commit_headroom_bytes'] = 1
    elif fault == 'gpu_guard': changed['roles']['gtgan_stock4']['settings']['minimum_gpu_free_bytes'] = 1
    elif fault == 'old_identity': changed['roles']['timegan_energy']['previous_owner']['create_time'] = 0
    else: changed['roles']['gtgan_stock4']['prior_metadata'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError):
        s.verify(changed)


@pytest.mark.parametrize('name', s.ROLES)
def test_real_controller_compilation_is_cold_without_importing_numeric_runtimes(name):
    code = ('from scripts.flow_matching import continue_three_gan_idle_hosts_cold_v1 as s;'
            'import sys,json,psutil;from pathlib import Path;'
            'c=s.registration();s.common.install_cached_loader(lambda x:None);'
            f'f=s.continued_main(c["roles"][{name!r}],Path("not-executed-metadata-probe"));'
            'assert not any(k in sys.modules for k in ["numpy","torch","tensorflow"]);'
            'print(json.dumps(dict(private=psutil.Process().memory_info().private)))')
    process = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code], cwd=s.ROOT,
                             capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout+process.stderr
    assert json.loads(process.stdout)['private'] < 128*2**20
