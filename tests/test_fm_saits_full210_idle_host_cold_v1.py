import ast
import inspect
import json
import subprocess
import sys

from scripts.flow_matching import run_saits_full210_idle_host_cold_v1 as s


def test_exact_idle_identity_and_no_numeric_or_owned_shared_lock_required():
    state = dict(pid=2972, create_time=s.OWNER['create_time'], active_pid=None,
                 shared_lock_owned=False, state='waiting_for_memory')
    assert s.idle_allowed(state, s.OWNER, [])
    assert s.idle_allowed(state, s.OWNER, [dict(name='conhost.exe', exe='C:/Windows/System32/conhost.exe')])
    for key, value in [('pid', 2973), ('create_time', 1), ('active_pid', 4),
                       ('shared_lock_owned', True), ('state', 'running')]:
        assert not s.idle_allowed(dict(state, **{key:value}), s.OWNER, [])
    assert not s.idle_allowed(state, s.OWNER, [dict(name='python.exe', exe='C:/Python/python.exe')])
    assert not s.idle_allowed(state, s.OWNER, [dict(name='conhost.exe', exe='D:/other/conhost.exe')])


def test_cold_compilation_reuses_original_admission_order_and_only_changes_state_path():
    runtime = s.read(s.ROOT / s.RUNTIME)
    c = dict(original_runtime=s.RUNTIME, state_root='F:/isolated-metadata-state')
    ns, original_runtime = s.namespace(c)
    assert original_runtime == runtime
    changed = ns['read'](s.ROOT / s.RUNTIME)
    assert {k for k in changed if changed[k] != runtime[k]} == {'state_root'}
    queue = ns['verify_binding'](runtime)
    jobs = ns['_scheduler_jobs'](queue, runtime)
    assert len(jobs) == len(queue['jobs']) == 210
    assert {j['config'] for j in jobs} == {j['config'] for j in queue['jobs']}
    assert ns['_job_admission'](runtime, s.SELECTED) == runtime['selected_recovery_admission_settings']
    assert ns['_job_admission'](runtime, 'other') == runtime['settings']
    assert ns['_job_admission'](runtime, s.SELECTED)['minimum_commit_headroom_bytes'] == 40 * 2**30
    assert ns['_job_admission'](runtime, 'other')['minimum_commit_headroom_bytes'] == 28 * 2**30
    original_tree = ast.parse(inspect.getsource(ns['original'].main))
    tree = ns['controller_ast']()
    class Undo(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_job_admission':
                return ast.parse("runtime['settings']", mode='eval').body
            if isinstance(node.func, ast.Name) and node.func.id == '_scheduler_jobs':
                return ast.parse("queue['jobs']", mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(Undo().visit(tree)) == ast.dump(original_tree)


def test_full_command_and_creation_time_are_all_identity_fields():
    class Process:
        pid = s.OWNER['pid']
        def create_time(self): return s.OWNER['create_time']
        def cmdline(self): return s.OWNER['command']
    assert s.exact_owner(Process(), s.OWNER)
    for key, value in [('pid', 3), ('create_time', 0), ('command', ['other'])]:
        assert not s.exact_owner(Process(), dict(s.OWNER, **{key:value}))


def test_pending_saved_epoch_guard_is_reused_without_claiming_its_completion():
    c = dict(original_runtime=s.RUNTIME, state_root='F:/isolated-metadata-state')
    ns, runtime = s.namespace(c)
    _, proof, incident = ns['verify'](runtime)
    ns['predecessor_absent'](incident)
    bound = s.selected_resume_binding(runtime)
    assert bound['state'] == 'pending_exact_saved_epoch271_resume'
    assert bound['saved_epoch'] == 271 and bound['saved_optimizer_updates'] == 1897
    ns['saved_epoch_still_current'](proof)


def test_cold_namespace_checks_all210_without_importing_numpy_or_torch():
    code = ('import sys,json,psutil;from scripts.flow_matching import run_saits_full210_idle_host_cold_v1 as s;'
        's.namespace(dict(original_runtime=s.RUNTIME,state_root="F:/isolated-test-state"));'
        'assert not any(k in sys.modules for k in ["numpy","torch","pandas"]);'
        'print(json.dumps({"private":psutil.Process().memory_info().private}))')
    result = subprocess.run([sys.executable,'-B','-c',code],cwd=s.ROOT,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['private'] < 64 * 2**20
