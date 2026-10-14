"""Hand off only the proven idle impossible-metric waiter; retain all original science."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from types import FunctionType
import psutil
from scripts.flow_matching import run_cfm_gp_budget_outcome_completion_v2 as successor
from scripts.flow_matching import audit_cfm_gp_conditional_budget_outcomes_v2 as analysis

ROOT = analysis.ROOT
cold = successor.planned.previous.cold
folder = ROOT / 'projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP400_GP600_all_five_original_outcomes_v2'
oldconfigpath = ROOT / 'configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_pairs.2026-10-14_v1.json'
oldstatepath = ROOT / 'experiments/runs/fm_cfm_conditional_GP_five_budget_pairs_v1/status.json'
oldstate = analysis.read(oldstatepath)
old = psutil.Process(5908)
assert old.create_time() == 1791941296.4286673
expected = dict(pid=old.pid, create_time=old.create_time(), command=old.cmdline())
assert oldstate['configuration_sha256'] == analysis.sha(oldconfigpath)
descendants = [dict(name=p.name(), exe=p.exe()) for p in old.children(recursive=True)]
assert successor.idle_handoff_allowed('original', oldstate, False, descendants)
initial = ROOT / 'configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_outcomes.2026-10-14_v2.json'
config = analysis.read(initial)
config.update(primary_runtime='configs/runtime/fm_cfm_completion_replay_queue.v4.json',
    paired_canonical_ids=successor.planned.identities(),
    state_root='experiments/runs/fm_cfm_conditional_GP_all_five_outcome_completion_v2', poll_seconds=10,
    frozen_complete_snapshot=(folder / 'frozen_all_original_native_records.json').relative_to(ROOT).as_posix(),
    previous_idle_metadata_owner=expected, previous_idle_metadata_state=oldstatepath.relative_to(ROOT).as_posix())
pins = [initial, ROOT / 'scripts/flow_matching/run_cfm_gp_budget_outcome_completion_v2.py',
    ROOT / 'tests/test_cfm_gp_budget_outcome_completion_v2.py', oldconfigpath,
    ROOT / 'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py']
config['source_receipts'] += [dict(path=p.relative_to(ROOT).as_posix(), sha256=analysis.sha(p)) for p in pins]
configpath = ROOT / 'configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_outcomes.2026-10-14_v3.json'
assert not configpath.exists()
configpath.write_text(json.dumps(config, indent=2) + '\n')
analysis.verify_config(config)
(folder / 'completion_configuration_v3.json').write_bytes(configpath.read_bytes())
(folder / 'preserved_previous_finite_only_reader_state.json').write_bytes(oldstatepath.read_bytes())
base = ROOT / config['state_root'] / 'handoff'
base.mkdir(parents=True, exist_ok=False)
route = dict(previous_owner=expected, previous_state_path=oldstatepath.relative_to(ROOT).as_posix(),
    runtime_configuration=configpath.relative_to(ROOT).as_posix())
handoff = FunctionType(cold.handoff.safe_handoff.__code__,
    dict(cold.handoff.__dict__, handoff_phase_allowed=successor.idle_handoff_allowed))
handoff('original', route, base, lambda state: cold.original.write(base / 'status.json', state), 2)
assert not psutil.pid_exists(expected['pid'])
(folder / 'actual_idle_metadata_handoff.json').write_bytes((base / 'handoff.json').read_bytes())
command = [str(ROOT / '.venv/cfmts-author-py312/Scripts/python.exe'), '-X', 'utf8', '-u', '-m',
    'scripts.flow_matching.run_cfm_gp_budget_outcome_completion_v2', '--config', configpath.relative_to(ROOT).as_posix()]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
    CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
with (base / 'successor.stdout.log').open('xb') as out, (base / 'successor.stderr.log').open('xb') as err:
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
        creationflags=subprocess.CREATE_NO_WINDOW)
statuspath = ROOT / config['state_root'] / 'status.json'
status = None
for _ in range(30):
    if child.poll() is not None:
        raise RuntimeError('Inspect the preserved metadata successor failure; never restart automatically')
    if statuspath.exists():
        candidate = analysis.read(statuspath)
        p = psutil.Process(candidate['pid'])
        if p.create_time() == candidate['create_time'] and p.cmdline()[1:] == command[1:]:
            status = candidate
            break
    time.sleep(1)
assert status is not None
receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(), previous_idle_metadata_owner=expected,
    actual_successor=dict(pid=status['pid'], create_time=status['create_time'], command=psutil.Process(status['pid']).cmdline()),
    state=status, native_training_or_replay_workers_stopped=0, model_guards_or_tolerances_changed=False,
    original_required_trained_model_outcomes=10, original_required_joint_seed_pairs=5,
    original_scope=config['original_scope'], all_original_experiments_complete=False)
(folder / 'actual_outcome_completion_controller_launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
(folder / 'registration_and_handoff_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
print(json.dumps(dict(old_idle_reader_replaced=expected['pid'], actual_new_reader=status['pid'], state=status['state'])))
