"""Archive the sixth original saved-model failure after one full-grid observation."""
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_conditional_GP6001104_full80_failure_v1 as native

ROOT = native.ROOT
runtime_path = ROOT / 'configs/runtime/fm_cfm_conditional_GP6001104_full80_failure_observation.v1.json'
runtime = native.read(runtime_path)
job, model = native.verify(runtime)
output = Path(runtime['output_root'])
state = ROOT / runtime['state_root']
receipt = native.read(state / 'receipt.json')
assert receipt['exit_code'] == 0 and receipt['full80_failure_observation_captured']
assert not receipt['resource_guard_aborted'] and not receipt['original_benchmark_scoring_complete']
assert native.read(state / 'status.json')['active_pid'] is None
config_path = ROOT / 'configs/reproducibility/fm_cfm_conditional_GP6001104_full80_observed_arrays.2026-10-14_v1.json'
folder = ROOT / 'projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP6001104_closed_full80_failure_v1'
proof_path = folder / 'independent_full80_observed_arrays.json'
pins = [runtime_path, output / 'full80_native_failure_observation.json',
        output / 'all_original_grid_attempts.npz', state / 'receipt.json',
        ROOT / 'scripts/flow_matching/audit_cfm_conditional_GP6001104_full80_failure_v1.py',
        ROOT / 'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py',
        ROOT / 'scripts/flow_matching/audit_cfm_full80_observed_arrays_v2.py',
        Path(__file__).resolve()]
pins.extend(output / ('trajectory_%02d.json' % position) for position in range(80))
def spelling(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()

config = dict(native_configuration=spelling(runtime_path), output_report=spelling(proof_path),
    source_receipts=[dict(path=spelling(p), sha256=native.sha(p)) for p in pins],
    original_scope=runtime['original_scope'], citation=runtime['citation'],
    boundary='Independent all80x500 original grids; no ODE solver calls, optimizer updates, success-subset aggregate or changed numerical settings.')
assert not config_path.exists() and not folder.exists()
config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
proof = native.post_audit(config)
assert proof['canonical_job_id'] == job['id']
assert proof['attempted_original_trajectories'] == 80 and proof['finite_full_grid_trajectories'] == 79
assert proof['failed_trajectory_ids'] == [284]
assert proof['independent_checked_grid_coordinates'] == 40000
assert proof['full_original_MSE'] is None and not proof['original_scoring_certified_complete']
assert proof['additional_ODE_solver_calls'] == proof['additional_optimizer_updates'] == 0
assert proof['solver_parameters'] == runtime['solver_parameters']
folder.mkdir(exist_ok=False)
proof_path.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
sources = dict(independent_config=config_path, native_configuration=runtime_path,
    native_report=output / 'full80_native_failure_observation.json', native_guarded_receipt=state / 'receipt.json',
    native_completed_state=state / 'status.json', original_failed_receipt=ROOT / runtime['original_failed_receipt'],
    original_failed_progress=ROOT / runtime['original_failed_progress'])
for name, path in sources.items():
    (folder / (name + '.json')).write_bytes(path.read_bytes())
(folder / 'original_failed_stderr.log').write_bytes((ROOT / runtime['original_failed_stderr']).read_bytes())
for position in range(80):
    name = 'trajectory_%02d.json' % position
    (folder / name).write_bytes((output / name).read_bytes())
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job_id=job['id'],
    original_training_epochs=600, original_optimizer_updates=375000,
    full_once_observation_and_independent_coordinate_audit_completed=True,
    attempted_trajectories=80, finite_trajectories=79, failed_trajectory_ids=[284],
    original_full_grid_shape=[80, 500, 1], independently_checked_coordinates=40000,
    original_source_failure_after_trajectories=58, original_full_MSE=None,
    original_scoring_certified_complete=False, successful_subset_promoted_as_benchmark=False,
    numeric_solver_settings_changed=False, additional_optimizer_updates=0,
    original_model_sha256=runtime['checkpoint_sha256'], independent_proof_sha256=native.sha(proof_path),
    full_grid_arrays_path=str(output / 'all_original_grid_attempts.npz'),
    full_grid_arrays_sha256=proof['array_sha256'], failure_evidence=proof['failure_evidence'],
    training_root_cause_or_author_code_bug_certified=False, author_equivalence_certified=False,
    strict_TAB_result=False, conditional_GP600_all_five_finite_seed_mean_defined=False,
    original_scope=runtime['original_scope'], all_original_experiments_complete=False)
(folder / 'actual_complete_full80_failure_observation_closure.json').write_text(json.dumps(closure, indent=2) + '\n', encoding='utf-8')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.log -text -diff\n', encoding='utf-8')
(folder / 'README.md').write_text('''CFM-TS conditional Gaussian GP / Appendix D Sine / 600 epochs / seed 1104

The complete 375000-update saved model was restored without fitting. One observation attempted all 80 original trajectories at every original 500-point grid, using unchanged float32, dopri5, rtol 1e-7, atol 1e-9 and maximum 10000 steps. The original 30 GiB commit/16 GiB physical admission and 16 GiB emergency guards were retained. The worker exited normally without a resource abort; this means diagnostic coverage completed, not successful original benchmark scoring.

79 trajectories completed; trajectory 284 failed. Independent review checked all 40000 coordinates, original ordering, disjoint complete-trajectory split, times, truth, initial states, every finite trajectory error and the failed all-NaN grid. Full original MSE remains null. No finite-subset aggregate, failed-seed exclusion, additional training updates, solver relaxation or repeat-until-pass claim is made. The original source underflow at 58/80 and its stderr are preserved. The observation supports the recorded growth/nonfinite RHS evidence, while the training cause and an author-code bug remain unproved.

The planned conditional GP600 five-seed finite mean remains undefined with this failed seed. Other seeds and the broader 45-paper / 681-method-record / 273-data-task scope remain required.

Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ
''', encoding='utf-8')
print(json.dumps(dict(closed=True, finite=79, failed=[284], failure_evidence=proof['failure_evidence'])))
