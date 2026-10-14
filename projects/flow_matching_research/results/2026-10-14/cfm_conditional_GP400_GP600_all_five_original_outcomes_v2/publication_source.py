"""Verify and publish all five original budget outcomes, keeping missing failed metrics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil
from scripts.flow_matching import audit_cfm_gp_conditional_budget_outcomes_v2 as module

root = module.ROOT
folder = root / 'projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP400_GP600_all_five_original_outcomes_v2'
configpath = root / 'configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_outcomes.2026-10-14_v3.json'
config = module.read(configpath)
module.verify_config(config)
reportpath = root / config['output_report']
report = module.read(reportpath)
statepath = root / config['state_root'] / 'status.json'
state = module.read(statepath)
assert state['state'] == 'completed_all_original_full_outcomes_with_preserved_failure'
assert state['configuration_sha256'] == module.sha(configpath)
assert state['output_report_sha256'] == module.sha(reportpath)
assert report['frozen_complete_snapshot_sha256'] == module.sha(root / config['frozen_complete_snapshot'])
assert report['all_five_original_full_training_and_scoring_outcomes_audited']
assert report['original_required_trained_models'] == 10
assert report['complete_finite_grid_models'] == 9 and report['full_once_observed_failed_grid_models'] == 1
assert report['all_original_grid_coordinates_checked'] == 400000
assert report['full_five_seed_finite_paired_performance_difference'] is None
assert not report['all_ten_original_finite_native_certificates_passed'] and not report['failed_seed_exclusion']
fresh = module.summarize_outcomes(report['rows'])
for epoch, summary in fresh.items():
    assert all(report['budget_outcome_summaries'][epoch][key] == value for key, value in summary.items())
assert report['budget_outcome_summaries']['600']['full_five_seed_MSE_mean'] is None
assert all(r['actual_first400_loss_prefix']['first400_epoch_losses_bitwise_identical'] for r in report['rows'])
(folder / 'completed_actual_outcome_controller_state.json').write_bytes(statepath.read_bytes())
(folder / 'outcome_audit_source.py').write_bytes((root / 'scripts/flow_matching/audit_cfm_gp_conditional_budget_outcomes_v2.py').read_bytes())
(folder / 'outcome_completion_source.py').write_bytes((root / 'scripts/flow_matching/run_cfm_gp_budget_outcome_completion_v2.py').read_bytes())
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
baseline = report['budget_outcome_summaries']['400']
lines = ['CFM-TS / Appendix D Sine / conditional Gaussian GP / all five original400-versus600 outcomes\n',
    'All ten original full-training attempts and all400000 original grid coordinates were audited: nine complete finite native/independent model replays and one complete once-observed numerical failure. The original five paired joint data/training seeds1103..1107 remain present. This closes attempt-outcome accounting; it does not claim ten successful finite score certificates.\n',
    '| Joint seed | 400-epoch MSE | 600-epoch MSE | 600 outcome |',
    '| ---: | ---: | ---: | --- |']
for row in report['rows']:
    lines.append('| %d | %.12g | %s | %s |' % (row['seed'], row['400']['MSE'],
        'undefined' if row['600']['MSE'] is None else '%.12g' % row['600']['MSE'], row['600']['status']))
lines += ['', '400 epochs/250000 updates per model: full-five MSE mean %.12g, sample standard deviation %.12g; hierarchical joint-seed/whole-trajectory bootstrap95%% interval %s,10000 resamples with frozen seed20261014.' % (
        baseline['full_five_seed_MSE_mean'], baseline['full_five_seed_MSE_sample_std'],
        str(baseline['hierarchical_seed_whole_trajectory_bootstrap_mean_CI95'])),
    '', '600 epochs/375000 updates per model: four finite full80x500 grids and the failed seed1104 trajectory284. Full-five mean, standard deviation, bootstrap interval and paired mean difference remain undefined. No average or uncertainty interval over only the four surviving seeds is promoted as complete performance. Failure fraction1/5 describes this small frozen cohort, not a general population failure rate.',
    '', 'The original first400 epoch loss histories are bitwise identical within every pair. Intermediate model weights/optimizer states were not retained, so their equality and an isolated training-RNG effect are not certified. Successful pairs show both improvement and deterioration; one600 model fails. These observed outcomes do not establish a training root cause or an author-code bug.',
    '', 'The previously idle finite-only coordinator was handed off only after exact PID/creation/command/state/descendant checks. No numerical child was stopped or restarted; original training, solver, score tolerance and resource guards were unchanged. The new coordinator waits for all original outcomes and reports the preserved failure explicitly. Original v1 comparison registration and failed-model proof remain preserved.',
    '', 'This Appendix80-test cohort remains separate from Main20-test/93600-update recipes and the Main400 five-seed cohort. Original-author equivalence and strict TAB remain unproved. The full285 CFM routes and45-paper/681-method-record/273-data-task research scope remain required.',
    '', 'Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ']
(folder / 'README.md').write_text('\n'.join(lines) + '\n')
handles = []
for pid, expected in [(41992, 1791948094.8603492), (25120, 1791941882.8006165), (35452, 1791769181.137207)]:
    p = psutil.Process(pid)
    assert p.create_time() == expected
    handles.append(dict(pid=pid, create_time=p.create_time(), command=p.cmdline(), cpu_seconds=sum(p.cpu_times()[:2]),
        children=[dict(pid=c.pid, create_time=c.create_time(), command=c.cmdline()) for c in p.children(recursive=True)
            if 'python' in c.name().lower()]))
(folder / 'actual_remaining_full_scope_live_source_handles.json').write_text(json.dumps(dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_live_workers=handles, original_scope=config['original_scope'], all_original_experiments_complete=False), indent=2) + '\n')
print(json.dumps(dict(original_seed_pairs_audited=5, finite_models=9, full_once_failure=1,
    baseline400=baseline, full600_MSE=None)))
