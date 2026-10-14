"""Audit every Main20-trajectory GP400/600 pair, retaining catastrophic finite errors."""
import ast
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import audit_cfm_gp_five_seed_targets_v1 as grouped
from scripts.flow_matching.audit_cfm_gp_conditional_five_seed_budgets_v1 import prefix_evidence

root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
runtimepaths = ['configs/runtime/fm_cfm_completion_replay_queue.v4.json',
    'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json', 'configs/runtime/fm_cfm_saved_score_full_replay.v1.json']
records, owners = {}, {}
for path in runtimepaths:
    runtime = read(root / path)
    for record in read(root / runtime['state_root'] / 'verified_full_model_records.json')['records']:
        identity = record['original_canonical_id']
        if identity in records:
            assert records[identity] == record
        else:
            records[identity], owners[identity] = record, runtime
canonical = {j['id']: j for j in read(root / 'configs/experiments/fm_cfm_ts_original_queue.v1.json')['jobs']}
pairs = [['cfm_ts_original__main_text__gp_gaussian_conditional_e%d__sine__seed%d' % (epochs, seed)
    for epochs in (400, 600)] for seed in grouped.SEEDS]
assert all(i in records for pair in pairs for i in pair)
folder = root / 'projects/flow_matching_research/results/2026-10-14/cfm_Main_GP400_GP600_all_five_complete_pairs_v1'
assert not folder.exists()
errors = [[], []]
pair_rows = []
ignored = {'id', 'epochs', 'expected_optimizer_updates', 'model_config', 'model_config_sha256', 'output_directory', 'artifact_directory'}
for seed, pair in zip(grouped.SEEDS, pairs):
    jobs = [canonical[i] for i in pair]
    assert {k: v for k, v in jobs[0].items() if k not in ignored} == {k: v for k, v in jobs[1].items() if k not in ignored}
    parameters = [read(root / job['model_config'])['parameters'] for job in jobs]
    assert {k for k in parameters[0].keys() | parameters[1].keys() if parameters[0].get(k) != parameters[1].get(k)} == {'epochs'}
    histories = []
    reference = None
    selected = []
    for position, identity in enumerate(pair):
        record = records[identity]
        cold.validate_closed_primary_record(owners[identity], record)
        assert record['seed'] == seed and record['epochs'] == (400, 600)[position]
        assert record['optimizer_updates'] == (62400, 93600)[position]
        assert record['full_replay_shape'] == [20, 500, 1]
        result = read(root / record['result_path'])
        histories.append(result['training_losses'])
        with np.load(record['replay_arrays'], allow_pickle=False) as arrays:
            assert arrays['predictions'].shape == arrays['truth'].shape == (20, 500, 1)
            assert np.isfinite(arrays['predictions']).all() and len(set(arrays['test_ids'].tolist())) == 20
            fields = {k: arrays[k].copy() for k in ('truth', 'times', 'test_ids')}
            if reference is None:
                reference = fields
            else:
                assert all(np.array_equal(reference[k], fields[k]) for k in fields)
            value = np.square(arrays['predictions'].astype(np.float64) - arrays['truth']).mean(axis=(1, 2))
            assert np.isclose(value.mean(), record['replayed_metrics']['MSE'], rtol=1e-12, atol=1e-12)
            errors[position].append(value)
        selected.append(record)
    pair_rows.append(dict(seed=seed, records=selected, original_whole_trajectory_ids=reference['test_ids'].tolist(),
        first400_loss_prefix=prefix_evidence(*histories)))

# Identical frozen estimator, with exactly its four trajectory-count constants changed80->20.
tree = ast.parse(inspect.getsource(grouped.grouped_summary))
count = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 80:
        node.value = 20
        count += 1
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value.replace('all80 full trajectory errors', 'all20 full trajectory errors')
assert count == 4
namespace = dict(grouped.__dict__)
exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
summary = namespace['grouped_summary'](*errors, grouped.SEEDS, 20261014, 10000)
summary['conditional_GP400'] = summary.pop('paper_literal')
summary['conditional_GP600'] = summary.pop('gaussian_conditional')
assert summary['whole_test_trajectory_groups_per_model'] == 20
assert summary['conditional_GP600']['per_joint_seed_MSE'][3] > 1e54
report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), original_track='main_text', dataset='sine',
    original_data_and_training_seeds=grouped.SEEDS, original_epochs=[400, 600],
    original_optimizer_updates=[62400, 93600], complete_original_model_count=10,
    all_original_full_coordinate_count=100000, full_native_and_independent_proofs_rechecked=True,
    pairs=pair_rows, **summary, no_seed_or_trajectory_removed=True,
    catastrophic_finite_seed1106_included_in_mean_and_uncertainty=True,
    additional_model_inference=0, additional_optimizer_updates=0,
    intermediate_weights_or_optimizer_state_equivalence_certified=False,
    author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False,
    boundary='All five original Main100-total/20-test pairs retained, including saved-weight score restoration of400 seed1106. Separate from Appendix400-total/80-test/250000-375000-update cohort. Same paired data, times, truth and target/model recipe; only registered budget differs. Five outer joint data/training groups and20 inner whole trajectories give descriptive uncertainty. Loss-prefix equality does not prove intermediate-state equality. No successful-subset or outlier-trimmed mean.')
folder.mkdir(exist_ok=False)
(folder / 'all_five_original_Main_budget_pair_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
lines = ['CFM-TS / Main-text Sine / conditional Gaussian GP / all five400-versus600 pairs\n',
    'All ten original complete models passed guarded native replay and independent full20x500-grid review. Paired trajectory IDs, truth and times match exactly; only the registered400/62400 versus600/93600 budget differs. Seed1106 uses its previously verified saved original400 weights without new fitting.\n',
    '| Joint seed | 400 MSE | 600 MSE |', '| ---: | ---: | ---: |']
for seed, a, b in zip(grouped.SEEDS, summary['conditional_GP400']['per_joint_seed_MSE'], summary['conditional_GP600']['per_joint_seed_MSE']):
    lines.append('| %d | %.12g | %.12g |' % (seed, a, b))
lines += ['', 'Full-five400 mean %.12g, sample standard deviation %.12g. Full-five600 mean %.12g, sample standard deviation %.12g.' % (
    summary['conditional_GP400']['mean_MSE'], summary['conditional_GP400']['joint_seed_std'],
    summary['conditional_GP600']['mean_MSE'], summary['conditional_GP600']['joint_seed_std']),
    '', 'Paired600-minus400 mean difference %.12g; hierarchical paired joint-seed/whole-trajectory bootstrap95%% interval %s (10000 resamples, frozen seed20261014).' % (
    summary['paired_difference_mean'], str(summary['hierarchical_paired_bootstrap_ci95'])),
    '', 'The catastrophic but finite600 seed1106 error remains in the mean and bootstrap. A lower bootstrap endpoint near ordinary errors can occur when resampling omits that seed or its dominant trajectory; this is not removal from the observed benchmark. Only five outer groups are available, so the interval is descriptive and does not establish a population-wide training effect or an author-code bug. No outlier-trimmed mean is published.',
    '', 'Main20-test recipes stay separate from Appendix80-test recipes. All original first400 loss histories are compared; intermediate weights and optimizer state equivalence are not certified. Original-author equivalence and strict TAB remain unproved. The broader285 routes and45/681/273research scope remain required.',
    '', 'Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ']
(folder / 'README.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(Main400=summary['conditional_GP400'], Main600=summary['conditional_GP600'],
    paired_difference=summary['paired_difference_mean'], paired_CI95=summary['hierarchical_paired_bootstrap_ci95'])))
