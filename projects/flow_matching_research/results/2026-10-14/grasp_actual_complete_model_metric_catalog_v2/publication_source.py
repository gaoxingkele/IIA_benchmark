"""Publish a reviewed 35-model interim catalog without changing numerical artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as audit
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding

root = audit.ROOT
folder = root / 'projects/flow_matching_research/results/2026-10-14/grasp_actual_complete_model_metric_catalog_v2'
report = audit.read(folder / 'actual_full_complete_model_and_protocol_metric_catalog.json')
assert report['unique_complete_original_models'] == 35
assert len(report['all_original_score_recipe_rows']) == 62
assert len(report['interim_entity_cohorts']) == 21
identity = 'grasp_protocol_v3__tau_8__swan__swan_sf__seed1105'
runtime = audit.read(root / 'configs/runtime/fm_grasp_completion_replay_queue.v1.json')
reference = audit.completed_pipeline(runtime, identity)
pipelinepath = root / reference['completed_pipeline_configuration']
pipeline = audit.read(pipelinepath)
producer = binding(pipeline)
row = next(r for r in report['all_original_score_recipe_rows'] if r['canonical_id'] == identity)
assert row['epochs'] == 1500 and row['optimizer_updates'] == 93000
assert row['complete_validation_points'] == 3935 and row['complete_test_points'] == 72000
assert row['recipe'] == dict(source_samples=5, flow_evaluations=10)
case = folder / identity
source_result = audit.read(case / 'original_result.json')
usagepath = root / source_result['job']['output_directory'] / 'resource_usage.json'
usage = audit.read(usagepath)
assert not usage['resource_guard_aborted']
sources = dict(completed_pipeline_configuration=pipelinepath,
    completed_pipeline_proof=root / reference['completed_full_replay_proof'],
    independent_configuration=root / pipeline['review_configuration'],
    independent_guarded_receipt=root / pipeline['state_root'] / 'receipt.json',
    original_training_resource_usage=usagepath)
for name, path in sources.items():
    destination = case / (name + '.json')
    assert not destination.exists()
    destination.write_bytes(path.read_bytes())
native = audit.read(case / 'native_report.json')['records'][0]
recipe = native['recipes'][0]
assert recipe['forward_evaluation_counts']['validation'] == dict(count=800, windows=194300)
assert recipe['forward_evaluation_counts']['test'] == dict(count=14100, windows=3597550)
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_id=identity,
    full_original_training_and_native_reinference_and_independent_review_passed=True,
    original_epochs=1500, optimizer_updates=93000, parameter_count=native['parameter_count'],
    full_validation_points=3935, full_test_points=72000, original_recipe=row['recipe'],
    every_original_forward_budget_verified=recipe['forward_evaluation_counts'],
    score_agreement=row['score_agreement'], independently_recomputed_metrics=row['replayed_metrics'],
    source_proof_hashes={name: audit.sha(path) for name, path in sources.items()},
    complete_current_unique_models=35, complete_current_score_recipe_rows=62,
    required_original_entity_models=9120, required_original_cohorts=480, required_original_seeds=10,
    full_ten_seed_cohort_complete=False, additional_optimizer_updates=0, additional_model_inference=0,
    bitwise_score_equivalence_claimed=False, author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=runtime['original_scope'], all_original_experiments_complete=False)
(case / 'actual_complete_model_and_one_recipe_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
lines = ['GRASP: 35 unique complete original models, 62 score recipe rows, 21 recipe groups\n',
    'Every listed full1500-epoch model, original update budget and complete validation/test grid has guarded native replay and independent review; current result/checkpoint/score/proof hashes were rechecked. Retain all9120entity models,480training cohorts,10registered seeds and45/681/273scope. Scoring recipes can share a trained model; recipe rows and recipe groups are not independent training seeds or training cohorts.\n',
    '| Profile | Entity | Sources / flow steps | Complete seeds / 10 | Oracle Best F1 mean | Validation-only F1 mean |',
    '| --- | --- | --- | ---: | ---: | ---: |']
for group in sorted(report['interim_entity_cohorts'], key=lambda c: (c['profile'], json.dumps(c['recipe'], sort_keys=True))):
    values, settings = group['descriptive_interim_metrics'], group['recipe']
    lines.append('| %s | %s | %d / %d | %d / 10 | %.6f | %.6f |' % (group['profile'], group['entity'],
        settings['source_samples'], settings['flow_evaluations'], group['complete_original_model_count'],
        values['paper_retrospective/Best_F1']['mean'], values['validation_only_control/f1']['mean']))
lines += ['', 'Newly closed: tau8 / Swan / swan_sf / seed1105,1500 epochs/93000 updates,3935 validation and72000 test points. Original5-source/10-flow-step forward budgets: validation800 calls/194300 window evaluations; test14100 calls/3597550 evaluations. Independent oracle AP0.5064573954984611, ROC0.6716647336535052, BestF1 0.4933623245782048; validation-only precision0.3781620236951649, recall0.7050746268656717, F1 0.49228845352230094. No point adjustment; validation controls use no test features or labels for calibration.',
    '', 'Native CPU/GPU score comparisons pass the unchanged rtol0.0002/atol0.001 and metric tolerances. They are not bitwise: maximum validation absolute difference0.00390625, test absolute difference0.27062499994644895; test maximum relative difference8.942361475836056e-7. No tolerances were relaxed.',
    '', 'Current means and sample standard deviations describe only the complete registered seeds shown. Most profiles have3/10 seeds; tau0.25 has2/10. No full ten-seed performance or uncertainty claim. Oracle test-label threshold metrics and validation-only controls remain separate; neither certifies author equivalence or strict TAB.',
    '', 'Legacy union entries retain actual complete per-model/per-recipe records. A shared native container report that omits a model is not treated as its direct replay proof. The catalog resolves the preserved per-model record and hashes its independently verified arrays explicitly. The previous34-model v1 catalog is preserved.']
(folder / 'README.md').write_text('\n'.join(lines) + '\n')
(folder / 'publication_source.py').write_bytes(Path(__file__).read_bytes())
handles = []
for pid, expected in [(39032, 1791948044.389774), (41992, 1791948094.8603492), (25120, 1791941882.8006165)]:
    p = psutil.Process(pid)
    assert p.create_time() == expected
    handles.append(dict(pid=pid, create_time=p.create_time(), command=p.cmdline(), cpu_seconds=sum(p.cpu_times()[:2])))
(folder / 'actual_other_full_scope_live_workers.json').write_text(json.dumps(dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    actual_live_workers=handles, all_original_experiments_complete=False), indent=2) + '\n')
print(json.dumps(dict(complete_models=35, full_recipe_rows=62, new_model_metrics=row['replayed_metrics'])))
