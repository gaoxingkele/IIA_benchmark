"""Archive one new full model against the preserved45-model closure."""
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as audit
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding

root, read, sha = audit.ROOT, audit.read, audit.sha
runtime = read(root / 'configs/runtime/fm_grasp_completion_replay_queue.v1.json')
originals, executions, _, _ = audit.verify(runtime)
identity = 'grasp_protocol_v3__tau_4__swan__swan_sf__seed1106'
reference = audit.completed_pipeline(runtime, identity)
pipelinepath = root / reference['completed_pipeline_configuration']
pipeline = read(pipelinepath)
producer = binding(pipeline)
config, native, receipt = audit.current_native(pipeline['producer_configuration'])
record = native['records'][0]
assert record['canonical_id'] == identity and len(native['records']) == 1
resultpath = root / record['result_path']
result = read(resultpath)
audit.check_complete_result(originals[identity], executions[identity], result)
assert sha(resultpath) == record['result_sha256'] and sha(record['checkpoint_path']) == record['checkpoint_sha256']
for artifact in result['artifacts']:
    assert sha(root / artifact['path']) == artifact['sha256']
for recipe in record['recipes']:
    for tag in ('original', 'replay'):
        assert sha(recipe[tag + '_score_path']) == recipe[tag + '_score_sha256']
    assert recipe['complete_validation_points'] == 3935 and recipe['complete_test_points'] == 72000
statepath = root / runtime['state_root'] / 'status.json'
state = read(statepath)
priorpath = root / 'projects/flow_matching_research/results/2026-10-16/grasp_tau1_seed1106_full45_model_increment_v1/actual_full45_model_increment_closure.json'
prior = read(priorpath)
priorids = set(prior['current_complete_original_ids'])
assert len(priorids) == 45 and set(state['verified_original_ids']) == priorids | {identity}
folder = root / 'projects/flow_matching_research/results/2026-10-16/grasp_tau4_seed1106_full46_model_increment_v1'
independent = read(root / pipeline['review_output_directory'] / 'independent_full_replay_verification.json')
assert independent['passed'] and independent['actual_completed_recipe_count'] == len(record['recipes']) == 1
assert independent['all_original_forward_counts_and_recipe_budgets_verified']
assert independent['all_metrics_independently_recomputed_from_full_replay']
assert record['epochs_completed'] == 1500 and record['optimizer_updates'] == 93000
assert record['parameter_count'] == result['parameter_count'] == 69244
assert reference['completed_full_replay_proof']
for source, digest in prior['current_source_proof_hashes'].items():
    prior_source_path = priorpath.parent / (source + '.json')
    assert sha(prior_source_path) == digest
cohort_seeds = sorted(int(i.rsplit('seed', 1)[1]) for i in state['verified_original_ids'] if i.startswith('grasp_protocol_v3__tau_4__swan__swan_sf__seed'))
assert cohort_seeds == [1103, 1104, 1105, 1106]
required_cohort_seeds = sorted(int(i.rsplit('seed', 1)[1]) for i in originals if i.startswith('grasp_protocol_v3__tau_4__swan__swan_sf__seed'))
assert required_cohort_seeds == list(range(1103,1113))
folder.mkdir(parents=True, exist_ok=False)
sources = dict(original_result=resultpath, native_configuration=root / pipeline['producer_configuration'],
    native_report=root / producer['output_directory'] / 'checkpoint_replay_audit.json',
    native_guarded_receipt=root / producer['state_root'] / 'receipt.json',
    completed_pipeline_configuration=pipelinepath, completed_pipeline_proof=root / reference['completed_full_replay_proof'],
    independent_configuration=root / pipeline['review_configuration'],
    independent_full_recipe_proof=root / pipeline['review_output_directory'] / 'independent_full_replay_verification.json',
    independent_guarded_receipt=root / pipeline['state_root'] / 'receipt.json',
    original_training_resource_usage=root / result['job']['output_directory'] / 'resource_usage.json',
    frozen_current_full9120_completion_state=statepath)
for name, path in sources.items():
    (folder / (name + '.json')).write_bytes(path.read_bytes())
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_id=identity,
    complete_original_training_and_full_native_replay_and_independent_recipe_review=True,
    inherited_native_boundary_metadata_discrepancy='Frozen native config prose names tau1 seed1104; actual canonical/profile/seed/checkpoint/source and independent bindings prove Tau4 variant seed1106 and its one original5x10 recipe. Frozen artifacts preserved; no scientific settings changed.',
    epochs=record['epochs_completed'], optimizer_updates=record['optimizer_updates'], parameter_count=record['parameter_count'],
    recipes=record['recipes'], actual_current_unique_complete_models=46,
    prior45_model_closure=priorpath.relative_to(root).as_posix(), prior45_closure_sha256=sha(priorpath),
    current_complete_original_ids=state['verified_original_ids'], current_source_proof_hashes={key: sha(path) for key, path in sources.items()},
    required_original_entity_models=9120, required_original_training_cohorts=480, required_original_training_seeds=10,
    complete_Tau4_Swan_seeds=cohort_seeds, required_Tau4_Swan_seeds=required_cohort_seeds,
    pending_Tau4_Swan_seeds=sorted(set(required_cohort_seeds)-set(cohort_seeds)), full_ten_seed_Tau4_cohort_complete=False,
    additional_model_inference=0, additional_optimizer_updates=0,
    author_equivalence_certified=False, strict_TAB_result=False, original_scope=runtime['original_scope'],
    all_original_experiments_complete=False)
(folder / 'actual_full46_model_increment_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
metrics = record['recipes'][0]['replayed_metrics']
(folder / 'README.md').write_text('GRASP Tau4 variant / Swan / swan_sf / seed1106\n\n'
    'The complete original1500-epoch/93000-update/69244-parameter Tau4 variant completed its one original5source-sample x10flow-evaluation recipe,3935validation/72000test points, guarded native checkpoint replay and independent full-array/forward-budget/metric review. Current46original completed IDs equal the immutable45-model closure plus this model. Source/checkpoint/all original and replay array/proof hashes were rechecked.\n\n'
    'Independent metrics: ' + json.dumps(metrics) + '\n\n'
    'Frozen native-template prose names tau1 seed1104; actual structured canonical/profile/seed and every source/weight/array/proof binding identify Tau4 variant seed1106. One recipe is the complete original baseline budget, not a shortened main-model10-recipe grid. The frozen template/proof was preserved.\n\n'
    'Tau4 variant currently has4/10original Swan seeds:1103,1104,1105,1106. All10 original seeds1103..1112 remain registered;1107..1112 remain required. Original rtol=0.0002/atol=0.001 score tolerance and all guards remain; no bitwise claim is made. Test-label oracle metrics and validation-only controls remain separate. No full ten-seed aggregate, PA or journal/strictTAB certificate. Keep all9120entity models/480cohorts/10seeds and45/681/273scope.\n')
print(json.dumps(dict(complete_models=46, new_model=identity, metrics=metrics)))
