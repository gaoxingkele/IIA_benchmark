"""Archive one new full model against the preserved35-model certificate catalog."""
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as audit
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding

root, read, sha = audit.ROOT, audit.read, audit.sha
runtime = read(root / 'configs/runtime/fm_grasp_completion_replay_queue.v1.json')
originals, executions, _, _ = audit.verify(runtime)
identity = 'grasp_protocol_v3__gnn__swan__swan_sf__seed1106'
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
priorpath = root / 'projects/flow_matching_research/results/2026-10-14/grasp_actual_complete_model_metric_catalog_v2/actual_full_complete_model_and_protocol_metric_catalog.json'
prior = read(priorpath)
priorids = {r['canonical_id'] for r in prior['all_current_complete_model_proof_references']}
assert len(priorids) == 35 and set(state['verified_original_ids']) == priorids | {identity}
folder = root / 'projects/flow_matching_research/results/2026-10-14/grasp_gnn_seed1106_full36_model_increment_v1'
folder.mkdir(exist_ok=False)
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
    epochs=record['epochs_completed'], optimizer_updates=record['optimizer_updates'], parameter_count=record['parameter_count'],
    recipes=record['recipes'], actual_current_unique_complete_models=36,
    prior35_model_catalog=priorpath.relative_to(root).as_posix(), prior35_catalog_sha256=sha(priorpath),
    current_complete_original_ids=state['verified_original_ids'], current_source_proof_hashes={key: sha(path) for key, path in sources.items()},
    required_original_entity_models=9120, required_original_training_cohorts=480, required_original_training_seeds=10,
    complete_gnn_Swan_seeds=[1103, 1104, 1105, 1106], full_ten_seed_gnn_cohort_complete=False,
    additional_model_inference=0, additional_optimizer_updates=0,
    author_equivalence_certified=False, strict_TAB_result=False, original_scope=runtime['original_scope'],
    all_original_experiments_complete=False)
(folder / 'actual_full36_model_increment_closure.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
metrics = record['recipes'][0]['replayed_metrics']
(folder / 'README.md').write_text('GRASP GNN / Swan / swan_sf / seed1106\n\n'
    'The original1500-epoch/93000-update model completed every original recipe,3935 validation and72000 test points, guarded native checkpoint replay and independent full-array/forward-budget/metric review. Original/replay score, checkpoint, result and proof hashes were rechecked. The current36-ID completion ledger equals the preserved35-model catalog plus this model.\n\n'
    'Independent metrics: ' + json.dumps(metrics) + '\n\n'
    'GNN currently has4/10 original Swan seeds. Score agreement follows the unchanged registered tolerances and need not be bitwise. Test-label oracle metrics and validation-only controls remain separate; no point-adjusted score, full ten-seed performance, author equivalence or strict TAB is claimed. Keep all9120entity models/480cohorts/10seeds and45/681/273research scope.\n')
print(json.dumps(dict(complete_models=36, new_model=identity, metrics=metrics)))
