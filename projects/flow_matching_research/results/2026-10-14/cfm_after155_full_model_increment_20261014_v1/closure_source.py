"""Verify deduplicated current full certificates and archive two new original models."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import FunctionType
import numpy as np
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays

root, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
runtimepaths = ['configs/runtime/fm_cfm_completion_replay_queue.v4.json',
    'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json', 'configs/runtime/fm_cfm_saved_score_full_replay.v1.json']
records, owners, snapshots = {}, {}, []
for path in runtimepaths:
    runtime = read(root / path)
    assert len(runtime['routes']) == 285
    snapshot = root / runtime['state_root'] / 'verified_full_model_records.json'
    raw = snapshot.read_bytes()
    for record in json.loads(raw)['records']:
        identity = record['original_canonical_id']
        if identity in records:
            assert records[identity] == record, identity
        else:
            records[identity] = record
            owners[identity] = runtime
    snapshots.append(raw)
old = read(root / 'projects/flow_matching_research/results/2026-10-14/cfm_after152_full_model_increment_20261014_v1/actual_current_deduplicated_complete_original_models.json')
prior = {r['original_canonical_id']: r for r in old['deduplicated_full_records']}
newids = sorted(set(records) - set(prior))
assert len(prior) == 155 and len(records) >= 156
assert set(records) - set(prior) == set(newids)
assert all(records[key] == value for key, value in prior.items())
legacy = read(root / owners[newids[0]]['preserved_verified_union'])
legacyconfigpath = root / legacy['configuration']
legacyconfig = read(legacyconfigpath)
assert sha(legacyconfigpath) == legacy['configuration_sha256']
for pin in legacyconfig['source_receipts']:
    assert sha(root / pin['path']) == pin['sha256']
legacy_native = {}
legacy_bindings = []
for reportpath in legacy['source_full_model_proofs']:
    report = read(root / reportpath)
    configpath = root / report['configuration']
    config = read(configpath)
    assert sha(configpath) == report['configuration_sha256']
    guardpath = root / config['state_root'] / 'receipt.json'
    cold.original.check_native_terminal(config, report, read(guardpath))
    for item in config['full_results']:
        identity = item['original_canonical_id']
        assert identity not in legacy_native
        legacy_native[identity] = (config, item)
    legacy_bindings.append(dict(native_configuration=str(configpath), configuration_sha256=sha(configpath),
        native_report=reportpath, report_sha256=sha(root / reportpath),
        native_guard_receipt=str(guardpath), guard_receipt_sha256=sha(guardpath)))
assert len(legacy_native) == legacy['full_canonical_models_verified'] == 85
verify_arrays = FunctionType(cold.original.verify_record.__code__, dict(cold.original.__dict__, sha=sha))
for identity, record in records.items():
    if identity in legacy_native:
        config, canonical = legacy_native[identity]
        checked = verify_arrays(config, canonical, record)
        assert all(checked[key] == value for key, value in record.items())
        assert checked['all_original_grid_coordinates_independently_checked']
    else:
        cold.validate_closed_primary_record(owners[identity], record)
folder = root / 'projects/flow_matching_research/results/2026-10-14/cfm_after155_full_model_increment_20261014_v1'
folder.mkdir(exist_ok=False)
for number, raw in enumerate(snapshots):
    (folder / ('frozen_full_model_snapshot_%d.json' % number)).write_bytes(raw)
closed = []
for identity in newids:
    record, runtime = records[identity], owners[identity]
    assert record['checkpoint_contents_and_full_budget_verified'] and record['original_full_grid_replayed']
    nativepath = root / runtime['child_config_root'] / (identity + '.native.json')
    reviewpath = nativepath.with_name(identity + '.review.json')
    native, review = read(nativepath), read(reviewpath)
    sources = dict(original_result=root / record['result_path'], native_configuration=nativepath,
        independent_configuration=reviewpath, native_report=root / native['output_directory'] / 'checkpoint_replay_audit.json',
        independent_proof=root / review['output_proof'], native_guarded_receipt=root / native['state_root'] / 'receipt.json',
        independent_guarded_receipt=root / review['state_root'] / 'receipt.json',
        original_source_guarded_receipt=root / 'experiments/runs/fm_cfm_ts_original_v1/receipts' / (identity + '.json'))
    receipt = read(sources['original_source_guarded_receipt'])
    assert receipt['exit_code'] == 0 and receipt['complete'] and not receipt['resource_observations']['resource_guard_aborted']
    result = read(sources['original_result'])
    predictionpath = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
    with np.load(predictionpath, allow_pickle=False) as arrays, np.load(root / result['job']['data_path'], allow_pickle=False) as data:
        metrics = audit_arrays(result['job'], result, arrays, data)
        assert not set(data['train_ids']) & set(data['test_ids'])
    case = folder / identity
    case.mkdir()
    for name, path in sources.items():
        (case / (name + '.json')).write_bytes(path.read_bytes())
    (case / 'independently_recomputed_original_source_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    closed.append(dict(record=record, current_source_proof_hashes={key: sha(path) for key, path in sources.items()}))
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), full_unique_canonical_models=len(records),
    required_original_models=285, new_full_models=closed, all_current_native_independent_proof_chains_and_artifacts_rehashed=True,
    previous155_records_identical=True, deduplicated_full_records=list(records.values()),
    preserved_original85_union_configuration_sha256=sha(legacyconfigpath),
    resolved_preserved_original85_native_proof_bindings=legacy_bindings,
    additional_optimizer_updates=0, additional_model_inference=0,
    Appendix_conditional_GP600_complete_seeds=[1103, 1105, 1106, 1107],
    Appendix_conditional_GP600_failed_seed=1104, Appendix_conditional_GP600_pending_seed=None,
    Appendix_conditional_GP600_all_five_finite_seed_mean=None,
    author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False)
(folder / 'actual_current_deduplicated_complete_original_models.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
lines = [str(len(records)) + '/285 deduplicated complete original CFM models\n',
    'Current native/independent proof chains and artifact hashes were rechecked. Overlapping snapshots are deduplicated by canonical identity, never summed. The previous155 records remain identical. All newly closed original source grids and metrics were independently recomputed.\n',
    '| New canonical model | Epochs | Updates | Original grid | MSE | MAE | RMSE | Bitwise native scores |',
    '| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |']
for item in closed:
    r=item['record'];m=r['replayed_metrics']
    lines.append('| %s | %d | %d | %s | %.12g | %.12g | %.12g | %s |' %
        (r['original_canonical_id'],r['epochs'],r['optimizer_updates'],str(r['full_replay_shape']),m['MSE'],m['MAE'],m['RMSE'],str(r['bitwise_identical'])))
lines += ['', 'Appendix conditional GP600/Sine now has four complete finite seeds1103,1105,1106,1107 and the independently observed full80-grid failed seed1104. Full five-seed600 MSE remains undefined. The separate all-five original400/600 attempt-outcome audit retains that failure. Main20-test/93600-update GP600 recipes remain distinct from Appendix80-test/375000-update recipes; no combined performance mean, seed exclusion, altered solver or successful-subset benchmark. Original-author equivalence and strict TAB remain unproved; all285 slots and45/681/273scope remain required.', '', 'Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ']
(folder / 'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(unique_models=len(records), new_full_models=[dict(id=r['record']['original_canonical_id'], metrics=r['record']['replayed_metrics']) for r in closed])))
