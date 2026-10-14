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
old = read(root / 'projects/flow_matching_research/results/2026-10-14/cfm_actual148_and_main_GP400_all_five_closure_v1/actual148_full_unique_canonical_model_closure.json')
prior = {r['original_canonical_id']: r for r in old['deduplicated_full_records']}
newids = ['cfm_ts_original__appendix_d__gp_gaussian_conditional_e600__sine__seed' + str(s) for s in (1105, 1106)]
assert len(prior) == 148 and len(records) == 150
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
folder = root / 'projects/flow_matching_research/results/2026-10-14/cfm_actual150_conditional_GP600_two_closed_models_v1'
folder.mkdir(exist_ok=False)
for number, raw in enumerate(snapshots):
    (folder / ('frozen_full_model_snapshot_%d.json' % number)).write_bytes(raw)
closed = []
for identity in newids:
    record, runtime = records[identity], owners[identity]
    assert record['epochs'] == 600 and record['optimizer_updates'] == 375000
    assert record['full_replay_shape'] == [80, 500, 1]
    assert record['bitwise_identical'] and record['max_absolute_difference'] == 0
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
closure = dict(captured_utc=datetime.now(timezone.utc).isoformat(), full_unique_canonical_models=150,
    required_original_models=285, new_full_models=closed, all150_current_native_independent_proof_chains_and_artifacts_rehashed=True,
    previous148_records_identical=True, deduplicated_full_records=list(records.values()),
    preserved_original85_union_configuration_sha256=sha(legacyconfigpath),
    resolved_preserved_original85_native_proof_bindings=legacy_bindings,
    additional_optimizer_updates=0, additional_model_inference=0,
    Appendix_conditional_GP600_complete_seeds=[1103, 1105, 1106],
    Appendix_conditional_GP600_failed_seed=1104, Appendix_conditional_GP600_pending_seed=1107,
    Appendix_conditional_GP600_all_five_finite_seed_mean=None,
    author_equivalence_certified=False, strict_TAB_result=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    all_original_experiments_complete=False)
(folder / 'actual150_deduplicated_complete_original_models.json').write_text(json.dumps(closure, indent=2) + '\n')
(folder / 'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n')
(folder / 'README.md').write_text('''150/285 deduplicated complete original CFM models

Two new Appendix D conditional Gaussian GP models on Sine, seeds1105 and1106, completed600 epochs/375000 updates each, all80x500 original test grids, bitwise native replay (maximum score difference0), guarded terminal checks and independent full-array review. Every current native/independent proof chain and artifact digest for all150 union records was rechecked. Overlapping producer snapshots are deduplicated by canonical identity; their counts are never added. The previous148 records remain identical.

| Seed | MSE | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 1105 | 0.005775390992284606 | 0.052522456818637236 | 0.07599599326467552 |
| 1106 | 0.0002904695369040557 | 0.011526650028284383 | 0.017043166868397894 |

These are individual joint data/training seeds. Seed1103 was previously closed; seed1104 has an independently checked original full80 numerical failure; seed1107 remains pending. There is no complete five-seed finite performance mean or uncertainty estimate. Keep Main20-test and Appendix80-test recipes separate. No seed exclusion, solver changes or successful-subset benchmark. Local checkpoint/score repeatability does not certify author equivalence or strict TAB. Preserve all285 routes and the full45/681/273 research scope.

Paper: https://openreview.net/forum?id=Hqn4Aj7xrQ
''')
print(json.dumps(dict(unique_models=150, new_full_models=[dict(id=r['record']['original_canonical_id'], metrics=r['record']['replayed_metrics']) for r in closed])))
