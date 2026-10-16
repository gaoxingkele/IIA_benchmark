"""Replay the diagnosed industrial SB checkpoint without fitting a second model.

Full source completion is a prerequisite. A separate output, unchanged CPU
resource admission and terminal failure preservation apply to this audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1'
CONFIG = 'configs/reproducibility/fm_industrial_SB_extended_full_replay.v1.json'
SOURCE = 'experiments/runs/fm_industrial_SB_transport_extension_v3'
SUFFIXES = ('_labels', '_timestamps', '_source_rows', '_fault_labels')


def registration():
    job_path = SOURCE + '/full_original_case_job.json'
    source = read(ROOT / job_path)
    runtime = read(ROOT / source['source_runtime'])
    files = [job_path, source['source_runtime'],
        MODULE.replace('.', '/') + '.py',
        'tests/test_fm_industrial_SB_extended_full_replay_v1.py',
        'scripts/flow_matching/audit_crossad_full139_independent_metrics_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py']
    return dict(source_registration=job_path, source_root=SOURCE,
        canonical_case=source['canonical_original_case'],
        state_root='experiments/runs/fm_industrial_SB_extended_full_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_SB_extended_full_replay_v1',
        python=runtime['python'], settings=runtime['settings'], resource_lock=runtime['resource_lock'],
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in files],
        score_comparison='exact_dtype_shape_and_full_payload_bytes',
        numerical_metric_tolerance=dict(rtol=2e-6, atol=2e-6),
        all_original_experiments_complete=False, original_global_scope=runtime['original_global_scope'],
        new_optimizer_updates=0, new_training_seed_slots=0,
        full_TAB_equivalence=False, paper_equivalence=False,
        citation=runtime['literature_citation'], data_citation=runtime['data_citation'])


def verify(c):
    if c != registration():
        raise ValueError('Frozen source registration, exact replay or resource requirements changed')
    original = read(ROOT / c['source_registration'])
    for pin in original['job']['frozen_files']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen full industrial source changed: ' + pin['path'])
    return original


def require_source_completion(original):
    base = ROOT / SOURCE
    paths = dict(result=base/'full_job/result.json', checkpoint=base/'full_job/model.pt',
        scores=base/'full_job/scores.npz', solver=base/'full_job/complete_transport_solver_audit.json',
        guard=base/'attempts/full_training/receipt.json')
    if not all(p.is_file() for p in paths.values()):
        raise FileNotFoundError('Wait for full source training, complete scores, solver audit and successful guard')
    result, solver, guard = (read(paths[k]) for k in ('result', 'solver', 'guard'))
    job = original['job']
    if (result['id'] != job['id'] or result['job'] != job or result['status'] != 'completed'
            or result['checkpoint_sha256'] != sha(paths['checkpoint'])
            or result['scores_sha256'] != sha(paths['scores'])
            or solver['source_result_sha256'] != sha(paths['result'])
            or not solver['passed'] or not guard['passed'] or guard['exit_code'] != 0
            or guard['resource_guard_aborted'] or guard['proof_sha256'] != sha(paths['solver'])):
        raise ValueError('Require complete source identity, bound artifacts and successful original resource guard')
    losses = result['training_losses']
    audits = solver['all20_actual_training_couplings']
    if (len(losses) != 20 or not all(math.isfinite(x) for x in losses)
            or result['epoch_audit']['declared_epochs'] != 20 or result['epoch_audit']['recorded_epochs'] != 20
            or result['unique_fit_windows'] != 8 or result['test_points'] != 11700 or result['validation_points'] != 4320
            or solver['optimizer_updates'] != 20 or len(audits) != 20
            or any(x['regularization'] != 2. or x['tolerance'] != 1e-6
                   or not math.isfinite(x['actual_absolute_marginal_error'])
                   or x['actual_absolute_marginal_error'] > 1e-6
                   or x['uniform_or_exact_OT_fallback'] for x in audits)):
        raise ValueError('All twenty original epochs/updates/couplings and full native points required')
    return paths, result


def roles(dataset):
    rows = [(role, g) for role, key in [('train', 'training_groups'), ('validation', 'validation_groups'), ('test', 'entities')]
        for g in dataset[key]]
    prefixes = [g['array_prefix'] for _, g in rows]
    if len(set(prefixes)) != len(prefixes) or any(not dataset[k] for k in ['training_groups', 'validation_groups', 'entities']):
        raise ValueError('Every distinct full industrial segment and all three roles required')
    role_by_group = {}
    for role, g in rows:
        if g['group_id'] in role_by_group and role_by_group[g['group_id']] != role:
            raise ValueError('Whole source group crosses training/calibration/test roles')
        role_by_group[g['group_id']] = role
    return rows


def native_windows(values, window):
    import numpy as np
    if values.ndim != 2 or len(values) < window or window <= 0 or not np.isfinite(values).all():
        raise ValueError('Full finite two-dimensional segment at original window required')
    starts = list(range(0, len(values) - window + 1, window))
    if starts[-1] != len(values) - window:
        starts.append(len(values) - window)
    return np.stack([values[i:i+window] for i in starts]), starts


def native_align(scores, starts, length, window):
    import numpy as np
    if scores.shape != (len(starts), window) or not np.isfinite(scores).all():
        raise ValueError('Every original window timestamp score required')
    total, counts = np.zeros(length), np.zeros(length, dtype=np.int64)
    for score, start in zip(scores, starts):
        total[start:start+window] += score
        counts[start:start+window] += 1
    if not counts.all():
        raise ValueError('Missing native timestamp or tail')
    return total/counts


def require_array_equal(actual, expected, name):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes(order='C') != expected.tobytes(order='C')):
        raise ValueError('Exact full array payload differs: ' + name)
    return dict(dtype=str(actual.dtype), shape=list(actual.shape),
        payload_sha256=hashlib.sha256(actual.tobytes(order='C')).hexdigest())


def strict_metrics(dataset, payload, source_metrics, tolerance, checkpoint_sha256):
    """Pure Python ranks/counts/thresholds; never call source metric functions."""
    from scripts.flow_matching.audit_crossad_full139_independent_metrics_v1 import percentile, ranking, at_threshold, equal_fields
    validation = [float(x) for g in dataset['validation_groups'] for x in payload[g['array_prefix']+'_validation']]
    validation_sha = hashlib.sha256(__import__('struct').pack('<'+'d'*len(validation), *validation)).hexdigest()
    validation.sort()
    entities = [(g['entity_id'], list(map(float, payload[g['array_prefix']+'_test'])),
        list(payload[g['array_prefix']+'_labels'])) for g in dataset['entities']]
    scores = [x for _, values, _ in entities for x in values]
    labels = [x for _, _, values in entities for x in values]
    if not all(math.isfinite(x) for x in validation+scores) or any(x not in (0, 1) for x in labels):
        raise ValueError('Every full validation/test score finite and original labels binary required')
    entities = [(name, s, list(map(int,y))) for name,s,y in entities]
    labels = list(map(int,labels))
    pooled_ranks = ranking(scores, labels)
    entity_ranks = {name: ranking(s, y) for name, s, y in entities}
    if set(source_metrics) != {'0.1', '1', '5'}:
        raise ValueError('All three frozen validation alarm budgets required')
    records = {}
    for ratio, source in source_metrics.items():
        calibration = source['calibration']
        if (source['point_adjustment'] or calibration['calibration_split'] != 'validation'
                or calibration['calibration_sha256'] != validation_sha or calibration['calibration_points'] != len(validation)
                or calibration['calibration_dataset_sha256'] != dataset['input_sha256']
                or calibration['model_artifact_sha256'] != checkpoint_sha256
                or calibration['false_alarm_percent'] != float(ratio)
                or calibration['rule'] != 'validation_score_percentile_linear_strict_greater'
                or set(source['per_entity']) != {e[0] for e in entities}):
            raise ValueError('Full validation-only calibration and original entity grouping required')
        threshold = percentile(validation, float(ratio))
        equal_fields(dict(threshold=threshold), calibration, tolerance)
        pooled = at_threshold(scores, labels, threshold)['micro']; pooled.update(pooled_ranks)
        per_entity, events = {}, []
        for name, s, y in entities:
            m = at_threshold(s, y, threshold)
            m['micro'].update(entity_ranks[name]); per_entity[name] = m['micro']; events.append(m['event_metrics'])
        event_count = sum(x['events'] for x in events)
        detected = sum(x['detected_events'] for x in events)
        normal = len(labels)-sum(labels)
        event_metrics = dict(events=event_count, detected_events=detected,
            event_recall=detected/event_count if event_count else None,
            mean_detection_delay_points=math.fsum(x['mean_detection_delay_points']*x['detected_events'] for x in events if x['detected_events'])/detected if detected else None,
            false_alarms_per_1000_normal_points=1000*pooled['false_positives']/normal if normal else None)
        computed = dict(micro=pooled, per_entity=per_entity,
            macro_f1=math.fsum(x['f1'] for x in per_entity.values())/len(per_entity), event_metrics=event_metrics)
        equal_fields(computed, source, tolerance)
        records[ratio] = dict(threshold=threshold, **computed)
    return records


def restore(checkpoint, configuration, overrides, shape):
    """Construct only the frozen MLP; never invoke fit or a transport solver."""
    import torch
    from torch import nn
    from scripts.flow_matching.model_registry import build_configured_method
    if (checkpoint['configuration'] != configuration or checkpoint['overrides'] != overrides
            or set(checkpoint['modules']) != {'_model'} or checkpoint['window'] != shape[0]):
        raise ValueError('Exact SB model configuration, device and complete checkpoint modules required')
    model = build_configured_method(configuration, overrides)
    if model.objective != 'sb' or model.device != 'cpu':
        raise ValueError('Only this frozen SB CPU case is registered')
    model._shape = shape; model._device = torch.device('cpu'); model._score_head = False
    dimensions = math.prod(shape)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model.seed)
        model._model = nn.Sequential(nn.Linear(dimensions+1, model.hidden_size), nn.SiLU(),
            nn.Linear(model.hidden_size, model.hidden_size), nn.SiLU(), nn.Linear(model.hidden_size, dimensions))
    model._model.load_state_dict(checkpoint['modules']['_model'], strict=True)
    for key, value in model._model.state_dict().items():
        saved = checkpoint['modules']['_model'][key]
        if value.dtype != saved.dtype or value.shape != saved.shape or not torch.equal(value, saved) or not torch.isfinite(value).all():
            raise ValueError('Exact finite checkpoint weights required')
    model._model.eval()
    return model


def worker(c):
    original = verify(c); paths, result = require_source_completion(original)
    source_hashes = {k:sha(p) for k,p in paths.items()}
    import numpy as np
    import torch
    job = original['job']; torch.set_num_threads(original['evaluation']['cpu_threads'])
    manifest = read(ROOT/job['dataset_manifest_path'])
    dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    if sha(ROOT/dataset['input_npz']) != dataset['input_sha256']:
        raise ValueError('Frozen full source input changed')
    for source in dataset['source_files']:
        if sha(ROOT/source['path']) != source['sha256']:
            raise ValueError('Original raw industrial data changed')
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    configuration = read(ROOT/job['model_config'])
    groups = roles(dataset)
    model = restore(checkpoint, configuration, job['parameter_overrides'], (job['window'], groups[0][1]['features']))
    if sum(p.numel() for p in model._model.parameters()) != result['parameter_count']:
        raise ValueError('Complete original parameter count differs')
    payload, bindings, coverage, fit_budget = {}, {}, [], []
    with np.load(ROOT/dataset['input_npz'], allow_pickle=False) as inputs, np.load(paths['scores'], allow_pickle=False) as saved:
        for role, g in groups:
            prefix = g['array_prefix']; key = prefix+'_'+role
            values = inputs[prefix+('_test' if role == 'test' else '_values')]
            if values.shape != (g['points'], g['features']):
                raise ValueError('Full original native feature coordinates required')
            windows, starts = native_windows(values, job['window'])
            if role == 'train':
                fit_budget.append(dict(group_id=g['group_id'], source_interval=g['source_interval'], points=len(values), windows=len(windows)))
            payload[key] = native_align(model.score(windows), starts, len(values), job['window'])
            bindings[key] = require_array_equal(payload[key], saved[key], key)
            for suffix in SUFFIXES:
                name = prefix+suffix
                if name in inputs:
                    payload[name] = inputs[name].copy(); bindings[name] = require_array_equal(payload[name], saved[name], name)
            coverage.append(dict(role=role, group=g['group_id'], points=len(values), windows=len(windows),
                final_start=starts[-1], tail_points_included=True))
        if set(saved.files) != set(payload):
            raise ValueError('Exact complete saved array set required')
    if (fit_budget != checkpoint['unique_training_groups'] or fit_budget != result['fit_group_budget']
            or sum(g['points'] for g in fit_budget) != 722 or sum(g['windows'] for g in fit_budget) != 8):
        raise ValueError('Original unique fit groups and all eight fit windows required without refitting')
    metrics = strict_metrics(dataset, payload, result['metrics']['strict'], c['numerical_metric_tolerance'], result['checkpoint_sha256'])
    if sum(x['points'] for x in coverage if x['role']=='test') != 11700 or sum(x['points'] for x in coverage if x['role']=='validation') != 4320:
        raise ValueError('Complete original native held-out points required')
    verify(c)
    if {k:sha(p) for k,p in paths.items()} != source_hashes:
        raise ValueError('Immutable full source artifacts changed during replay')
    target = Path(c['artifact_root']); target.mkdir(exist_ok=False, parents=True)
    np.savez_compressed(target/'replayed_scores.npz', **payload)
    proof = dict(passed=True, canonical_case=c['canonical_case'], source_id=job['id'],
        bindings={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        native_coverage=coverage, all_array_payloads=bindings, exact_full_payload_match=True,
        independent_strict_metrics=metrics, replayed_scores_sha256=sha(target/'replayed_scores.npz'),
        new_optimizer_updates=0, new_training_seed_slots=0, full_checkpoint_inference_executed=True,
        bootstrap_recomputed=False, VUS_recomputed=False, affiliation_recomputed=False,
        full_TAB_equivalence=False, paper_equivalence=False, all_original_experiments_complete=False)
    write(target/'report.json', proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.register:
        path = ROOT/CONFIG
        with path.open('x', encoding='utf-8') as f: json.dump(registration(), f, indent=2)
        return
    c = read(ROOT/CONFIG); original = verify(c)
    if args.probe_only:
        print(json.dumps(dict(passed=True, full_source_complete=(ROOT/SOURCE/'attempts/full_training/receipt.json').exists(),
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules, numerical_workers_started=0)))
        return
    if args.worker:
        worker(c); return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT/c['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base/'controller.lock')
    if owner is None: raise RuntimeError('Registered SB replay already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, all_original_experiments_complete=False)
    def update(v):
        state.update(v, captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json', state)
    try:
        while not (ROOT/SOURCE/'attempts/full_training/receipt.json').exists():
            update(dict(state='waiting_full_source_training_and_guard', **api['resource_snapshot']())); time.sleep(10)
        require_source_completion(original)
        if (base/'receipt.json').exists() or (base/'worker.stdout.log').exists():
            raise RuntimeError('Preserve terminal replay attempt; no automatic retry')
        with api['resource_slot'](ROOT/c['resource_lock'], c['settings'], update, poll_seconds=10):
            with (base/'worker.stdout.log').open('xb') as out, (base/'worker.stderr.log').open('xb') as err:
                command = [c['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='full_checkpoint_replay', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, c, api, update)
            proof = Path(c['artifact_root'])/'report.json'
            passed = code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base/'receipt.json', dict(passed=passed, exit_code=code, **usage, proof_sha256=sha(proof) if proof.exists() else None))
            if not passed: raise RuntimeError('Preserved terminal SB replay failure; no automatic retry')
        update(dict(state='complete_full_native_and_independent_strict_metric_audit', active_pid=None, passed=True))
    except Exception as error:
        update(dict(state='terminal_failure_preserved', active_pid=None, error=str(error))); raise
    finally:
        owner.close()


if __name__ == '__main__': main()
