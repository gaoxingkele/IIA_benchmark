"""Automatic full checkpoint and independent array review of all500 paper models.

Training queues and frozen numerical sources remain unchanged. No training,
sampling, automatic retry, reduced coverage or leaderboard promotion occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT, read, sha, write

FAMILIES = {
    'main': ('configs/experiments/fm_anomaly_transformer_paper_contract.v1.json', 25,
             'scripts.flow_matching.run_anomaly_transformer_paper_contract_v1'),
    'sensitivity': ('configs/experiments/fm_anomaly_transformer_paper_sensitivity.v1.json', 400,
                    'scripts.flow_matching.run_anomaly_transformer_paper_sensitivity_v1'),
    'max': ('configs/experiments/fm_anomaly_transformer_paper_max_ablation.v1.json', 50,
            'scripts.flow_matching.run_anomaly_transformer_paper_max_ablation_v1'),
    'plain': ('configs/experiments/fm_plain_transformer_paper_baseline.v1.json', 25,
              'scripts.flow_matching.run_plain_transformer_paper_baseline_v1')}
MODULE = 'scripts.flow_matching.run_anomaly_transformer_full_checkpoint_review_v1'


def verify(runtime):
    from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent
    base = read(ROOT/FAMILIES['main'][0])
    if (runtime['families'] != {k: dict(configuration=v[0], required_models=v[1], module=v[2]) for k, v in FAMILIES.items()}
            or runtime['required_models'] != 500 or runtime['settings'] != base['settings']
            or runtime['python'] != base['python'] or runtime['resource_lock'] != base['resource_lock']
            or runtime['original_scope'] != base['original_scope']
            or runtime['formula_tolerance'] != dict(rtol=2e-6, atol=2e-6)
            or runtime['metric_absolute_tolerance'] != 2e-12
            or runtime['native_array_comparison'] != 'exact_dtype_shape_and_content_bytes'
            or runtime['cpu_threads'] != 2):
        raise ValueError('All500 models and frozen full replay/resource contracts required')
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen full checkpoint review input changed: '+pin['path'])
    cases = {}
    for family, (path, count, module) in FAMILIES.items():
        config = read(ROOT/path)
        importlib.import_module(module).verify(config)
        if len(config['jobs']) != count:
            raise ValueError('Incomplete original training family')
        for job in config['jobs']:
            if job['id'] in cases:
                raise ValueError('Repeated model; baseline scores must reuse one training run')
            cases[job['id']] = dict(family=family, path=path, config=config, job=job)
    if len(cases) != 500 or runtime['required_ids'] != list(cases):
        raise ValueError('Complete unique frozen model list required')
    return cases, parent.verify(base)


def parameters(case):
    return (case['job']['parameters'] if case['family'] == 'sensitivity'
            else dict(window=100, weight=3, layers=3, hidden=512))


def streams(case):
    if case['family'] == 'plain':
        return ['all.reconstruction']
    return ['all.reconstruction', 'all.association', 'all.addition', 'all.multiplication']+[
        f'layer{i}.multiplication' for i in range(1, parameters(case)['layers']+1)]


def source_paths(case):
    config, job = case['config'], case['job']
    base = Path(config['artifact_root'])/job['id']
    cache = Path(config['data_cache_root'])/job['dataset']
    receipt = ROOT/config['state_root']/'receipts'/(job['id']+'.json')
    return base, cache, receipt


def source_ready(case):
    """A terminal failed training receipt is not mistaken for a pending model."""
    base, _, receipt = source_paths(case)
    if not receipt.exists():
        return False
    guard = read(receipt)
    if (guard.get('job') != case['job']['id'] or not guard.get('passed')
            or guard.get('exit_code') != 0 or guard.get('resource_guard_aborted') is not False):
        raise ValueError('Original training guard failed or differs; source attempt preserved')
    if not (base/'result.json').exists():
        raise ValueError('Passed original training receipt has no completed source result')
    return True


def check_epoch_budget(result, split, window, backward_factor):
    epochs = result['epochs']
    if not 1 <= len(epochs) <= 10:
        raise ValueError('Full original maximum10/early-stop training required')
    patience, best, updates = 3, None, 0
    for index, epoch in enumerate(epochs, 1):
        losses = epoch['validation_losses']
        if len(losses) != 2 or not all(math.isfinite(v) for v in losses):
            raise ValueError('Complete finite independent validation losses required')
        improved = best is None or all(a <= b for a, b in zip(losses, best))
        patience = 3 if improved else patience-1
        if improved:
            best = losses
        if (epoch['epoch'] != index or epoch['steps'] != math.ceil((split['train']//window)/32)
                or epoch['train_windows'] != split['train']//window
                or epoch['validation_windows'] != split['validation']//window
                or epoch['checkpoint_saved'] != improved or epoch['remaining_patience'] != patience
                or epoch['learning_rate'] != .0001*.5**max(index-2, 0)
                or (patience == 0 and index != len(epochs))):
            raise ValueError('Original full epoch coverage/learning rate/checkpoint selector differs')
        updates += epoch['steps']
    if len(epochs) < 10 and patience != 0:
        raise ValueError('Partial training cannot stand in for the original early stopping')
    if result['optimizer_updates'] != updates or result['actual_backward_calls'] != updates*backward_factor:
        raise ValueError('Complete optimizer/backward counts required')


def check_source(case, contract):
    base, cache, receipt = source_paths(case)
    result = read(base/'result.json'); guard = read(receipt)
    if (result['status'] != 'completed' or result['job'] != case['job']
            or result['configuration_sha256'] != sha(ROOT/case['path'])
            or guard['job'] != case['job']['id'] or not guard['passed'] or guard['exit_code'] != 0
            or guard['resource_guard_aborted'] or result['checkpoint_selection_uses_test']
            or result['threshold_calibration_uses_test'] or result['strict_TAB_result']
            or result['paper_algorithm_equivalence_certified']):
        raise ValueError('Full original result, passed training guard and truthful boundaries required')
    expected = {'model.pt', 'epochs.json', 'source_patch.json', 'covered_test_labels.npy',
                'train_scores.npz', 'validation_scores.npz', 'test_scores.npz'}
    actual = {Path(a['path']).name for a in result['artifacts']}
    if actual != expected or len(result['artifacts']) != len(expected):
        raise ValueError('Every original full-model artifact required exactly once')
    for item in result['artifacts']:
        if Path(item['path']).resolve() != (base/Path(item['path']).name).resolve() or sha(item['path']) != item['sha256']:
            raise ValueError('Original model artifact changed or points elsewhere')
    if read(base/'epochs.json') != result['epochs']:
        raise ValueError('Saved epoch trace differs from the completed result')
    manifest = read(cache/'manifest.json')
    split = contract['reported_table13_splits'][case['job']['dataset']]
    if (manifest != result['data_audit'] or sha(cache/'manifest.json') != result['data_manifest_sha256']
            or manifest['split'] != split or manifest['contract_sha256'] != case['config']['contract_sha256']
            or manifest['dataset'] != case['job']['dataset']):
        raise ValueError('Full primary-paper data split/cache binding differs')
    if {a['name'] for a in manifest['artifacts']} != {'train.npy', 'validation.npy', 'test.npy', 'labels.npy', 'scaler.json'}:
        raise ValueError('Every original derived data artifact required')
    for item in manifest['artifacts']:
        if sha(cache/item['name']) != item['sha256']:
            raise ValueError('Derived full data changed')
    for item in manifest['source_data']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Original raw dataset content changed')
    scaler = read(cache/'scaler.json')
    if scaler['fitted_rows'] != split['train'] or not scaler['fit_on_training_only']:
        raise ValueError('Normalization must be fitted to the independent training subset')
    window = parameters(case)['window']
    if (result['covered_points'] != {k: split[k]//window*window for k in ['train', 'validation', 'test']}
            or result['omitted_tail_points'] != {k: split[k]%window for k in ['train', 'validation', 'test']}
            or set(result['metrics']) != set(streams(case))):
        raise ValueError('All original covered timestamps and every scoring stream required')
    check_epoch_budget(result, split, window, 1 if case['family'] in ['max', 'plain'] else 2)
    return result, manifest, dict(result_sha256=sha(base/'result.json'), training_guard_sha256=sha(receipt),
        checkpoint_sha256=sha(base/'model.pt'), data_manifest_sha256=sha(cache/'manifest.json'),
        configuration_sha256=sha(ROOT/case['path']))


def make_model(case, cache_manifest, destination):
    """Fresh copied source, exact frozen patch, strict checkpoint restoration."""
    import torch
    from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent
    base, _, _ = source_paths(case)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/case['config']['source_directory'], destination/'source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = destination/'source/model/attn.py'
    source = path.read_text(encoding='utf-8')
    patch = read(base/'source_patch.json')
    if sha(path) != patch['original_sha256']:
        raise ValueError('Original source attention differs')
    if case['family'] == 'plain':
        from scripts.flow_matching.run_plain_transformer_paper_baseline_v1 import patched_attention
        source = patched_attention('model/attn.py', source)
    elif case['family'] == 'max':
        from scripts.flow_matching.run_anomaly_transformer_paper_max_ablation_v1 import patched_attention
        source = patched_attention('model/attn.py', source, case['job']['variant'])
    else:
        source = parent.author.patched('model/attn.py', source)
    path.write_text(source, encoding='utf-8')
    if sha(path) != patch['runtime_sha256']:
        raise ValueError('Exact numerical attention patch differs')
    if any(k == 'model' or k.startswith('model.') for k in sys.modules):
        raise RuntimeError('One model per fresh interpreter; no cross-variant import reuse')
    sys.path.insert(0, str(destination/'source'))
    from model.AnomalyTransformer import AnomalyTransformer
    values = parameters(case); features = cache_manifest['split']['features']
    model = AnomalyTransformer(win_size=values['window'], enc_in=features, c_out=features,
        d_model=values['hidden'], n_heads=8, e_layers=values['layers'], d_ff=512, dropout=0.0)
    state = torch.load(base/'model.pt', map_location='cpu', weights_only=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint tensors')
    model.load_state_dict(state, strict=True)
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Actual replay parameters must be on CPU')
    return model.eval()


def native_worker(runtime, case, contract):
    import numpy as np
    import torch
    from scripts.flow_matching import anomaly_transformer_independent_math_v1 as independent
    from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.set_default_device('cpu')
    result, manifest, binding = check_source(case, contract)
    base, cache, _ = source_paths(case)
    output = Path(runtime['artifact_root'])/case['job']['id']/'native'
    model = make_model(case, manifest, output)
    if sum(p.numel() for p in model.parameters()) != result['parameter_count']:
        raise ValueError('Original complete model parameter count differs')
    scorer = parent.score_components
    plain = case['family'] == 'plain'
    if plain:
        from scripts.flow_matching.run_plain_transformer_paper_baseline_v1 import reconstruction_scores
        scorer = reconstruction_scores
    w, layers = parameters(case)['window'], parameters(case)['layers']
    component_key = 'diagnostic.layer_attention_entropy' if plain else 'component.layer_discrepancies'
    records = []; start = time.perf_counter()
    for split in ['train', 'validation', 'test']:
        data = np.load(cache/(split+'.npy'), mmap_mode='r')
        if list(data.shape) != [manifest['split'][split], manifest['split']['features']] or data.dtype != np.float32:
            raise ValueError('Full float32 model inputs required')
        windows = len(data)//w
        buffers = {k: [] for k in streams(case)}
        buffers.update({'component.reconstruction': [], component_key: []})
        maximum_errors = {}; batches = 0
        with torch.no_grad():
            for index in range(0, windows, 32):
                stop = min(index+32, windows)
                x = torch.from_numpy(np.array(data[index*w:stop*w], copy=True).reshape(stop-index, w, -1))
                reconstruction, series, priors, _ = model(x)
                scores, rec, layer = scorer(reconstruction, x, priors, series)
                native = {k: v.numpy() for k, v in scores.items()}
                native.update({'component.reconstruction': rec.numpy(), component_key: layer.numpy()})
                checked, irec, ilayers = independent.formula_scores(reconstruction.numpy(), x.numpy(),
                    [p.numpy() if p is not None else None for p in priors], [s.numpy() for s in series], plain)
                checked.update({'component.reconstruction': irec, component_key: ilayers})
                errors = independent.compare_formula(native, checked)
                for k, value in errors.items():
                    maximum_errors[k] = max(maximum_errors.get(k, 0.), value)
                for k, value in native.items():
                    buffers[k].append(value.transpose(1, 2, 0).reshape(-1, layers) if k == component_key else value.reshape(-1))
                batches += 1
        arrays = {k: np.concatenate(v, axis=0) for k, v in buffers.items()}
        original = base/(split+'_scores.npz')
        with np.load(original, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError('All original native score/components required')
            for k, value in arrays.items():
                independent.require_identical(value, saved[k])
        path = output/(split+'_scores.npz'); np.savez(path, **arrays)
        records.append(dict(split=split, covered_points=windows*w, omitted_tail_points=len(data)%w,
            batches=batches, independent_formula_checked_points=windows*w,
            independent_formula_maximum_absolute_errors=maximum_errors,
            arrays={k: independent.array_digest(v) for k, v in arrays.items()},
            replay_path=str(path), replay_sha256=sha(path), source_score_sha256=sha(original)))
    write(output/'native_replay.json', dict(passed=True, job=case['job'], family=case['family'],
        runtime_sha256=sha(ROOT/runtime['_config_path']), source_binding=binding,
        parameter_count=result['parameter_count'], records=records, wall_seconds=time.perf_counter()-start,
        training_updates_performed=0, all_covered_points_reinferred=True,
        native_dtype_shape_content_identical=True, formula_tolerance=independent.FORMULA_TOLERANCE,
        paper_equivalence_certified=False, strict_TAB_result=False))


def review_worker(runtime, case, contract):
    import numpy as np
    from scripts.flow_matching import anomaly_transformer_independent_math_v1 as independent
    result, manifest, binding = check_source(case, contract)
    base, cache, _ = source_paths(case)
    output = Path(runtime['artifact_root'])/case['job']['id']
    native_path = output/'native/native_replay.json'; native = read(native_path)
    if (not native['passed'] or native['source_binding'] != binding or native['job'] != case['job']
            or native['runtime_sha256'] != sha(ROOT/runtime['_config_path'])
            or not native['native_dtype_shape_content_identical'] or not native['all_covered_points_reinferred']
            or native['formula_tolerance'] != independent.FORMULA_TOLERANCE
            or native['training_updates_performed'] != 0
            or [r['split'] for r in native['records']] != ['train', 'validation', 'test']):
        raise ValueError('Complete unchanged native full checkpoint proof required')
    covered = result['covered_points']['test']
    labels = np.load(base/'covered_test_labels.npy', allow_pickle=False)
    independent.require_identical(labels, np.asarray(np.load(cache/'labels.npy', mmap_mode='r')[:covered]))
    for row in native['records']:
        split = row['split']
        if (sha(row['replay_path']) != row['replay_sha256']
                or sha(base/(split+'_scores.npz')) != row['source_score_sha256']
                or row['covered_points'] != result['covered_points'][split]
                or row['independent_formula_checked_points'] != row['covered_points']
                or row['omitted_tail_points'] != result['omitted_tail_points'][split]
                or row['batches'] != math.ceil(row['covered_points']/parameters(case)['window']/32)):
            raise ValueError('Complete native score coverage and unchanged artifacts required')
        with np.load(row['replay_path'], allow_pickle=False) as replay, np.load(base/(split+'_scores.npz'), allow_pickle=False) as saved:
            if set(replay.files) != set(saved.files) or set(row['arrays']) != set(replay.files):
                raise ValueError('All original full score streams/components required')
            for key in replay.files:
                independent.require_identical(replay[key], saved[key])
                if independent.array_digest(replay[key]) != row['arrays'][key]:
                    raise ValueError('Native array proof content differs')
    evaluation = contract['paper_evaluation']; dataset = case['job']['dataset']
    metrics = {}
    with np.load(base/'test_scores.npz', allow_pickle=False) as test, np.load(base/'validation_scores.npz', allow_pickle=False) as validation:
        for stream in streams(case):
            metrics[stream] = independent.independent_evaluation(labels, test[stream], validation[stream],
                evaluation['r_percent'][dataset], evaluation['fixed_delta'][dataset],
                evaluation['ROC_predefined_r_percent'], case['family'] == 'plain')
            independent.require_same_metrics(result['metrics'][stream], metrics[stream])
    output.joinpath('independent').mkdir(exist_ok=False)
    write(output/'independent/review.json', dict(passed=True, job=case['job'], source_binding=binding,
        native_replay_sha256=sha(native_path), all_original_streams_and_covered_arrays_checked=True,
        all_original_thresholds_confusion_counts_PA_and_ranking_metrics_checked=True,
        independent_metrics=metrics, test_used_for_calibration=False,
        independent_raw_preprocessing_value_reconstruction_performed=False,
        training_updates_performed=0, fresh_inference_performed_by_this_review=False,
        paper_equivalence_certified=False, strict_TAB_result=False,
        all_original_experiments_complete=False))


def controller(runtime, cases):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    import psutil
    api = install_facades()
    state_root = ROOT/runtime['state_root']; state_root.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](state_root/'controller.lock')
    if lane is None:
        raise RuntimeError('Full checkpoint review already has a live owner')
    common = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_models=500,
        original_scope=runtime['original_scope'], all_original_experiments_complete=False,
        paper_equivalence_certified=False, strict_TAB_result=False)
    verified, failed = [], {}
    def update(**values):
        write(state_root/'status.json', dict(common, verified_model_ids=list(verified),
            verified_model_count=len(verified), failed_preserved=dict(failed),
            failed_preserved_count=len(failed), captured_utc=datetime.now(timezone.utc).isoformat(), **values))
    def stage(case, phase):
        directory = state_root/'attempts'/case['job']['id']/phase
        receipt_path = directory/'guard_receipt.json'
        proof = Path(runtime['artifact_root'])/case['job']['id']/('native/native_replay.json' if phase == 'native' else 'independent/review.json')
        if receipt_path.exists():
            receipt = read(receipt_path)
            if (receipt['passed'] and receipt['exit_code'] == 0 and not receipt['resource_guard_aborted']
                    and receipt['proof_sha256'] == sha(proof) and read(proof)['passed']):
                contract = read(ROOT/case['config']['contract'])
                _, _, binding = check_source(case, contract)
                if read(proof)['source_binding'] != binding:
                    raise ValueError('Existing stage belongs to changed original source artifacts')
                return True
            raise ValueError('Prior failed replay/review is preserved; no automatic retry')
        if directory.exists() or proof.exists() or (phase == 'native' and (Path(runtime['artifact_root'])/case['job']['id']).exists()):
            raise ValueError('Prior incomplete replay/review preserved for explicit diagnosis')
        directory.mkdir(parents=True, exist_ok=False)
        command = [runtime['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE,
            '--config', runtime['_config_path'], '--stage', phase, '--job-id', case['job']['id']]
        with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
                         OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            handle = dict(pid=process.pid, create_time=psutil.Process(process.pid).create_time(), command=command)
            write(directory/'launch.json', handle)
            update(state='running_full_'+phase, active_pid=process.pid, active_create_time=handle['create_time'], job=case['job']['id'])
            code, peaks = api['guarded_wait'](process, runtime['settings'], poll_seconds=2)
        passed = code == 0 and not peaks['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
        write(receipt_path, dict(passed=passed, exit_code=code, **peaks,
            proof_sha256=sha(proof) if proof.exists() else None, source_configuration=case['path'],
            source_configuration_sha256=sha(ROOT/case['path']), exact_child_handle=handle))
        return passed
    try:
        while len(verified)+len(failed) < len(cases):
            progress = False
            for identity, case in cases.items():
                if identity in verified or identity in failed:
                    continue
                try:
                    if not source_ready(case):
                        continue
                except Exception as error:
                    failed[identity] = str(error); continue
                def waiting(value):
                    update(**dict(value, active_pid=None, job=identity))
                try:
                    with api['resource_slot'](ROOT/runtime['resource_lock'], runtime['settings'], waiting, poll_seconds=10):
                        if not stage(case, 'native') or not stage(case, 'independent'):
                            raise ValueError('Full guarded native/independent stage failed; preserved without retry')
                    verified.append(identity); progress = True
                except Exception as error:
                    failed[identity] = str(error)
                update(state='scanning_actual_full_training_completions', active_pid=None, job=None)
            update(state='waiting_actual_full_training_completion', active_pid=None, job=None)
            if not progress:
                time.sleep(10)
        update(state='all500_attempts_terminal_other_original_experiments_remain', active_pid=None, job=None)
    finally:
        lane.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--stage', choices=['native', 'independent']); parser.add_argument('--job-id')
    args = parser.parse_args(); runtime = read(ROOT/args.config)
    cases, contract = verify(runtime); runtime['_config_path'] = args.config
    if args.probe_only:
        import json
        print(json.dumps(dict(passed=True, required_models=len(cases),
            family_counts={k: sum(c['family'] == k for c in cases.values()) for k in FAMILIES},
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            numerical_workers_started=0, all_original_experiments_complete=False)))
    elif args.stage:
        (native_worker if args.stage == 'native' else review_worker)(runtime, cases[args.job_id], contract)
    else:
        controller(runtime, cases)


if __name__ == '__main__':
    main()
