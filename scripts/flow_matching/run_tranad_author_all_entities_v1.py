"""Run five source TranAD branches on every released SMAP/MSL/SMD entity.

Author preprocessing, sequential batches, double precision, five epochs,
full-population inference and source POT/PA/diagnosis are separate from all old
local-adaptation tracks. Seed1 preserves the release default; five fixed seeds
add robustness. A one-shot failure is retained, never silently retried.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.run_tranad_author_all_entities_v1'
CONFIG = 'configs/experiments/fm_tranad_author_all_entities.v1.json'
TEST = 'tests/test_fm_tranad_author_all_entities_v1.py'
MODEL_PREFIX = 'configs/models/fm_tranad_author_source_v1/'
VARIANTS = ('TranAD', 'TranAD_Basic', 'TranAD_Transformer', 'TranAD_Adversarial', 'TranAD_SelfConditioning')
SEEDS = [1103, 1104, 1105, 1106, 1107, 1]


def exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def entity_catalog(source):
    records = []
    with (source/'data/SMAP_MSL/labeled_anomalies.csv').open(encoding='utf-8', newline='') as f:
        metadata = list(csv.DictReader(f))
    for dataset in ['SMAP', 'MSL']:
        seen = set()
        for row in metadata:
            if row['spacecraft'] == dataset and row['chan_id'] not in seen:
                seen.add(row['chan_id']); records.append(dict(dataset=dataset, entity=row['chan_id']))
    records += [dict(dataset='SMD', entity=p.stem) for p in sorted((source/'data/SMD/train').glob('*.txt'))]
    if len(records) != 109 or len({(r['dataset'], r['entity']) for r in records}) != 109:
        raise ValueError('Every released54SMAP/27MSL/28SMD entity required')
    defaults = {('SMAP', 'P-1'), ('MSL', 'C-1'), ('SMD', 'machine-1-1')}
    return sorted(records, key=lambda r: ((r['dataset'], r['entity']) not in defaults, r['dataset'], r['entity']))


def registration():
    model = read(ROOT/(MODEL_PREFIX+'TranAD.json'))
    training_policy = 'configs/experiments/fm_source_cnf_full120.v2.json'
    readonly_policy = 'configs/reproducibility/fm_TAB_full878_original21_metric_suite.2026-10-16_v1.json'
    source = ROOT/model['source_root']
    entities = entity_catalog(source)
    jobs = []
    # Run all original CLI defaults first, then every remaining original entity.
    for default in [True, False]:
        for seed in SEEDS:
            for variant in VARIANTS:
                for record in entities:
                    is_default = (record['dataset'], record['entity']) in {('SMAP','P-1'),('MSL','C-1'),('SMD','machine-1-1')}
                    if is_default != default:
                        continue
                    identity = f"tranad_author__{variant}__{record['dataset']}__{record['entity']}__seed{seed}"
                    jobs.append(dict(id=identity, **record, variant=variant, seed=seed,
                        original_CLI_default_entity=is_default, original_source_default_seed=seed == 1,
                        model_config=MODEL_PREFIX+variant+'.json'))
    paths = [MODULE.replace('.', '/')+'.py', TEST, 'src/iia_benchmark/models/tranad_author_source_v1.py', training_policy, readonly_policy]
    paths += [MODEL_PREFIX+v+'.json' for v in VARIANTS]
    paths += [(source/p['path']).relative_to(ROOT).as_posix() for p in model['source_receipts']]
    raw = [p for d in ['SMAP_MSL', 'SMD'] for p in (source/'data'/d).rglob('*') if p.is_file() and p.suffix in ['.csv','.npy','.txt']]
    paths += [p.relative_to(ROOT).as_posix() for p in raw]
    return dict(schema_version=1, source_root=model['source_root'], entities=entities, jobs=jobs,
        primary_seeds=SEEDS[:5], source_default_seed=1, required_models=3270, required_primary_five_seed_cohorts=545,
        required_original_CLI_default_conditions=90, epochs=5, CPU_threads=2, python=sys.executable,
        state_root='experiments/runs/fm_tranad_author_all_entities_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/tranad_author_all_entities_v1',
        resource_reference_configuration=training_policy, preparation_resource_reference_configuration=readonly_policy,
        resource_lock=read(ROOT/training_policy)['resource_lock'],
        preparation_resource_lock=read(ROOT/readonly_policy)['resource_lock'],
        settings=dict(minimum_available_memory_bytes=24*2**30, minimum_commit_headroom_bytes=40*2**30,
            emergency_commit_headroom_bytes=16*2**30, maximum_worker_rss_bytes=10*2**30, maximum_worker_private_bytes=12*2**30),
        preparation_settings=read(ROOT/readonly_policy)['settings'],
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273, paper_scalar_targets=702),
        citation='https://doi.org/10.14778/3514061.3514067',
        author_commit='7ffb98d0c18189cc3d9ab732b4cb0278200a0af0',
        data_citations=['Frozen author data/SMAP_MSL including original first-row/exclusive labels', 'Frozen author data/SMD including original interpretation_label per-feature labels'],
        boundary='Five exact released class/forward definitions, original sin+cos encoding, no LayerNorm, literal LeakyReLU(True), '
            'source dataset lr, AdamW/StepLR, five epochs, no DataLoader shuffle, source prior-row padded windows, '
            'full one-batch source inference, complete per-feature POT/PA plus average-score POT/PA and Hit/NDCG. '
            'Only source-compatible sequential torch containers replace modern container keyword assumptions. '
            'All109 entities,5methods,source seed1 and5primary seeds remain required;default90 is a subset. '
            'Full fresh-process checkpoint array replay required before metric stage;no automatic retry. '
            'No old registered/local FP64 result is overwritten or reclassified;historical environment and paper-table equivalence unproven.',
        all_original_experiments_complete=False)


def verify(q):
    if q != registration() or len(q['jobs']) != 3270:
        raise ValueError('Complete original source, raw data, branches, seeds and budgets must remain frozen')
    return q['jobs']


def preparation_report(q):
    path = Path(q['artifact_root'])/'preparation/complete_author_preprocessing_and_real_preflight.json'
    guard_path = ROOT/q['state_root']/'attempts/preparation/guard_receipt.json'
    if not path.exists() or not guard_path.exists():
        return None
    report, guard = read(path), read(guard_path)
    if not report['passed'] or report['configuration_sha256'] != sha(ROOT/CONFIG) or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted'] or guard['proof_sha256'] != sha(path):
        raise ValueError('Original preprocessing/preflight failed; preserve without retry')
    if len(report['entities']) != 109 or len(report['files']) != 327 or {x['variant'] for x in report['actual_real_input_preflights']} != set(VARIANTS):
        raise ValueError('All109 complete source entities and all five actual real-input preflights required')
    for item in report['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Complete preprocessed original arrays changed')
    return report


def prepare(q):
    import copy
    import numpy as np
    import pandas as pd
    import torch
    from iia_benchmark.models.tranad_author_source_v1 import compile_definition, author_model
    torch.set_num_threads(q['CPU_threads'])
    output = Path(q['artifact_root'])/'preparation'; output.mkdir(parents=True, exist_ok=False)
    processed = output/'processed'; processed.mkdir()
    source = (ROOT/q['source_root']).resolve()
    ns = dict(os=os, np=np, pd=pd, output_folder=str(processed), copyfile=__import__('shutil').copyfile)
    for name in ['normalize3', 'load_and_save', 'load_and_save2', 'load_data']:
        compile_definition(source/'preprocess.py', name, ns)
    previous_cwd = Path.cwd()
    try:
        os.chdir(source)
        for dataset in ['SMAP', 'MSL', 'SMD']:
            ns['load_data'](dataset)
    finally:
        os.chdir(previous_cwd)
    files, shapes = [], []
    for entity in q['entities']:
        arrays = {}
        for role in ['train', 'test', 'labels']:
            path = processed/entity['dataset']/(entity['entity']+'_'+role+'.npy')
            a = np.load(path, allow_pickle=False)
            if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
                raise ValueError('Complete finite original author preprocessing required')
            arrays[role] = a
            files.append(dict(path=str(path), sha256=sha(path), shape=list(a.shape), dtype=a.dtype.str))
        if arrays['test'].shape != arrays['labels'].shape or not np.isin(arrays['labels'], [0,1]).all():
            raise ValueError('Every original feature-specific test label required')
        shapes.append(dict(**entity, train_shape=list(arrays['train'].shape), test_shape=list(arrays['test'].shape)))
    probes = []
    real = np.load(processed/'SMAP/P-1_train.npy', allow_pickle=False)
    for variant in VARIANTS:
        torch.manual_seed(1103)
        model, runtime = author_model(read(ROOT/(MODEL_PREFIX+variant+'.json')), 'SMAP', variant, real.shape[1])
        data = torch.DoubleTensor(real)
        windows = runtime['convert_to_windows'](data, model)[:128]
        src = windows.permute(1,0,2); target = src[-1].unsqueeze(0)
        model.eval(); predicted = model(src, target)
        values = predicted if isinstance(predicted, tuple) else (predicted,)
        assert all(v.shape == target.shape and torch.isfinite(v).all() for v in values)
        loss = sum((v-target).square().mean() for v in values); loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert not any(isinstance(x, torch.nn.LayerNorm) for x in model.modules())
        probes.append(dict(variant=variant, full_original_training_batch=128, parameters=sum(p.numel() for p in model.parameters()),
            output_shapes=[list(v.shape) for v in values], finite_forward_and_backward=True, optimizer_updates=0,
            real_input='SMAP/P-1 complete author-preprocessed first128 source windows', benchmark_performance=False))
    exclusive(output/'complete_author_preprocessing_and_real_preflight.json', dict(passed=True, configuration_sha256=sha(ROOT/CONFIG),
        entities=shapes, files=files, actual_real_input_preflights=probes, original_source_AST_unchanged=True,
        raw_files_overwritten=False, old_local_tracks_modified=False, complete_trained_models=0,
        all_original_experiments_complete=False))


def job_directory(q, job):
    return Path(q['artifact_root'])/'jobs'/job['id']


def prior_stage(q, job, phase):
    path = job_directory(q,job)/(phase+'.json')
    guard = read(ROOT/q['state_root']/'attempts'/job['id']/phase/'guard_receipt.json')
    report = read(path)
    expected = [q['python'],'-B','-X','utf8','-u','-m',MODULE,'--phase',phase,'--job-id',job['id']]
    if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['proof_sha256'] != sha(path) or guard['exact_child_handle']['command'] != expected
            or not report['passed'] or report['job'] != job):
        raise ValueError('Actual successful exact source stage guard required before successor')
    return report


def scalar_outcomes(values):
    import math
    converted = {k:float(v) if math.isfinite(float(v)) else None for k,v in values.items()}
    return dict(values=converted, nonfinite_fields=[k for k,v in converted.items() if v is None])


def worker(q, job, phase):
    import numpy as np
    import torch
    from iia_benchmark.models.tranad_author_source_v1 import author_model, load_author_metrics
    pre = preparation_report(q)
    if pre is None:
        raise ValueError('Actual successful complete preprocessing and all five real-input preflights required')
    torch.set_num_threads(q['CPU_threads']); torch.manual_seed(job['seed']); np.random.seed(job['seed'])
    model_config = read(ROOT/job['model_config'])
    model, runtime = author_model(model_config, job['dataset'], job['variant'], next(x['train_shape'][1] for x in pre['entities'] if x['dataset']==job['dataset'] and x['entity']==job['entity']))
    data_root = Path(q['artifact_root'])/'preparation/processed'/job['dataset']
    train = np.load(data_root/(job['entity']+'_train.npy'), allow_pickle=False)
    test = np.load(data_root/(job['entity']+'_test.npy'), allow_pickle=False)
    labels = np.load(data_root/(job['entity']+'_labels.npy'), allow_pickle=False)
    out = job_directory(q, job)
    checkpoint, scores_path = out/'checkpoint.pt', out/'full_feature_scores.npz'
    if phase == 'metrics':
        replay = prior_stage(q, job, 'replay')
        assert replay['passed'] and replay['score_sha256'] == sha(scores_path)
        assert replay['checkpoint_sha256'] == sha(checkpoint)
        with np.load(scores_path, allow_pickle=False) as z:
            train_loss, test_loss = z['train_loss'], z['test_loss']
        metrics = load_author_metrics(model_config, job['dataset'])
        feature_metrics, predictions = [], []
        for feature in range(test.shape[1]):
            values, pred = metrics['pot_eval'](train_loss[:,feature], test_loss[:,feature], labels[:,feature])
            feature_metrics.append(scalar_outcomes(values)); predictions.append(pred)
        final, final_pred = metrics['pot_eval'](train_loss.mean(axis=1), test_loss.mean(axis=1), (labels.sum(axis=1)>=1)+0)
        diagnosis = {**metrics['hit_att'](test_loss, labels), **metrics['ndcg'](test_loss, labels)}
        # Nonfinite source diagnosis outcomes must remain explicit, never zero-filled.
        diagnosis = {k:dict(value=float(v) if np.isfinite(v) else None, finite=bool(np.isfinite(v))) for k,v in diagnosis.items()}
        np.savez_compressed(out/'original_POT_PA_predictions.npz', feature=np.stack(predictions,axis=1), overall=final_pred)
        exclusive(out/'metrics.json', dict(passed=True, job=job, original_feature_POT_PA=feature_metrics,
            original_overall_POT_PA=scalar_outcomes(final), original_hit_ndcg=diagnosis,
            original_POT_risk=1e-5, original_POT_lm=metrics['lm'], undefined_original_metrics_preserved=True,
            ROC_AUC_field_is_binary_PA_prediction_AUC=True, test_labels_used_for_PA=True,
            original_source_metric_functions_unmodified=True, raw_score_sha256=sha(scores_path),
            full_TAB_training_or_author_historical_equivalence=False, all_original_experiments_complete=False))
        return
    train_data, test_data = torch.DoubleTensor(train), torch.DoubleTensor(test)
    train_windows = runtime['convert_to_windows'](train_data, model)
    test_windows = runtime['convert_to_windows'](test_data, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, .9)
    if phase == 'train':
        out.mkdir(parents=True, exist_ok=False)
        trace = []
        for epoch in range(q['epochs']):
            value, lr = runtime['backprop'](epoch, model, train_windows, train_data, optimizer, scheduler)
            trace.append(dict(epoch=epoch, mean_loss=float(value), learning_rate=float(lr),
                full_batches=(len(train)+model.batch-1)//model.batch, optimizer_updates=(len(train)+model.batch-1)//model.batch))
            write(out/'training_trace.json', dict(job=job, epochs=trace))
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            job=job, configuration_sha256=sha(ROOT/CONFIG), completed_epochs=5), checkpoint)
    else:
        training = prior_stage(q, job, 'train')
        assert training['checkpoint_sha256'] == sha(checkpoint) and training['score_sha256'] == sha(scores_path)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['job'] == job and saved['completed_epochs'] == 5
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    model.eval()
    # This preserves the author's full one-batch inference; no chunking or sampling.
    test_loss, test_prediction = runtime['backprop'](0, model, test_windows, test_data, optimizer, scheduler, training=False)
    train_loss, train_prediction = runtime['backprop'](0, model, train_windows, train_data, optimizer, scheduler, training=False)
    arrays = dict(train_loss=train_loss, test_loss=test_loss, train_prediction=train_prediction, test_prediction=test_prediction)
    assert train_loss.shape == train.shape and test_loss.shape == test.shape
    assert all(np.isfinite(x).all() for x in arrays.values())
    if phase == 'train':
        np.savez_compressed(scores_path, **arrays)
        exclusive(out/'train.json', dict(passed=True, job=job, completed_epochs=5,
            optimizer_updates=5*((len(train)+127)//128), train_points=len(train), test_points=len(test), features=train.shape[1],
            checkpoint_sha256=sha(checkpoint), score_sha256=sha(scores_path), original_source_training_and_inference=True,
            historical_author_environment_certified=False, all_original_experiments_complete=False))
    else:
        with np.load(scores_path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            assert all(saved[k].dtype == v.dtype and saved[k].shape == v.shape and saved[k].tobytes() == v.tobytes() for k,v in arrays.items())
        exclusive(out/'replay.json', dict(passed=True, job=job, all_full_feature_scores_and_predictions_byte_equal=True,
            fresh_process_replay=True, checkpoint_sha256=sha(checkpoint), score_sha256=sha(scores_path),
            new_optimizer_updates=0, all_original_experiments_complete=False))


def controller(q):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT/q['state_root']; base.mkdir(parents=True, exist_ok=True)
    lock = api['exclusive_lock'](base/'controller.lock')
    if lock is None or (base/'status.json').exists():
        if lock: lock.close()
        raise FileExistsError('Preserve every existing controller attempt')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        required_models=3270, required_primary_cohorts=545, completed_ids=[], failed_preserved={}, all_original_experiments_complete=False)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat()); write(base/'status.json', state)
    def stage(job, phase):
        key = 'preparation' if phase == 'prepare' else job['id']+'/'+phase
        attempt = base/'attempts'/key; attempt.mkdir(parents=True, exist_ok=False)
        settings = q['preparation_settings'] if phase == 'prepare' else q['settings']
        lock_path = q['preparation_resource_lock'] if phase == 'prepare' else q['resource_lock']
        control = dict(q, settings=settings)
        with api['resource_slot'](ROOT/lock_path, settings, update):
            command = [q['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--phase', phase]
            if job: command += ['--job-id', job['id']]
            env = dict(os.environ, PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'), PYTHONDONTWRITEBYTECODE='1',
                CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
            with (attempt/'stdout.log').open('xb') as stdout, (attempt/'stderr.log').open('xb') as stderr:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=proc.pid, create_time=psutil.Process(proc.pid).create_time(), command=command)
                update(dict(state='running_original_source_'+phase, active_pid=proc.pid, exact_child_handle=handle,
                    active_job=job['id'] if job else 'preparation'))
                def stage_update(value):
                    value = dict(value)
                    if value.get('state') == 'replaying_full_checkpoints':
                        value['state'] = 'running_original_source_'+phase
                    update(value)
                code, usage = capped_wait(proc, control, api, stage_update)
            path = Path(q['artifact_root'])/'preparation/complete_author_preprocessing_and_real_preflight.json' if phase == 'prepare' else job_directory(q,job)/(phase+'.json')
            passed = code == 0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
            exclusive(attempt/'guard_receipt.json', dict(passed=passed, exit_code=code, **usage, exact_child_handle=handle,
                proof_sha256=sha(path) if path.exists() else None))
            update(dict(active_pid=None))
            if not passed:
                raise ValueError('Original source stage failed; preserve all attempts without retry')
    try:
        update(dict(state='waiting_original_author_preprocessing'))
        stage(None, 'prepare')
        for job in q['jobs']:
            try:
                for phase in ['train', 'replay', 'metrics']:
                    stage(job, phase)
                state['completed_ids'].append(job['id'])
            except Exception as error:
                state['failed_preserved'][job['id']] = repr(error)
            update(dict(state='between_original_source_cases', active_pid=None))
        update(dict(state='all3270_attempts_terminal_other_experiments_remain'))
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--phase', choices=['prepare','train','replay','metrics']); parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.register:
        exclusive(ROOT/CONFIG, registration()); return
    q = read(ROOT/CONFIG); verify(q)
    if args.probe_only:
        print(json.dumps(dict(models=3270, primary_five_seed_cohorts=545, original_CLI_default_conditions=90,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules))); return
    if args.phase == 'prepare': prepare(q)
    elif args.phase: worker(q, next(j for j in q['jobs'] if j['id']==args.job_id), args.phase)
    else: controller(q)


if __name__ == '__main__':
    main()
