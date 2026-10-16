"""Recover only the full test of an actually completed1000-epoch CPU model.

No epoch history is invented: all100 retained validation checkpoints, native
CSV observations and the terminal Lightning loop/optimizer state are audited.
Original numerical test and independent full-LP replay code stay unchanged.
"""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import inspect as python_inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching import run_otcfm_singlecell_native_v4 as native
from scripts.flow_matching import audit_otcfm_singlecell_full_replay_v2 as replay
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.recover_otcfm_completed1000_CPU_test_v1'
CONFIG = 'configs/reproducibility/fm_otcfm_completed1000_CPU_test_recovery.v1.json'
TEST = 'tests/test_fm_otcfm_completed1000_CPU_test_recovery_v1.py'
SELECTED = 'otcfm_singlecell__released_yaml__otcfm__released_author_donor32606__leave1__seed43'


def registration():
    q = read(ROOT / native.QUEUE)
    j = next(j for j in q['jobs'] if j['id'] == SELECTED)
    old = Path(j['artifact_directory'])
    if len(list((old / 'checkpoints').glob('*.ckpt'))) != 101:
        raise ValueError('All100 validation checkpoints and actual final checkpoint required')
    guard = ROOT / q['state_root'] / 'attempts' / old.name / 'guard_receipt.json'
    files = [native.QUEUE, native.ENVIRONMENT, MODULE.replace('.', '/') + '.py', TEST,
        'scripts/flow_matching/run_otcfm_singlecell_native_v4.py',
        'scripts/flow_matching/audit_otcfm_singlecell_full_replay_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        str(guard), str(old / 'progress.json'), str(old / 'csv/version_0/metrics.csv'),
        *[str(p) for p in sorted((old / 'checkpoints').glob('*.ckpt'))]]
    settings = dict(q['settings'], minimum_commit_headroom_bytes=40 * 2**30)
    return dict(original_queue=native.QUEUE, canonical_id=SELECTED, original_job=j,
        original_terminal_guard=str(guard), original_artifact_root=str(old),
        preserved_checkpoint_files=101, observed_validation_checkpoints=100,
        state_root='experiments/runs/fm_otcfm_completed1000_CPU_test_recovery_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_completed1000_CPU_test_recovery_v1',
        python=q['python'], settings=settings,
        proof_settings=dict(q['settings'], minimum_commit_headroom_bytes=24 * 2**30),
        proof_boundary='Read-only checkpoint/CSV inspection, no model/data instantiation, training, ODE or LP. Full numerical tests/replays retain stronger40GiB admission and all original emergency/RSS/private caps.',
        resource_lock=q['resource_lock'], readonly_lock='experiments/runs/fm_otcfm_singlecell_readonly_replay_cpu.lock',
        native_metric_tolerance=dict(rtol=2e-6, atol=2e-6),
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in files],
        original_scope=q['original_scope'], citation=q['citation'],
        actual_new_model_training_updates=0, per_epoch_callback_history_available=False,
        all_original_experiments_complete=False)


def verify(c):
    if c != registration():
        raise ValueError('Preserved complete training, exact checkpoints and unchanged test recipe required')
    q = read(ROOT / c['original_queue'])
    native.verify(q)
    guard = read(c['original_terminal_guard'])
    if guard['exit_code'] != 1 or not guard['resource_guard_aborted'] or guard['result_sha256'] is not None:
        raise ValueError('Only the actual post-training resource-aborted case may recover')
    if (Path(c['original_artifact_root']) / 'result.json').exists():
        raise ValueError('Unexpected original full result; do not duplicate it')
    return q, c['original_job']


def validation_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        raw = list(csv.DictReader(stream))
    rows = [dict(epoch=int(r['epoch']) + 1, optimizer_updates=int(r['step']) + 1,
                 finite=math.isfinite(float(r['val/loss'])), validation_loss=float(r['val/loss']))
            for r in raw if r.get('val/loss')]
    if [r['epoch'] for r in rows] != list(range(10, 1001, 10)) or not all(r['finite'] for r in rows):
        raise ValueError('Every original100 finite validation observation must be present, without inferred epochs')
    if any(r['optimizer_updates'] != r['epoch'] * 10 for r in rows):
        raise ValueError('Actual native CSV counters must agree with complete10-batch epochs')
    return rows


def validate_checkpoint(checkpoint, epoch, torch):
    if checkpoint['epoch'] + 1 != epoch or checkpoint['global_step'] != epoch * 10:
        raise ValueError('Actual checkpoint epoch/update pair differs')
    batch = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']
    if any(batch[k] != epoch * 10 for k in ['ready', 'started', 'processed', 'completed']):
        raise ValueError('Every native Lightning full training batch counter required')
    model = checkpoint['state_dict']
    if not model or any(not torch.isfinite(v).all().item() for v in model.values()):
        raise ValueError('All complete model tensors must be finite')
    optimizers = checkpoint['optimizer_states']
    if len(optimizers) != 1 or len(optimizers[0]['state']) != 8:
        raise ValueError('All8 original AdamW parameter slots required')
    opt = optimizers[0]
    if (sum(v['exp_avg'].numel() for v in opt['state'].values()) != 9093
            or any(int(v['step']) != epoch * 10 for v in opt['state'].values())):
        raise ValueError('Every original optimizer moment and update counter required')
    if any(g['lr'] != .001 or g['weight_decay'] != 1e-5 for g in opt['param_groups']):
        raise ValueError('Frozen original AdamW hyperparameters differ')
    for state in opt['state'].values():
        if any(torch.is_tensor(v) and not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('Every full original optimizer tensor must be finite')


def prove_training(c, q, j):
    import importlib.metadata
    import platform
    import torch
    environment = read(ROOT / native.ENVIRONMENT)
    if (platform.python_version() != environment['python_version']
            or any(importlib.metadata.version(name) != version for name, version in environment['packages'].items())):
        raise ValueError('Actual isolated author environment must remain unchanged')
    old = Path(c['original_artifact_root'])
    observations = validation_rows(old / 'csv/version_0/metrics.csv')
    progress = read(old / 'progress.json')
    if progress['phase'] != 'full_original_test' or progress['epochs'] != 1000 or progress['optimizer_updates'] != 10000:
        raise ValueError('Actual full-training terminal progress required')
    checkpoints = []
    for row in observations:
        path = old / 'checkpoints' / ('epoch_%04d.ckpt' % (row['epoch'] - 1))
        checkpoint = torch.load(path, map_location='cpu')
        validate_checkpoint(checkpoint, row['epoch'], torch)
        checkpoints.append(dict(epoch=row['epoch'], optimizer_updates=checkpoint['global_step'], path=str(path), sha256=sha(path)))
    last = torch.load(old / 'checkpoints/last.ckpt', map_location='cpu')
    validate_checkpoint(last, 1000, torch)
    callback = next(v for k, v in last['callbacks'].items() if k.startswith('ModelCheckpoint'))
    best = min(observations, key=lambda v: v['validation_loss'])
    if (best['epoch'] != 350 or Path(callback['best_model_path']).name != 'epoch_0349.ckpt'
            or float(callback['best_model_score']) != best['validation_loss']
            or Path(progress['best_checkpoint']) != Path(callback['best_model_path'])):
        raise ValueError('Actual validation-selected epoch350 weights and all100 observations required')
    stopping = next(v for k, v in last['callbacks'].items() if k.startswith('EarlyStopping'))
    if stopping['stopped_epoch'] != 0 or stopping['patience'] != 100:
        raise ValueError('Original max1000 termination, not a shortened early-stop recipe required')
    if torch.cuda.is_initialized():
        raise ValueError('Original CPU recovery must not initialize CUDA')
    proof = dict(passed=True, canonical_id=j['id'], original_completed_epochs=1000,
        original_optimizer_updates=10000, original_training_batches=10000,
        actual_checkpoint_observations=checkpoints, actual_validation_history=observations,
        best_checkpoint=str(old / 'checkpoints/epoch_0349.ckpt'), best_checkpoint_sha256=sha(old / 'checkpoints/epoch_0349.ckpt'),
        best_epoch=350, selected_checkpoint_optimizer_updates=3500,
        final_checkpoint_sha256=sha(old / 'checkpoints/last.ckpt'), original_terminal_guard_sha256=sha(c['original_terminal_guard']),
        full_per_epoch_callback_history_available=False, no_unobserved_epoch_records_invented=True,
        environment_lock_sha256=sha(ROOT / native.ENVIRONMENT), torch_version=torch.__version__,
        actual_new_training_updates=0, cuda_context_initialized=False, all_original_experiments_complete=False)
    out = Path(c['artifact_root'])
    out.mkdir(parents=True, exist_ok=True)
    native.exclusive(out / 'actual_completed1000_training_proof.json', proof)
    return proof


def source_test(c, q, j):
    proof = read(Path(c['artifact_root']) / 'actual_completed1000_training_proof.json')
    torch, pl, dm, model = native.instantiate_case(q, j)
    if len(dm.train_dataloader()) != 10:
        raise ValueError('Original full10 training batches per epoch must remain')
    out = Path(c['artifact_root']) / 'recovered_source_test'
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    trainer = pl.Trainer(default_root_dir=out, accelerator='cpu', devices=1, precision=32,
        logger=False, enable_progress_bar=False, enable_checkpointing=False)
    values = trainer.test(model=model, datamodule=dm, ckpt_path=proof['best_checkpoint'])
    if len(values) != 1 or q['primary_metric'] not in values[0] or not all(map(math.isfinite, values[0].values())):
        raise ValueError('Every original full-distribution native test scalar required')
    if torch.cuda.is_initialized():
        raise ValueError('Original CPU test must remain on CPU')
    result = dict(status='completed_full_test_after_preserved_completed_training', job=j,
        actual_completed_epochs=1000, observed_checkpoint_training_history=proof['actual_checkpoint_observations'],
        validation_history=proof['actual_validation_history'], best_checkpoint=proof['best_checkpoint'],
        best_checkpoint_sha256=proof['best_checkpoint_sha256'], optimizer_updates=proof['selected_checkpoint_optimizer_updates'],
        trainer_global_step_after_readonly_test=int(trainer.global_step),
        update_counter_semantics='selected checkpoint actual3500 updates, not new updates or a fabricated training history',
        original_full_training_optimizer_updates=10000, actual_new_training_updates=0,
        per_epoch_callback_history_available=False, test_metrics=values[0],
        primary_heldout_W1=values[0][q['primary_metric']], training_proof_sha256=sha(Path(c['artifact_root']) / 'actual_completed1000_training_proof.json'),
        original_terminal_resource_abort_preserved=True, all_original_experiments_complete=False)
    native.exclusive(out / 'result.json', result)


def inspect_recovered(c, queue, job):
    verify(c)
    result_path = Path(c['artifact_root']) / 'recovered_source_test/result.json'
    result = read(result_path)
    guard = read(ROOT / c['state_root'] / 'source_test.guard.json')
    proof_path = Path(c['artifact_root']) / 'actual_completed1000_training_proof.json'
    if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['report_sha256'] != sha(result_path) or result['training_proof_sha256'] != sha(proof_path)
            or result['job'] != job or result['actual_completed_epochs'] != 1000
            or result['actual_new_training_updates'] != 0 or len(result['observed_checkpoint_training_history']) != 100):
        raise ValueError('Actual recovered complete source test and unchanged sparse observations required')
    return result, dict(original_guard_sha256=sha(c['original_terminal_guard']),
        completed_training_proof_sha256=sha(proof_path), recovered_source_result_sha256=sha(result_path),
        actual_completed_training_epochs=1000, actual_completed_training_updates=10000,
        no_missing_epoch_histories_invented=True)


def observed_epoch(result, epoch):
    rows = [r for r in result['observed_checkpoint_training_history'] if r['epoch'] == epoch]
    if len(rows) != 1:
        raise ValueError('An actual retained checkpoint observation is required')
    return rows[0]


def replay_worker_ast():
    tree = ast.parse(python_inspect.getsource(replay.worker))
    class MetadataOnly(ast.NodeTransformer):
        histories = 0
        lookups = 0
        def visit_Subscript(self, node):
            if ast.dump(node) == ast.dump(ast.parse("result['epoch_history'][last_epoch-1]", mode='eval').body):
                self.lookups += 1
                return ast.parse('_observed_epoch(result,last_epoch)', mode='eval').body
            node = self.generic_visit(node)
            if ast.dump(node) == ast.dump(ast.parse("result['epoch_history']", mode='eval').body):
                self.histories += 1
                return ast.parse("result['observed_checkpoint_training_history']", mode='eval').body
            return node
    change = MetadataOnly()
    tree = change.visit(tree)
    if change.lookups != 1 or change.histories != 1:
        raise ValueError('Only two training-provenance reads may change; all native test/LP equations retained')
    return ast.fix_missing_locations(tree)


def native_replay(c, q, j):
    namespace = dict(replay.__dict__, CONFIG=CONFIG, _observed_epoch=observed_epoch,
        inspect=lambda queue, job: inspect_recovered(c, queue, job))
    exec(compile(replay_worker_ast(), __file__, 'exec'), namespace)
    namespace['worker'](dict(c, artifact_root=c['artifact_root'] + '/independent_native_replay'), q, j)


def phase_report(c, phase):
    root = Path(c['artifact_root'])
    if phase == 'proof':
        return root / 'actual_completed1000_training_proof.json'
    if phase == 'source_test':
        return root / 'recovered_source_test/result.json'
    return root / 'independent_native_replay' / Path(c['original_artifact_root']).name / 'full_checkpoint_replay.json'


def controller(c, q, j):
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    install_cached_loader(lambda value: None)
    api = resource_api()
    # The actual failed numeric child must be absent; other CPU/GPU models run.
    old = read(c['original_terminal_guard'])['exact_child_handle']
    try:
        previous = psutil.Process(old['pid'])
    except psutil.NoSuchProcess:
        previous = None
    if previous is not None and (previous.create_time() == old['create_time'] or previous.cmdline() == old['command']):
        raise RuntimeError('Original failed numerical handle is still present; never duplicate it')
    base = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=False)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Completed1000 CPU recovery already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        canonical_id=SELECTED, completed_phases=[], new_model_training_updates=0,
        recovered_CPU_condition_complete=False, all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    update(dict(state='starting_completed_training_recovery'))
    try:
        for phase in ['proof', 'source_test', 'native_replay']:
            settings = c['proof_settings'] if phase == 'proof' else c['settings']
            lock = c['readonly_lock'] if phase == 'proof' else c['resource_lock']
            while True:
                with api['resource_slot'](ROOT / lock, settings, update):
                    second = None if phase == 'proof' else api['exclusive_lock'](ROOT / c['readonly_lock'])
                    if phase != 'proof' and second is None:
                        update(dict(state='waiting_for_original_readonly_test_lock'))
                        time.sleep(1)
                        continue
                    try:
                        command = [str(c['python']), '-B', '-X', 'utf8', '-u', '-m', MODULE, '--phase', phase]
                        with (base / (phase + '.stdout.log')).open('xb') as out, (base / (phase + '.stderr.log')).open('xb') as err:
                            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
                                    PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                            update(dict(state='running_' + phase, active_phase=phase, active_pid=child.pid, exact_child_handle=handle))
                            code, usage = capped_wait(child, dict(settings=settings), api, update)
                    finally:
                        if second is not None:
                            second.close()
                break
            report = phase_report(c, phase)
            passed = code == 0 and not usage['resource_guard_aborted'] and report.exists()
            if passed:
                record = read(report)
                passed = record.get('passed', False) if phase == 'proof' else record.get('status') == (
                    'completed_full_test_after_preserved_completed_training' if phase == 'source_test' else 'passed')
            native.exclusive(base / (phase + '.guard.json'), dict(passed=passed, exit_code=code,
                exact_child_handle=handle, **usage, report_path=str(report), report_sha256=sha(report) if report.exists() else None))
            if not passed:
                update(dict(state='failed_preserved_requires_isolated_repair', active_pid=None))
                return
            state['completed_phases'].append(phase)
            update(dict(state='between_recovery_phases', active_pid=None))
        update(dict(state='complete_CPU_training_test_and_independent_full_replay_recovered',
                    recovered_CPU_condition_complete=True, active_pid=None))
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--phase', choices=['proof', 'source_test', 'native_replay'])
    args = parser.parse_args()
    if args.register:
        native.exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    q, j = verify(c)
    if args.probe_only:
        print(json.dumps(dict(canonical_id=SELECTED, full_recovery_cases=1, original_full_scope_cases=120,
            original_epochs_required=1000, preserved_checkpoints=101, new_training_updates=0,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
        return
    if args.phase:
        {'proof': prove_training, 'source_test': source_test, 'native_replay': native_replay}[args.phase](c, q, j)
    else:
        controller(c, q, j)


if __name__ == '__main__':
    main()
