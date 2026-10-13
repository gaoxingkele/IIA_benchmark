"""Freeze the terminal68-epoch attempt and register complete native recovery."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha

PREFIX = 'configs/reproducibility/fm_rcgan_seed1106_'
CHAIN = PREFIX + 'full68_random_order_chain.2026-10-13_v1.json'
STATE = PREFIX + 'full68_actual61_resume_state.2026-10-13_v1.json'
RESUME = PREFIX + 'full432_state_recovery.2026-10-13_v1.json'
DISPATCH = PREFIX + 'full68_recovery_dispatch.2026-10-13_v1.json'
RESULT = 'projects/flow_matching_research/results/2026-10-13/rcgan_seed1106_full68_recovery_v1'


def pin(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=sha(path))


def register():
    import psutil
    queue_path = 'configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json'
    queue = read(ROOT / queue_path)
    job = next(j for j in queue['jobs'] if j['id'] == 'rcgan_mnistfull__seed1106')
    output = ROOT / job['output_directory']
    terminal = ROOT / queue['state_root'] / 'receipts' / (job['id'] + '.json')
    receipt = read(terminal)
    progress = read(output / 'progress.json')
    if psutil.pid_exists(12384) and psutil.Process(12384).create_time() == 1791863687.9790325:
        raise ValueError('Never freeze a live training process')
    if (receipt['exit_code'] == 0 or not receipt['resource_guard_aborted']
            or receipt['full_native_result_verified'] or progress['job'] != job
            or progress['full_epochs_completed'] != 68 or progress['required_epochs'] != 500
            or progress['full_training_complete']
            or any(progress['optimizer_updates'][k] != 68 * 1284 for k in ('D', 'G'))):
        raise ValueError('Require the actual resource-aborted full68-of500 attempt')
    evaluations = read(output / 'completed_full_evaluators.json')
    if ([e['epoch'] for e in evaluations] != list(range(68))
            or sum(e['sigma_optimizer_updates'] for e in evaluations) != progress['optimizer_updates']['sigma']):
        raise ValueError('All68 original full evaluator ledgers required')
    trace = output / 'experiments/traces' / (job['id'] + '.trace.txt')
    rows = trace.read_text(encoding='utf-8').splitlines()
    if rows[0] != 'epoch time D_loss G_loss mmd2 that pdf real_pdf' or [int(x.split()[0]) for x in rows[1:]] != list(range(61)):
        raise ValueError('Preserve the actual61 trace rows and seven missing elapsed times')
    targets = [ROOT / p for p in (CHAIN, STATE, RESUME, DISPATCH, RESULT)]
    if any(p.exists() for p in targets):
        raise FileExistsError('Preserve every registration and failed attempt')
    common = [pin(ROOT / queue_path), pin(terminal), pin(output / 'progress.json'),
              pin(output / 'completed_full_evaluators.json'), pin(output / 'full_native_split_indices.npz'),
              pin(output / 'experiments/settings' / (job['id'] + '.txt')), pin(trace)]
    filenames = [
        'rcgan_seed1106_resume_binding_v1.py', 'audit_rcgan_seed1106_chain_v1.py',
        'audit_rcgan_seed1106_state_v1.py', 'rcgan_seed1106_resume_runtime_v1.py',
        'run_rcgan_seed1106_recovery_v1.py', 'run_rcgan_seed1106_recovery_dispatch_v1.py',
        'register_rcgan_seed1106_recovery_v1.py', 'audit_rcgan_random_order_chain_v1.py',
        'audit_rcgan_checkpoint_resume_state_v1.py', 'rcgan_native_training_resume_runtime_v1.py',
        'run_rcgan_native_state_recovery_v1.py', 'run_rcgan_full366_native_audit_dispatch_v1.py',
        'cached_resource_snapshot_v1.py', 'rcgan_native_training_runtime_v2.py',
        'rcgan_seed1105_resume_binding_v1.py', 'run_preserved_native_gan_retry_v1.py',
    ]
    common += [pin(ROOT / 'scripts/flow_matching' / n) for n in filenames]
    common += [pin(ROOT / 'tests/test_fm_rcgan_seed1106_recovery_v1.py')]
    checkpoints, random_states, parameter_dumps = [], [], []
    expected = {'checkpoint', 'model.data-00000-of-00001', 'model.index', 'model.meta',
                'numpy_random_state.npz', 'python_random_state.json'}
    for epoch in range(68):
        directory = output / 'native_checkpoints' / ('epoch_%03d' % epoch)
        files = sorted(directory.iterdir())
        if {p.name for p in files} != expected or not all(p.is_file() for p in files):
            raise ValueError('Every full checkpoint, optimizer and RNG file required')
        for path in files:
            item = pin(path)
            checkpoints.append(item)
            if path.suffix in ('.npz', '.json'): random_states.append(item)
        dump = output / 'experiments/parameters' / (job['id'] + '_' + str(epoch) + '.npy')
        if dump.exists(): parameter_dumps.append(pin(dump))
    chain = deepcopy(read(ROOT / 'configs/reproducibility/fm_rcgan_random_order_chain.2026-10-11_v1.json'))
    chain.update(job_id=job['id'], failed_terminal_receipt=str(terminal.relative_to(ROOT)),
        completed_original_epochs=68, retry_admission_commit_headroom_bytes=40 * 1024 ** 3, output_directory=RESULT + '/random_chain',
        artifact_root='F:/aicoding/IIA_Data/experiments/rcgan_seed1106_full68_random_chain_v1',
        state_root='experiments/runs/fm_rcgan_seed1106_full68_random_chain_v1',
        source_receipts=common + random_states,
        boundary='All68 original full RNG/order epochs; preserve original500 budget, five seeds, and seven missing elapsed times. No native state or completion claim before actual proof.')
    write(ROOT / CHAIN, chain)
    state = deepcopy(read(ROOT / 'configs/reproducibility/fm_rcgan_checkpoint_resume_state.2026-10-11_v1.json'))
    state.update(random_chain_configuration=CHAIN, retry_admission_commit_headroom_bytes=40 * 1024 ** 3, output_directory=RESULT + '/checkpoint_state',
        state_root='experiments/runs/fm_rcgan_seed1106_full68_checkpoint_state_v1',
        source_receipts=common + [pin(ROOT / CHAIN)] + checkpoints + parameter_dumps,
        boundary='Restore all68 model/optimizer checkpoints and61 persisted losses bitwise; recompute seven missing losses, preserve missing times, replay all4984 latest samples and exact native next update. No training until both native proofs pass.')
    write(ROOT / STATE, state)
    resume = dict(configuration_path=RESUME, checkpoint_state_configuration=STATE,
        canonical_job=job, recovery_job=dict(id=job['id'] + '__state_recovery_v1', seed=job['seed'],
            output_directory='F:/aicoding/IIA_Data/experiments/rcgan_seed1106_full432_state_recovery_v1/jobs/seed1106'),
        full_original_epochs=500, verified_completed_epochs=68, remaining_full_epochs=432,
        retry_admission_commit_headroom_bytes=40 * 1024 ** 3,
        python=queue['python'], state_root='experiments/runs/fm_rcgan_seed1106_full432_state_recovery_v1',
        resource_lock=queue['resource_lock'], settings=deepcopy(queue['settings']),
        original_scope=chain['original_scope'], source_receipts=common + [pin(ROOT / CHAIN), pin(ROOT / STATE)],
        boundary='Prospective exact-state full432 continuation after both complete native proofs and stricter40GiB admission. Original500 epochs/all500 evaluators/five seeds/96929 parameters and guards unchanged. Preserve resource failure. No author-equivalence or benchmark claim.')
    write(ROOT / RESUME, resume)
    dispatch = dict(controller_python=chain['controller_python'],
        state_root='experiments/runs/fm_rcgan_seed1106_full68_recovery_dispatch_v1',
        resume_configuration=RESUME, original_scope=chain['original_scope'],
        stages=[dict(name='random_order_chain', module='scripts.flow_matching.audit_rcgan_seed1106_chain_v1',
                     configuration=CHAIN, proof_name='random_order_chain_audit.json', receipt_key='random_order_chain_passed'),
                dict(name='checkpoint_state', module='scripts.flow_matching.audit_rcgan_seed1106_state_v1',
                     configuration=STATE, proof_name='checkpoint_resume_state_audit.json', receipt_key='checkpoint_resume_state_passed')],
        source_receipts=common + [pin(ROOT / CHAIN), pin(ROOT / STATE), pin(ROOT / RESUME)])
    write(ROOT / DISPATCH, dispatch)
    from scripts.flow_matching.run_rcgan_seed1106_recovery_dispatch_v1 import bind
    bind()['verify_dispatch_registration'](dispatch)
    result = ROOT / RESULT
    result.mkdir(parents=True)
    write(result / 'actual_resource_abort_and_registered_full_recovery.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job=job,
        stopped_worker=dict(pid=12384, create_time=1791863687.9790325),
        actual_terminal_receipt=receipt, terminal_receipt_pin=pin(terminal),
        actual_commit_available_bytes=receipt["abort_snapshot"]["commit_available_bytes"],
        original_emergency_commit_headroom_bytes=queue["settings"]["emergency_commit_headroom_bytes"],
        strictly_stronger_retry_admission_commit_headroom_bytes=40 * 1024 ** 3,
        identical_retry_performed=False, frozen_author_environment_changed=False,
        complete_saved_checkpoints=68, complete_saved_evaluators=68, persisted_trace_epochs=61,
        missing_original_elapsed_times=list(range(61, 68)), remaining_original_epochs=432,
        native_state_recovery_proved=False, full500_result_verified=False,
        benchmark_performance=False, additional_training_seed_slots=0,
        configuration_receipts=[pin(ROOT / p) for p in (CHAIN, STATE, RESUME, DISPATCH)],
        original_scope=chain['original_scope'], all_original_experiments_complete=False))
    print(dict(dispatch_configuration=DISPATCH, checkpoints=68, remaining_epochs=432, registered_only=True))


if __name__ == '__main__':
    register()
