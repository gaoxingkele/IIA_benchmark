"""Preserve a terminal resource incident and queue a full native state audit."""
import copy
import json
from pathlib import Path
import pickletools
import shutil
import zipfile

from scripts.flow_matching import audit_full813_image_checkpoint_CPU_v1 as audit
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive

RUNTIME = 'configs/runtime/fm_full813_image_checkpoint_CPU_audit.v1.json'
ARCHIVE = 'configs/reproducibility/fm_full813_checkpoint_immutable_archive.2026-10-13_v1.json'
EVIDENCE = 'projects/flow_matching_research/results/2026-10-13/full813_resource_incident_and_CPU_audit_v1'


def counts(path):
    """Read primitive checkpoint counters without unpickling or importing Torch."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 64 * 1024 ** 2:
            raise ValueError('One bounded original PyTorch metadata stream required')
        raw = archive.read(names[0])
    found = {}; pending = None
    for op, value, _ in pickletools.genops(raw):
        if pending:
            if op.name not in ('BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4'):
                if op.name in ('BINPUT','LONG_BINPUT','MEMOIZE'): continue
                raise ValueError('Expected an original primitive integer counter')
            found[pending] = value; pending = None
            if len(found) == 2: return found
        elif isinstance(value,str) and value in ('completed_epochs','global_steps'):
            pending = value
    raise ValueError('Both actual checkpoint epoch and optimizer counters required')


def require_terminal(state):
    import psutil
    if state['state'] != 'failed_full_training_attempt_preserved' or state['active_pid'] is not None:
        raise ValueError('Preserved actual terminal full-training incident required')
    for pid, created in ((state['pid'],state['create_time']),
                         (state['last_terminal']['worker_pid'],state['last_terminal']['worker_create_time'])):
        try:
            if psutil.Process(pid).create_time() == created:
                raise ValueError('Original coordinator and scientific worker must already be terminal')
        except psutil.NoSuchProcess: pass
    usage = state['last_terminal']['resource_usage']
    if (state['last_terminal']['returncode'] == 0 or not usage['resource_guard_aborted']
            or usage['abort_resource_snapshot']['commit_available_bytes'] >= 16 * 1024 ** 3):
        raise ValueError('This registration requires the actual below16GiB commit incident')


def main():
    root = audit.ROOT
    previous = 'configs/reproducibility/fm_full_image_training_verified401_resume.2026-10-12_v4.json'
    original = audit.read(root/previous)
    state_path = root/original['state_root']/'status.json'
    state = audit.read(state_path); require_terminal(state)
    job_path = original['ordered_original_jobs'][0]; job = audit.read(root/job_path)
    output = root/job['output_directory']; progress = audit.read(output/'progress.json')
    if (progress['completed_epochs'] != 813 or progress['global_steps'] != 92682
            or progress['registered_epochs'] != 2000):
        raise ValueError('Actual complete813 epoch progress and all92682 updates required')
    source = output/'resume.pt'; observed = counts(source)
    if observed != dict(completed_epochs=813,global_steps=92682):
        raise ValueError('Latest saved checkpoint primitive counters must match actual813 progress')
    prior_archive = audit.read(root/'configs/reproducibility/fm_full500_checkpoint_immutable_archive.2026-10-12_v2.json')
    destination = Path(prior_archive['immutable_archive']['path']).parent/'epoch_0813_full_state.pt'
    if destination.exists() or (root/RUNTIME).exists() or (root/ARCHIVE).exists():
        raise FileExistsError('Preserve all existing registrations and immutable archives')
    source_receipt = dict(path=str(source),sha256=audit.sha(source),bytes=source.stat().st_size)
    if shutil.disk_usage(destination.parent).free < 2*source_receipt['bytes']+2*1024**3:
        raise OSError('Complete checkpoint archive capacity required')
    with source.open('rb') as inp, destination.open('xb') as out:
        shutil.copyfileobj(inp,out,1024**2)
        out.flush()
        import os
        os.fsync(out.fileno())
    saved = dict(path=str(destination),sha256=audit.sha(destination),bytes=destination.stat().st_size)
    if saved['sha256'] != source_receipt['sha256'] or saved['bytes'] != source_receipt['bytes'] or audit.sha(source) != saved['sha256']:
        raise ValueError('Preserve and reject any nonidentical full-state archive')
    archive = dict(source_checkpoint=source_receipt,immutable_archive=saved,
        required_original_epochs=2000,expected_completed_original_epochs=813,
        original_scope=original['original_scope'],original_checkpoint_untouched=True,
        training_worker_already_terminal=True,new_optimizer_updates=0,
        full_native_state_restore_verified=False,all_original_experiments_complete=False)
    exclusive(root/ARCHIVE,archive)
    config = copy.deepcopy(audit.read(root/'configs/runtime/fm_full500_image_checkpoint_CPU_audit.v2.json'))
    config.update(completed_epochs=813,checkpoint_receipt=saved,source_checkpoint_receipt=source_receipt,
        immutable_archive_configuration=ARCHIVE,state_root='experiments/runs/fm_full813_image_checkpoint_CPU_audit_v1',
        output_root=EVIDENCE+'/native_CPU_state')
    paths = [previous,ARCHIVE,'configs/runtime/fm_full500_image_checkpoint_CPU_audit.v2.json',
        'scripts/flow_matching/audit_full813_image_checkpoint_CPU_v1.py',
        'scripts/flow_matching/run_full813_image_checkpoint_CPU_audit_v1.py',
        'scripts/flow_matching/register_full813_checkpoint_CPU_audit_v1.py',
        'tests/test_fm_full813_image_checkpoint_CPU_v1.py']
    config['source_receipts'] += [dict(path=p,sha256=audit.sha(root/p)) for p in paths]
    config['boundary'] = ('All813 complete original epoch receipts, full45M model and StableAdamW variables and '
        'CPU/Python/NumPy/order RNG must restore bitwise. Saved CUDA bytes retained without CUDA initialization. '
        'Original CPU gates and full20FM+5WT/all2000 budgets retained;1187 remaining epochs require a separate '
        'successful full native CPU proof followed by full original GPU state restoration. No training or performance credit.')
    audit.verify(config); exclusive(root/RUNTIME,config)
    terminal = Path(progress['attempt_directory'])/'terminal.json'
    exclusive(root/EVIDENCE/'actual_resource_abort_and_full813_archive_registration.json',dict(
        actual_terminal=audit.read(terminal),actual_progress=progress,
        terminal_receipt=dict(path=str(terminal),sha256=audit.sha(terminal)),
        stderr_receipt=dict(path=str(terminal.parent/'stderr.log'),sha256=audit.sha(terminal.parent/'stderr.log')),
        immutable_archive=archive,CPU_audit_configuration=RUNTIME,
        native_CPU_audit_passed=False,full_GPU_state_restore_passed=False,
        required_original_image_jobs=25,required_epochs_per_job=2000,remaining_first_job_epochs=1187,
        original_scope=original['original_scope'],all_original_experiments_complete=False))
    print(json.dumps(dict(archived_original_epochs=813,remaining_original_epochs=1187,
        full_native_CPU_state_verification_pending=True,original_image_jobs=25)))


if __name__ == '__main__': main()
