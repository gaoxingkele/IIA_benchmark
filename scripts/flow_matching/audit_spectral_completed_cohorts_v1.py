"""Guarded full native artifact reviews for registered completed cohorts."""
import argparse
from datetime import datetime, timezone
import os
import subprocess

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def validate_selection(config, queue, audits):
    jobs = {job['id']: job for job in queue['jobs']}
    if len(jobs) != 70 or config['original_required_seeds'] != [42, 43, 44, 45, 46]:
        raise ValueError('Preserve all70 jobs and original five-seed obligations')
    required = {job['id'] for job in jobs.values()
        if job['model'] == 'diffusion_ts' and job['data_track'] == 'released_author_data'
        and job['dataset'] in config['completed_cohort_seeds']
        and job['seed'] in config['completed_cohort_seeds'][job['dataset']]}
    actual = [entry['id'] for entry in config['audits']]
    if not required or len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError('Every registered complete model requires a unique full audit')
    for entry in config['audits']:
        job = jobs[entry['id']]
        audit = audits[entry['id']]
        if (audit['queue'] != config['original_queue'] or audit['job_id'] != job['id']
                or audit['terminal_receipt'] != queue['state_root'] + '/receipts/' + job['id'] + '.json'
                or job['generation_batch_size'] != 2001 or job['generation_field_chunk_size'] is not None):
            raise ValueError('Original full job identity, result and sampling recipe required')
    keys = ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes']
    if any(config['settings'][key] != queue['settings'][key] for key in keys):
        raise ValueError('Preserve original Spectral memory safeguards')


def registration(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen full artifact review input changed: ' + pin['path'])
    queue = read(ROOT / config['original_queue'])
    audits = {entry['id']: read(ROOT / entry['configuration']) for entry in config['audits']}
    validate_selection(config, queue, audits)
    for entry in config['audits']:
        receipt = read(ROOT / audits[entry['id']]['terminal_receipt'])
        if receipt['exitcode'] != 0 or receipt['resource_guard_aborted'] or not receipt['full_result_validated']:
            raise ValueError('Actual terminal full success required before artifact review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    registration(config)
    api = resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base / 'controller.lock')
    if lane is None:
        raise RuntimeError('This full native review controller is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)

    records = []
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            for entry in config['audits']:
                audit = read(ROOT / entry['configuration'])
                target = ROOT / audit['output_directory']
                if target.exists():
                    raise FileExistsError('Preserve every previous full audit attempt')
                command = [config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.audit_spectral_completed_generation_v2', '--config', entry['configuration']]
                with (base / (entry['id'] + '.stdout.log')).open('xb') as out, (base / (entry['id'] + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                            PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), creationflags=subprocess.CREATE_NO_WINDOW)
                    handle = psutil.Process(child.pid)
                    update(dict(active_pid=child.pid, active_create_time=handle.create_time(),
                        actual_command=handle.cmdline(), active_job=entry['id']))
                    code, usage = capped_wait(child, config, api, update)
                proof = target / 'independent_full_native_artifact_audit.json'
                passed = bool(code == 0 and not usage['resource_guard_aborted'] and proof.exists()
                    and read(proof)['passed'] and not read(proof)['cuda_context_initialized'])
                record = dict(id=entry['id'], configuration=entry['configuration'],
                    configuration_sha256=sha(ROOT / entry['configuration']), native_exit_code=code,
                    full_native_artifact_audit_passed=passed, **usage)
                if passed:
                    record.update(proof_path=str(proof), proof_sha256=sha(proof))
                records.append(record)
                write(base / 'progress.json', dict(required_full_native_audits=len(config['audits']), records=records))
                if not passed:
                    write(base / 'receipt.json', dict(passed=False, records=records, all_original_experiments_complete=False))
                    update(dict(state='failed_preserved', active_pid=None))
                    raise RuntimeError('Preserve failed full artifact review')
        registration(config)
        write(base / 'receipt.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
            full_native_artifact_audits_completed=len(records), original_required_seeds=config['original_required_seeds'],
            audited_cohort_seeds=config['completed_cohort_seeds'], records=records,
            unchanged_v1_tensor_array_and_metric_checks=True, additional_training_seed_slots=0,
            author_equivalence_certified=False, all_original_experiments_complete=False))
        update(dict(state='completed', active_pid=None))
        print(dict(full_native_artifact_audits_completed=len(records)))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
