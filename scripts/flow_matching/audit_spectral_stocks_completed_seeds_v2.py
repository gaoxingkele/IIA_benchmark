"""Run unchanged native full-artifact audits for newly completed original seeds."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def registration(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Registered full native artifact-audit source changed')
    queue = read(ROOT / config['original_queue'])
    jobs = {j['id']: j for j in queue['jobs']}
    if (config['original_required_seeds'] != [42, 43, 44, 45, 46]
            or config['newly_completed_seeds'] != [43, 44, 45]
            or len(config['audits']) != 3 or len({r['id'] for r in config['audits']}) != 3):
        raise ValueError('All three newly completed original seeds and the five-seed obligation are required')
    for entry, seed in zip(config['audits'], config['newly_completed_seeds']):
        job = jobs[entry['id']]
        audit = read(ROOT / entry['configuration'])
        if (job['seed'] != seed or job['model'] != 'diffusion_ts' or job['dataset'] != 'stocks'
                or job['data_track'] != 'released_author_data' or audit['job_id'] != job['id']
                or audit['queue'] != config['original_queue'] or job['training_updates'] != 10000
                or job['gradient_accumulate_every'] != 2 or job['generation_batch_size'] != 2001
                or job['generation_field_chunk_size'] is not None):
            raise ValueError('Keep every original complete job, model, batch and full budget')
    for key in ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes']:
        if config['settings'][key] != queue['settings'][key]:
            raise ValueError('Preserve the original Spectral native memory safeguards')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(ROOT / args.config); registration(config)
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base / 'controller.lock')
    if lane is None: raise RuntimeError('This full native artifact-audit controller is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(values): state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)
    records = []
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            for entry in config['audits']:
                audit = read(ROOT / entry['configuration']); target = ROOT / audit['output_directory']
                if target.exists(): raise FileExistsError('Never reuse an original full-artifact audit attempt')
                command = [config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.audit_spectral_completed_generation_v1', '--config', entry['configuration']]
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
                if passed: record.update(proof_path=str(proof), proof_sha256=sha(proof))
                records.append(record); write(base / 'progress.json', dict(required_full_native_audits=3, records=records))
                if not passed:
                    write(base / 'receipt.json', dict(passed=False, records=records, all_original_experiments_complete=False))
                    update(dict(state='failed_preserved', active_pid=None)); raise RuntimeError('Preserve failed complete native artifact audit')
        registration(config)
        write(base / 'receipt.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
            full_native_artifact_audits_completed=3, original_required_seeds=config['original_required_seeds'],
            audited_new_original_seeds=config['newly_completed_seeds'], records=records,
            unchanged_native_audit_source=True, additional_training_seed_slots=0,
            author_equivalence_certified=False, all_original_experiments_complete=False))
        update(dict(state='completed', active_pid=None))
        print(dict(full_native_artifact_audits_completed=3, required_original_training_seeds=5))
    finally: lane.close()


if __name__ == '__main__': main()
