"""Bind both existing complete native capacity proofs to unchanged scientific scope."""
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha


def verify_capacity_reuse(queue):
    old = read(ROOT / queue['preserved_queue'])
    if sha(ROOT / queue['preserved_queue']) != queue['preserved_queue_sha256']:
        raise ValueError('Preserved native queue changed')
    for key in ('source_root', 'python', 'environment_lock', 'prepared_root', 'loader_registration',
                'loader_proof', 'cauchy_correction_config', 'cpu_threads', 'profiles', 'seeds',
                'expected_full_training_optimizer_updates', 'resource_lock'):
        if queue[key] != old[key]:
            raise ValueError('Original native scientific scope changed: ' + key)
    old_jobs = {j['id']: j for j in old['jobs']}
    if set(old_jobs) != {j['id'] for j in queue['jobs']}:
        raise ValueError('Recovery must keep all ten canonical jobs, no additional seeds')
    for job in queue['jobs']:
        for key in ('id', 'profile', 'seed', 'model_config'):
            if job[key] != old_jobs[job['id']][key]:
                raise ValueError('Full original model/seed configuration changed')
        if job['output_directory'] == old_jobs[job['id']]['output_directory'] or job['artifact_directory'] == old_jobs[job['id']]['artifact_directory']:
            raise ValueError('Every original training attempt must be preserved')
    settings, previous = queue['settings'], old['settings']
    if settings['minimum_commit_headroom_bytes'] < 52 * 1024 ** 3:
        raise ValueError('At least52GiB admission required after full-training peak')
    for key in previous:
        if key != 'minimum_commit_headroom_bytes' and settings[key] != previous[key]:
            raise ValueError('Emergency guards and other original resource settings must stay unchanged')
    for profile in queue['profiles']:
        record = next(x for x in queue['reused_capacity_proofs'] if x['profile'] == profile)
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Original full patient capacity evidence changed')
        proof = read(ROOT / record['path'])
        batch, updates, test_batches = (64, 100, 25) if profile == 'interpolation' else (32, 200, 50)
        parameters = 535054 if profile == 'interpolation' else 783310
        if (not proof['passed'] or proof['queue_sha256'] != queue['preserved_queue_sha256'] or
                proof['profile'] != profile or proof['seed'] != 42 or proof['original_batch_size'] != batch or
                proof['actual_optimizer_updates'] != updates or proof['full_test_batches'] != test_batches or
                proof['full_model_parameters'] != parameters or not proof['full_patient_capacity_passed'] or
                not proof['native_evaluation_interface_passed'] or not proof['full_original_first_batch_retained'] or
                not proof['diagnostic_only'] or proof['performance_result']):
            raise ValueError('Complete original GPU patient capacity is not established')
    return True
