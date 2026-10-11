"""Preserve terminal native attempts and register the same full35 scientific jobs."""
import copy
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.run_crnngan_native_original_queue_v2 import verify_registration


def recovery_queue(previous):
    queue = copy.deepcopy(previous)
    queue['id'] = 'crnngan_classical_released_full_native_memory_recovery_v3'
    queue['state_root'] = 'experiments/runs/fm_crnngan_native_classical_released_v3'
    queue['prior_attempt_queue'] = 'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json'
    queue['settings'] = dict(previous['settings'], minimum_available_memory_bytes=32*1024**3,
        minimum_commit_headroom_bytes=40*1024**3, maximum_worker_rss_bytes=22*1024**3,
        maximum_worker_private_bytes=24*1024**3)
    queue['preserved_prior_attempts'] = []
    for old, job in zip(previous['jobs'], queue['jobs']):
        queue['preserved_prior_attempts'].append(dict(id=old['id'], output_directory=old['output_directory']))
        job['output_directory'] = 'F:/aicoding/IIA_Data/experiments/crnngan_native_classical_released_v3/jobs/' + job['variant'] + '/seed' + str(job['seed'])
    queue['source_receipts'].append(dict(path=queue['prior_attempt_queue'], sha256=sha(ROOT/queue['prior_attempt_queue'])))
    for path in ['scripts/flow_matching/register_crnngan_full_memory_recovery_v3.py',
                 'tests/test_fm_crnngan_full_memory_recovery_v3.py']:
        queue['source_receipts'].append(dict(path=path, sha256=sha(ROOT/path)))
    queue['recovery_boundary'] = ('Full source-classical3014 MIDI/10323257 notes, seven variants/five seeds, original350x2/batch20/curriculum100 and500 epochs retained. '
        'Only resource admission/caps and fresh preserved attempt destinations change. 40GiB commit reserves the24GiB private cap plus original16GiB emergency margin. '
        'The native stateful TensorFlow RNG cannot yet be restored exactly; restart the same canonical seed from initialization, never label partial checkpoints as an exact continuation or select attempts by test scores.')
    verify_scientific_equivalence(previous, queue)
    verify_registration(queue)
    return queue


def verify_scientific_equivalence(previous, queue):
    if len(previous['jobs']) != 35 or len(queue['jobs']) != 35:
        raise ValueError('All35 full canonical jobs required')
    for key in previous:
        if key not in {'id', 'state_root', 'settings', 'jobs', 'source_receipts', 'prior_attempt_queue'} and queue[key] != previous[key]:
            raise ValueError('Scientific queue configuration changed: ' + key)
    for old, new in zip(previous['jobs'], queue['jobs']):
        if {k: v for k, v in old.items() if k != 'output_directory'} != {k: v for k, v in new.items() if k != 'output_directory'}:
            raise ValueError('Original full scientific job changed')
        if new['output_directory'] == old['output_directory']:
            raise ValueError('Preserve the prior attempt in its original directory')
    if queue['settings']['emergency_commit_headroom_bytes'] != previous['settings']['emergency_commit_headroom_bytes']:
        raise ValueError('Original global emergency margin must remain unchanged')
    if queue['settings']['minimum_commit_headroom_bytes'] < queue['settings']['maximum_worker_private_bytes'] + queue['settings']['emergency_commit_headroom_bytes']:
        raise ValueError('Reserve full worker private cap and emergency commit before dispatch')
    if queue['source_receipts'][:len(previous['source_receipts'])] != previous['source_receipts']:
        raise ValueError('Existing frozen source receipts must remain unchanged')


def main():
    import psutil
    previous = read(ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json')
    state = read(ROOT/previous['state_root']/'status.json')
    receipt = read(ROOT/previous['state_root']/'receipts'/(previous['jobs'][0]['id']+'.json'))
    observed = read(ROOT/'projects/flow_matching_research/results/2026-10-11/crnngan_native_classical_execution_v2/verified_runtime_observation.json')
    for handle in observed['verified_processes'].values():
        try:
            process = psutil.Process(handle['pid'])
            if process.create_time() == handle['create_time'] and process.cmdline() == handle['actual_command']:
                raise ValueError('Original controller or worker is still live')
        except psutil.NoSuchProcess:
            pass
    if (state['state'] != 'stopped_for_failed_job_diagnosis' or state['active_pid'] is not None
            or not receipt['resource_guard_aborted'] or receipt['full_native_result_verified']):
        raise ValueError('Authoritative terminal resource receipt and missing original handles required')
    for source in previous['source_receipts']:
        if sha(ROOT/source['path']) != source['sha256']:
            raise ValueError('Preserved full native scientific source changed')
    path = ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v3.json'
    if path.exists():
        raise FileExistsError('Preserve existing recovery registration')
    queue = recovery_queue(previous)
    if any(Path(j['output_directory']).exists() for j in queue['jobs']):
        raise FileExistsError('Every new attempt destination must be fresh')
    write(path, queue)


if __name__ == '__main__':
    main()
