"""Dispatch one untouched full NODE10 recovery beside the long original queue."""
from contextlib import contextmanager
import sys

from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as base
from scripts.flow_matching.run_cfm_ts_receipt_runtime_v3 import write_with_parents
from scripts.flow_matching.timegan_native_common_v1 import read, sha

original_resource_api = base.resource_api


def verify_runtime(runtime):
    if len(runtime['controllers']['recovery']['selected_job_ids']) != 1:
        raise ValueError('Only the registered independent complete NODE10 recovery slot may use this lane')
    selected = runtime['controllers']['recovery']
    if sha(base.ROOT / selected['queue']) != selected['queue_sha256']:
        raise ValueError('Untouched full recovery queue required')
    queue = read(base.ROOT / selected['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == selected['selected_job_ids'][0])
    if (job['recovery_original_job_id'] != 'cfm_ts_original__appendix_d__node_e10__pendulum__seed1104'
            or job['epochs'] != 10 or job['expected_optimizer_updates'] != 10
            or job['test_points_per_trajectory'] != 500 or job['training_items'] != 1
            or job['test_trajectories'] != 1 or job['model_overrides']['grid_points'] != 160):
        raise ValueError('Complete exact original data/network/grid/budget required')
    policy = runtime['parallel_admission_policy']
    old = queue['resource_policy']
    for key in ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes']:
        if policy[key] < old[key]:
            raise ValueError('Do not weaken existing global resource protection')
    if (not 0 < policy['maximum_worker_private_bytes'] <= 10*1024**3
            or not 0 < policy['maximum_worker_rss_bytes'] <= 8*1024**3):
        raise ValueError('Independent worker must remain explicitly capped')
    if runtime['family_lock'] == 'experiments/runs/fm_cfm_ts_original_cpu.lock':
        raise ValueError('Use the independently registered lane without stealing the original family slot')
    return queue, job


def verify_controller(arguments):
    if '--controller' not in arguments or arguments[arguments.index('--controller')+1] != 'recovery':
        raise ValueError('The independent lane only accepts an explicit recovery controller')


def main():
    verify_controller(sys.argv)
    runtime = read(base.ROOT / sys.argv[sys.argv.index('--runtime-config')+1])
    queue, job = verify_runtime(runtime)
    policy = runtime['parallel_admission_policy']

    def resource_api():
        api = original_resource_api()
        original_slot = api['resource_slot']

        @contextmanager
        def independent_slot(path, settings, update):
            if settings != queue['resource_policy']:
                raise ValueError('Only unchanged original scientific worker policy may be bound')
            with original_slot(path, policy, update) as snapshot:
                yield snapshot

        def independent_wait(child, settings, update):
            if settings != queue['resource_policy']:
                raise ValueError('Complete original worker guard binding changed')
            from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
            def training_update(values):
                # This guard also serves checkpoint replay; here the worker trains.
                update(dict(values, state='running'))
            return capped_wait(child, dict(settings=policy), api, training_update)

        api['resource_slot'] = independent_slot
        api['guarded_wait'] = independent_wait
        return api

    base.resource_api = resource_api
    base.write = write_with_parents
    base.main()


if __name__ == '__main__':
    main()
