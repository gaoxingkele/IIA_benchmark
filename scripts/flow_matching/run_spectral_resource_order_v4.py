"""Dispatch lower-admission independent full jobs first without narrowing the queue."""
import argparse
import ast
import copy
import inspect
import json
import sys

from scripts.flow_matching import run_spectral_resource_admission_v3 as previous


def scheduler_jobs(queue):
    jobs = queue['jobs']
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Distinct complete canonical experiment slots required')
    if any(job['model'] not in ('spectral_flow', 'diffusion_ts') for job in jobs):
        raise ValueError('Only registered independent SpectralFlow/DiffusionTS slots supported')
    settings = queue['settings']
    return sorted(jobs, key=lambda job: previous.admission_settings(
        queue, job, settings)['minimum_commit_headroom_bytes'])


def ordered_main_ast(source):
    tree = previous.selected_main_ast(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'job'
        and ast.dump(node.iter) == ast.dump(ast.parse("queue['jobs']", mode='eval').body)]
    if len(loops) != 1:
        raise ValueError('One complete original job dispatch loop required')
    loops[0].iter = ast.parse('scheduler_jobs(queue)', mode='eval').body
    return ast.fix_missing_locations(tree)


def verify_registration(config):
    queue = previous.verify_registration(config)
    if config['dispatch_policy'] != 'stable_ascending_original_admission_all70_jobs':
        raise ValueError('Frozen full-registry dispatch policy required')
    ordered = scheduler_jobs(queue)
    if len(ordered) != 70 or {job['id'] for job in ordered} != {job['id'] for job in queue['jobs']}:
        raise ValueError('Retain every original canonical full job')
    return queue


def transition(config, path):
    # Reuse the entire exact-PID/time/command/idle/child/lock handoff, with only
    # the successor module changed. No active native model may be interrupted.
    tree = ast.parse(inspect.getsource(previous.transition))
    literals = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
        and node.value == 'scripts.flow_matching.run_spectral_resource_admission_v3']
    if len(literals) != 1:
        raise ValueError('Exact one original successor command binding required')
    literals[0].value = 'scripts.flow_matching.run_spectral_resource_order_v4'
    namespace = dict(previous.__dict__, verify_registration=verify_registration)
    exec(compile(ast.fix_missing_locations(tree), '<unchanged idle handoff with v4 successor>', 'exec'), namespace)
    return namespace['transition'](config, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--transition-idle', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    config = previous.read(previous.ROOT / args.config)
    queue = verify_registration(config)
    if args.verify_only:
        ordered = scheduler_jobs(queue)
        print(json.dumps(dict(full_original_jobs=len(ordered), full_original_cohorts=14,
            original_queue_sha256=previous.sha(previous.ROOT / config['queue']),
            first_full_job=ordered[0]['id'], dispatch_policy=config['dispatch_policy'],
            original_model_sampling_evaluator_budgets_and_guards_unchanged=True,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules)))
        return
    if args.transition_idle:
        return transition(config, args.config)
    namespace = dict(previous.original.__dict__, admission_settings=previous.admission_settings,
        scheduler_jobs=scheduler_jobs, registered_settings=copy.deepcopy(queue['settings']))
    exec(compile(ordered_main_ast(inspect.getsource(previous.original.main)),
        '<original full native queue with stable resource ordering>', 'exec'), namespace)
    sys.argv = ['original-spectral-queue', '--queue', config['queue'], '--preflight', config['preflight']]
    namespace['main']()


if __name__ == '__main__':
    main()
