"""Full point/model/post review of the three new same-seed CrossAD retries.

All139 canonical main/release/ablation tasks remain. Existing live queues and
their failures are preserved. CrossAD literature/data bindings come from the
original frozen queue; no training, new seeds or reduced replay is introduced.
"""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import psutil

from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as original
from scripts.flow_matching import run_crossad_recovery_completion_handoff_v1 as recovery

ROOT, read, digest, write = original.ROOT, original.read, original.digest, original.write
EXPECTED = {'crossad_author__fresh_train__' + d + '__seed1103' for d in ('gecco', 'psm', 'smap')}
POST_MODULE = 'scripts.flow_matching.replay_crossad_post_metric_keywords_v2'


def exact_science(canonical, job, original_queue, native_queue):
    expected = copy.deepcopy(original_queue)
    selected = next(j for j in expected['jobs'] if j['id'] == canonical['id'])
    selected['output_directory'] = job['output_directory']
    if (canonical['id'] != job['id'] or canonical['output_directory'] == job['output_directory']
            or native_queue != expected or selected != job):
        raise ValueError('All139 scientific fields must remain; only one output may change')


def verify(config, check_live=False):
    # This controller waits for existing producers; their canonical training
    # process may be live. Only successful terminal receipts admit any review.
    for pin in config['source_receipts']:
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete retry review binding changed: ' + pin['path'])
    base = read(config['original_completion_configuration'])
    jobs, template, preserved = original.verify(base)
    if (config['required_original_jobs'] != 139 or config['original_scope'] != original.SCOPE
            or {r['canonical_id'] for r in config['retry_routes']} != EXPECTED
            or len(config['retry_routes']) != 3
            or any(config[k] != base[k] for k in original.GUARDS)
            or config['score_tolerance'] != original.SCORE_TOLERANCE
            or config['metric_tolerance'] != original.METRIC_TOLERANCE):
        raise ValueError('Full139 scope, three actual retries and original guards/tolerances required')
    from scripts.flow_matching import crossad_gecco1103_full_retry_v1 as gecco
    from scripts.flow_matching import crossad_psm1103_full_retry_v1 as psm
    from scripts.flow_matching import crossad_smap1103_full_retry_v1 as smap
    methods = {'GECCO': gecco, 'PSM': psm, 'SMAP': smap}
    parent = read(base['original_queue']); routes = {}
    for route in config['retry_routes']:
        retry_config = read(route['retry_configuration'])
        queue, job, _ = methods[jobs[route['canonical_id']]['dataset']].verify(retry_config, check_live=check_live)
        if route['native_queue'] != retry_config['native_queue']:
            raise ValueError('Exact registered fresh retry queue required')
        exact_science(jobs[route['canonical_id']], job, parent, queue)
        if (ROOT / jobs[route['canonical_id']]['output_directory'] / 'result.json').exists():
            raise ValueError('Inspect unexpected original completion before a duplicate retry review')
        routes[route['canonical_id']] = (route, job)
    return base, jobs, template, routes


def effective_stage_settings(config, base, route):
    effective = copy.deepcopy(base)
    for key in ('source_receipts', 'proof_root', 'child_state_root', 'configuration_root', 'artifact_root'):
        effective[key] = copy.deepcopy(config[key])
    effective['original_queue'] = route['native_queue']
    return effective


def prepare_point(config, path, base, template, route, job):
    effective = effective_stage_settings(config, base, route)
    point_path, point = original.point_configuration(effective, path, job)
    native_template = copy.deepcopy(template); native_template['queue'] = route['native_queue']
    return effective, point_path, point, native_template


def prepare_post(config, route, job, point, native_path, native):
    post = {k: copy.deepcopy(point[k]) for k in (*original.GUARDS, 'original_scope', 'job_id')}
    post.update(phase='post', queue=route['native_queue'], native_configuration=native_path,
        output_directory=config['proof_root'] + '/' + job['id'] + '/post',
        state_root=config['child_state_root'] + '/' + job['id'] + '/post',
        source_receipts=copy.deepcopy(native['source_receipts']) + [
            dict(path=native_path, sha256=digest(native_path)),
            dict(path=native['state_root'] + '/receipt.json', sha256=digest(native['state_root'] + '/receipt.json')),
            dict(path=str(Path(native['artifact_directory']) / 'full_native_model_replay.json'),
                 sha256=digest(ROOT / native['artifact_directory'] / 'full_native_model_replay.json'))])
    path = (Path(config['configuration_root']) / (job['id'] + '_post.json')).as_posix()
    original.exclusive_configuration(path, post)
    return path


def verify_completed_retry(config, route, job):
    stage_path = (Path(config['configuration_root']) / (job['id'] + '_post.json')).as_posix()
    certificate = recovery.completed_review(stage_path, job, config)
    parent = read(read(config['original_completion_configuration'])['original_queue'])
    canonical = next(j for j in parent['jobs'] if j['id'] == route['canonical_id'])
    exact_science(canonical, job, parent, read(route['native_queue']))
    return dict(certificate, canonical_id=canonical['id'], recovery_job_id=job['id'],
        native_queue=route['native_queue'], additional_training_seed_slots=0,
        completion_origin='actual_same_seed_full_retry_native_and_independent_post_review')


def collect_existing(config, jobs, routes, covered):
    base = read(config['original_completion_configuration'])
    sources = [(ROOT / base['state_root'] / 'completed', False),
               (ROOT / read(config['existing_recovery_completion_configuration'])['state_root'] / 'completed', True)]
    old_recovery = read(config['existing_recovery_completion_configuration'])
    old_routes = {r['canonical_id']: r for r in old_recovery['recovery_routes']}
    for directory, is_recovery in sources:
        for path in directory.glob('*.json'):
            item = read(path); identity = item['canonical_id'] if is_recovery else item['job_id']
            if identity in covered or identity in routes:
                continue
            if identity not in jobs:
                raise ValueError('Unknown original job cannot enter139-case evidence union')
            job = jobs[identity]
            if is_recovery:
                route = old_routes[identity]
                job = recovery.exact_route(route, job, read(route['native_queue']), read(base['original_queue']))
            proof = recovery.completed_review(item['post_review_configuration'], job, config)
            covered[identity] = dict(proof, canonical_id=identity, recovery_job_id=job['id'],
                additional_training_seed_slots=0, completion_origin='existing_full_native_and_post_review')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True); parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(args.config)
    base, jobs, template, routes = verify(config)
    if args.probe_only:
        print(dict(required_original_jobs=139, selected_retry_reviews=3, full_native_and_post_review_required=True,
            score_tolerance=config['score_tolerance'], metric_tolerance=config['metric_tolerance'],
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
            numerical_replay_executed=False, all_original_experiments_complete=False)); return
    from scripts.flow_matching.run_crossad_post_fixed_controllers_v2 import cached_resource_api, live_stage_jobs
    api = cached_resource_api(); root = ROOT / config['state_root']; root.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](root / 'controller.lock')
    if owner is None:
        raise RuntimeError('This exact three-retry full review already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=digest(args.config), all_original_experiments_complete=False)
    previous = read(root / 'status.json') if (root / 'status.json').exists() else {}
    if previous and previous['configuration_sha256'] != state['configuration_sha256']:
        raise ValueError('Preserve the prior immutable review attempt')
    covered, failed = {}, dict(previous.get('failed_preserved', {}))
    for path in (root / 'completed').glob('*.json'):
        evidence = read(path); identity = evidence['canonical_id']
        if identity not in routes:
            raise ValueError('Unknown retry completion cannot enter the original139 roster')
        route, job = routes[identity]
        if evidence != verify_completed_retry(config, route, job):
            raise ValueError('Preserved complete retry certificate changed')
        covered[identity] = evidence

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(root / 'status.json', state)

    try:
        while len(covered) < 139:
            collect_existing(config, jobs, routes, covered)
            for identity, (route, job) in routes.items():
                if identity in covered or identity in failed or identity in live_stage_jobs():
                    continue
                terminal = ROOT / read(route['retry_configuration'])['state_root'] / 'receipt.json'
                if not terminal.exists():
                    continue
                receipt = read(terminal)
                if receipt.get('full_native_pipeline_verified') is not True or receipt.get('exit_code') != 0:
                    failed[identity] = dict(error='Native fresh retry did not complete; no review or automatic repeat', native_terminal=str(terminal)); continue
                if receipt['resource_usage']['resource_guard_aborted']:
                    raise ValueError('Resource-aborted native retry cannot enter full review')
                try:
                    if digest(args.config) != state['configuration_sha256']:
                        raise ValueError('Live three-retry review configuration changed')
                    effective, point_path, point, replay_template = prepare_point(config, args.config, base, template, route, job)
                    recovery.run_stage(config, point_path, 'scripts.flow_matching.run_crossad_completion_replay_queue_v1', 'point', identity, update)
                    native_path, native = original.replay_configuration(effective, replay_template, job, point_path, point)
                    recovery.run_stage(config, native_path, 'scripts.flow_matching.replay_crossad_registered_model_v3', 'native', identity, update)
                    original.native_terminal(native, job)
                    post_path = prepare_post(config, route, job, point, native_path, native)
                    recovery.run_stage(config, post_path, POST_MODULE, 'post', identity, update)
                    evidence = verify_completed_retry(config, route, job)
                    original.exclusive_json(root / 'completed' / (identity + '.json'), evidence)
                    covered[identity] = evidence
                except Exception as error:
                    failed[identity] = dict(error_type=type(error).__name__, error=str(error), failed_attempt_preserved=True)
                update(dict(active_pid=None, active_job=None))
            union = dict(required_original_jobs=139, verified_original_ids=sorted(covered),
                verified_original_jobs=len(covered), pending_original_jobs=139-len(covered),
                verified_new_retry_ids=sorted(EXPECTED & set(covered)), failed_preserved=failed,
                all139_original_full_native_and_independent_executions_verified=len(covered)==139,
                all_original_experiments_complete=False, additional_training_seed_slots=0)
            write(root / 'verified_full139_union.json', dict(union, evidence=covered))
            update(dict(union, state='waiting_actual_three_native_retries_and_full139_post_certificates'))
            time.sleep(config['poll_seconds'])
    finally:
        owner.close()


if __name__ == '__main__':
    main()
