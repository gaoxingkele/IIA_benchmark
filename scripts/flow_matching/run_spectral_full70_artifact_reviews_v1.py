"""Review future complete original generators without duplicating the closed30.

All70 original canonical models and the explicit full-budget fMRI43 recovery
remain distinct. Uses unchanged native tensor/array/evaluator checks; this is
neither generative reinference nor learned-evaluator retraining.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.run_spectral_full70_artifact_reviews_v1'
CONFIG = 'configs/runtime/fm_spectral_full70_artifact_reviews.v1.json'
TEST = 'tests/test_fm_spectral_full70_artifact_reviews_v1.py'
QUEUE = 'configs/experiments/fm_spectral_original_resource_admission_queue.v3.json'
CATALOG = 'projects/flow_matching_research/results/2026-10-16/spectral_after29_PaperSines42_full30_native_model_increment_v1/unique30_full_native_model_catalog.json'


def exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def registration():
    parent = read(ROOT / 'configs/reproducibility/fm_spectral_sines_paper_seed42_full_native_review.2026-10-16_v1.json')
    queue = read(ROOT / QUEUE)
    paths = [MODULE.replace('.', '/') + '.py', TEST, QUEUE, CATALOG, queue['protocol'],
        'scripts/flow_matching/audit_spectral_completed_generation_v1.py',
        'scripts/flow_matching/audit_spectral_completed_generation_v2.py',
        'scripts/flow_matching/run_spectral_original_job.py',
        'src/iia_benchmark/models/fm_spectral_author.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'configs/reproducibility/fm_spectral_sines_paper_seed42_full_native_review.2026-10-16_v1.json',
        'configs/runtime/fm_spectral_fmri43_actual500_full_retry.v1.json',
        'configs/experiments/fm_spectral_fmri43_actual500_full_retry_queue.v1.json']
    return dict(original_queue=QUEUE, required_original_models=70, required_original_cohorts=14,
        original_required_seeds=[42, 43, 44, 45, 46], reviewed_catalog=CATALOG,
        preserved_reviewed_models=30, python=parent['python'], settings=parent['settings'],
        resource_lock=parent['resource_lock'], numerical_threads=2,
        state_root='experiments/runs/fm_spectral_full70_artifact_reviews_v1',
        configuration_root='configs/reproducibility/fm_spectral_full70_artifact_reviews_v1',
        proof_root='projects/flow_matching_research/results/2026-10-16/spectral_full70_automatic_artifact_reviews_v1/models',
        poll_seconds=10, recovery_routes=[dict(canonical_id='spectral_original__diffusion_ts__fmri__released_author_data__seed43',
            execution_queue='configs/experiments/fm_spectral_fmri43_actual500_full_retry_queue.v1.json')],
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in paths],
        original_scope=parent['original_scope'], citation=read(ROOT / queue['protocol'])['citation'],
        boundary='Full70 canonical generator jobs/14 cohorts/5 training seeds. Reuse only existing hash-bound30 full '
            'native artifact reviews; perform unchanged full checks for future complete sources under original '
            '12GiB physical/20GiB commit/12GiB emergency,3GiB RSS/6GiB private policy and shared readonly lock. '
            'Full fMRI43 recovery changes only isolated output/artifact paths, no extra seed. No training/generation '
            'retry or sampling/evaluator change; failures remain preserved, no automatic failed-audit retry. '
            'This is artifact/statistic verification, not generative replay or independently retrained learned evaluators.',
        all_original_experiments_complete=False)


def validate_scope(config, queue, catalog):
    jobs = {j['id']: j for j in queue['jobs']}
    cohorts = Counter((j['model'], j['dataset'], j['data_track']) for j in jobs.values())
    if (len(jobs) != len(queue['jobs']) or len(jobs) != 70 or len(cohorts) != 14
            or any(n != 5 for n in cohorts.values())
            or config['required_original_models'] != 70 or config['required_original_cohorts'] != 14
            or config['original_required_seeds'] != [42, 43, 44, 45, 46]):
        raise ValueError('Preserve all70 unique original jobs,14 complete cohorts and five original seeds')
    for key in cohorts:
        if {j['seed'] for j in jobs.values() if (j['model'], j['dataset'], j['data_track']) == key} != set(config['original_required_seeds']):
            raise ValueError('Every cohort must retain all five registered training seeds')
    records = catalog['records']
    if (catalog['full_native_generators'] != config['preserved_reviewed_models'] or len(records) != 30
            or len({r['canonical_id'] for r in records}) != 30
            or not {r['canonical_id'] for r in records}.issubset(jobs)):
        raise ValueError('Every previously closed30 canonical model must remain unique and registered')
    return jobs


def recovery_binding(original, execution):
    fields = {'output_directory', 'artifact_directory'}
    if {k: v for k, v in original.items() if k not in fields} != {k: v for k, v in execution.items() if k not in fields}:
        raise ValueError('Recovery may only isolate original output/artifact paths; no changed science or seed')


def verify(config):
    if config != registration():
        raise ValueError('Frozen full70 artifact-review contract changed')
    queue = read(ROOT / config['original_queue']); catalog = read(ROOT / config['reviewed_catalog'])
    jobs = validate_scope(config, queue, catalog)
    cases = {i: dict(job=j, queue=config['original_queue']) for i, j in jobs.items()}
    for route in config['recovery_routes']:
        execution = read(ROOT / route['execution_queue'])
        job = next(j for j in execution['jobs'] if j['id'] == route['canonical_id'])
        recovery_binding(jobs[job['id']], job)
        cases[job['id']] = dict(job=job, queue=route['execution_queue'])
    return cases, catalog


def source_ready(case):
    job = case['job']; queue = read(ROOT / case['queue'])
    result_path = ROOT / job['output_directory'] / 'result.json'
    guard_path = ROOT / queue['state_root'] / 'receipts' / (job['id'] + '.json')
    if not result_path.exists() or not guard_path.exists():
        return None
    result, guard = read(result_path), read(guard_path)
    if (guard['exitcode'] != 0 or guard['resource_guard_aborted'] or not guard['full_result_validated']):
        return None
    if (result['status'] != 'completed' or result['job'] != job or result['benchmark_smoke']
            or result['optimizer_updates'] != job['training_updates']
            or result['training_minibatches'] != job['training_updates'] * job['gradient_accumulate_every']
            or result['evaluation_repeats'] != 5):
        raise ValueError('Every complete source must retain its exact full frozen original training/evaluation budget')
    return result_path, guard_path, result


def controller(config, cases, catalog):
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value: None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full70 artifact-review controller already owned')
    verified = {}; failures = {}
    for row in catalog['records']:
        proof = read(ROOT / row['full_native_proof'])
        if (sha(ROOT / row['full_native_proof']) != row['full_native_proof_sha256']
                or not proof['passed'] or proof['canonical_id'] != row['canonical_id']
                or proof['metrics'] != row['metrics']):
            raise ValueError('Existing full native artifact proof changed')
        ready = source_ready(cases[row['canonical_id']])
        if ready is None or sha(ready[0]) != row['original_result_sha256'] or sha(ready[1]) != proof['actual_terminal_receipt_sha256']:
            raise ValueError('Preserved proof must retain its actual full source result and successful guard')
        for artifact in row['artifacts']:
            if sha(ROOT / artifact['path']) != artifact['sha256']:
                raise ValueError('Preserved full checkpoint/generated/reference artifact changed')
        verified[row['canonical_id']] = dict(kind='preserved_full_native_artifact_review',
            proof=row['full_native_proof'], proof_sha256=row['full_native_proof_sha256'])
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(values):
        state.update(values, verified=verified, failed_preserved=failures, verified_count=len(verified),
            required_original_models=70, required_original_cohorts=14,
            captured_utc=datetime.now(timezone.utc).isoformat(), all_original_experiments_complete=False)
        write(base / 'status.json', state)
    try:
        while len(verified) + len(failures) < 70:
            for identity, case in cases.items():
                if identity in verified or identity in failures:
                    continue
                try:
                    ready = source_ready(case)
                    if ready is None:
                        continue
                    attempt = base / 'attempts' / identity
                    output = ROOT / config['proof_root'] / identity
                    if attempt.exists() or output.exists():
                        raise FileExistsError('Previous full or partial artifact audit preserved; no automatic retry')
                    with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
                        result_path, guard_path, result = source_ready(case)
                        paths = [case['queue'], str(result_path), str(guard_path), case['job']['model_config'],
                            case['job']['author_yaml'], case['job']['frozen_samples'], case['job']['reference_path'],
                            *[r['path'] for r in result['artifacts']], CONFIG,
                            *[r['path'] for r in config['source_receipts']]]
                        target_config = ROOT / config['configuration_root'] / (identity + '.json')
                        exclusive(target_config, dict(queue=case['queue'], job_id=identity,
                            terminal_receipt=str(guard_path), output_directory=str(output),
                            source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in dict.fromkeys(paths)]))
                        attempt.mkdir(parents=True, exist_ok=False)
                        command = [config['python'], '-B', '-X', 'utf8', '-u', '-m',
                            'scripts.flow_matching.audit_spectral_completed_generation_v2', '--config', str(target_config)]
                        with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                                    PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                            update(dict(state='running_full_native_artifact_review', active_job=identity,
                                active_pid=child.pid, exact_child_handle=handle))
                            code, usage = capped_wait(child, config, api, update)
                        proof = output / 'independent_full_native_artifact_audit.json'
                        passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                        exclusive(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code,
                            exact_child_handle=handle, **usage, proof_sha256=sha(proof) if proof.exists() else None))
                        if not passed:
                            raise ValueError('Full native artifact audit failed; retain exact attempt without retry')
                        verified[identity] = dict(kind='new_full_native_artifact_review', proof=str(proof), proof_sha256=sha(proof))
                except Exception as error:
                    failures[identity] = str(error)
                update(dict(state='scanning_full_original_sources', active_pid=None))
            update(dict(state='waiting_full_original_sources', active_pid=None))
            time.sleep(config['poll_seconds'])
        update(dict(state='all70_artifact_review_attempts_terminal_other_original_experiments_remain', active_pid=None))
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        exclusive(ROOT / CONFIG, registration()); return
    config = read(ROOT / CONFIG); cases, catalog = verify(config)
    if args.probe_only:
        print(json.dumps(dict(required_models=len(cases), preserved_reviews=len(catalog['records']),
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            all_original_experiments_complete=False))); return
    controller(config, cases, catalog)


if __name__ == '__main__':
    main()
