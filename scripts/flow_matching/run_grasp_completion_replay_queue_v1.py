"""Continue full native replay and independent recipe review for all9120 GRASP jobs."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT, read, sha, SCIENTIFIC_FIELDS
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.verify_grasp_completed_full_replay_v1 import check_terminal


EXECUTION_FIELDS = {'id', 'output_directory', 'recovery_original_job_id'}
GUARDS = ('python', 'cpu_threads', 'settings', 'resource_lock', 'score_tolerance', 'metric_tolerance')
SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)


def check_binding(original, execution):
    if (execution.get('recovery_original_job_id', execution['id']) != original['id']
            or {k:v for k,v in original.items() if k not in EXECUTION_FIELDS}
            != {k:v for k,v in execution.items() if k not in EXECUTION_FIELDS}):
        raise ValueError('Preserve every original GRASP scientific field, seed and budget')


def check_complete_result(original, execution, result):
    check_binding(original, execution)
    if (result['status'] != 'completed' or result['job'] != execution or result['diagnostic']
            or result['epochs_completed'] != 1500
            or result['optimizer_updates'] != original['expected_optimizer_updates']
            or result['scored_points'] != original['test_points']
            or result['validation_points'] != original['validation_points']
            or result['score_recipes'] != original['score_recipes']
            or [r['recipe'] for r in result['metrics']] != original['score_recipes']
            or result['paper_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('Only the unchanged full original training and every original score recipe may enter replay')


def validate_scope(queue, runtime):
    jobs = {j['id']:j for j in queue['jobs']}
    if (len(jobs) != len(queue['jobs']) or len(jobs) != 9120 or queue['cohort_count'] != 480
            or sorted(queue['seeds']) != list(range(1103,1113))
            or {j['seed'] for j in jobs.values()} != set(range(1103,1113))
            or runtime['required_original_models'] != 9120 or runtime['required_original_cohorts'] != 480
            or runtime['required_original_training_seeds'] != 10 or runtime['original_scope'] != SCOPE):
        raise ValueError('Keep all9120 original entity jobs/480cohorts/10original seeds and the full research scope')
    return jobs


def current_native(config_path):
    config = read(ROOT/config_path)
    report = read(ROOT/config['output_directory']/'checkpoint_replay_audit.json')
    receipt = read(ROOT/config['state_root']/'receipt.json')
    check_terminal(report, receipt, config)
    if (report['configuration_sha256'] != sha(ROOT/config_path)
            or report['score_tolerance'] != config['score_tolerance']
            or report['metric_tolerance'] != config['metric_tolerance']):
        raise ValueError('Completed native evidence must bind its actual frozen original configuration')
    return config, report, receipt


def check_preserved(binding, original, execution):
    result = read(ROOT/binding['result_path'])
    check_complete_result(original, execution, result)
    if sha(ROOT/binding['result_path']) != binding['result_sha256']:
        raise ValueError('Preserved complete original training result changed')
    proof = read(ROOT/binding['independent_proof'])
    config, report, receipt = current_native(binding['native_configuration'])
    if binding['kind'] == 'legacy_union':
        if (proof['full_original_models'] != 13
                or not proof['all_complete_arrays_point_mapping_and_metrics_independently_verified']
                or not proof['original_full_native_worker_exit_success_verified']
                or not proof['no_new_training_seed_slots']):
            raise ValueError('Preserved full thirteen-model union must remain complete and independently checked')
        row = next(r for r in proof['records'] if r['canonical_id'] == original['id'])
        if (row['original_complete_training_result_sha256'] != binding['result_sha256']
                or not row['all_original_checkpoint_history_and_scoring_artifact_hashes_verified']
                or row['epochs_completed'] != result['epochs_completed']
                or row['optimizer_updates'] != result['optimizer_updates']
                or row['parameter_count'] != result['parameter_count']):
            raise ValueError('Preserved union must bind the exact completed original model')
        recipes = row['recipes']
        if not all(r['passed'] for r in recipes):
            raise ValueError('Every preserved original scoring recipe must pass')
    else:
        native = next(r for r in report['records'] if r['canonical_id'] == original['id'])
        if native['result_path'] != binding['result_path'] or native['result_sha256'] != binding['result_sha256']:
            raise ValueError('Preserved native replay belongs to another execution')
        recipes = native['recipes']
        if binding['kind'] in ('generic', 'lin'):
            if (any(not proof[k] for k in ('full_original_model_restoration_and_reinference_completed',
                    'all_saved_full_score_arrays_and_identities_reverified',
                    'all_original_forward_counts_and_recipe_budgets_verified',
                    'all_metrics_independently_recomputed_from_full_replay'))
                    or proof['terminal_resource_receipt'] != receipt
                    or proof['additional_training_seed_slots'] != 0
                    or proof['completed_native_replay_configuration'] != binding['native_configuration']
                    or proof['completed_native_replay_report_sha256'] != sha(ROOT/config['output_directory']/'checkpoint_replay_audit.json')
                    or {json.dumps(r['recipe'], sort_keys=True) for r in proof['recipes']}
                    != {json.dumps(r, sort_keys=True) for r in original['score_recipes']}
                    or binding['kind'] == 'generic' and not proof['passed']):
                raise ValueError('Preserved full-model/forward-count/array/metric review must match actual terminal native evidence')
        elif binding['kind'] == 'er':
            row = next(r for r in proof['grasp_er_records'] if r['canonical_id'] == original['id'])
            if (not row['full_native_checkpoint_replay_and_original_arrays_verified']
                    or row['result_sha256'] != binding['result_sha256']
                    or row['original_score_sha256'] != recipes[0]['original_score_sha256']
                    or row['replay_score_sha256'] != recipes[0]['replay_score_sha256']):
                raise ValueError('Preserved ER independent full native and original-array evidence differs')
        elif binding['kind'] == 'transformer':
            if (not proof['passed'] or not proof['independent_full_array_and_original_native_metric_validation']
                    or proof['full_native_model_replay_receipt'] != receipt or proof['new_training_seed_slots'] != 0
                    or proof['checkpoint_replay_audit_sha256'] != sha(ROOT/config['output_directory']/'checkpoint_replay_audit.json')
                    or proof['original_native_configuration_sha256'] != sha(ROOT/binding['native_configuration'])):
                raise ValueError('Preserved Transformer full native and independent evidence differs')
        else:
            raise ValueError('Unknown preserved original proof format')
    if len(recipes) != len(original['score_recipes']) or [r['recipe'] for r in recipes] != original['score_recipes']:
        raise ValueError('All original preserved recipes required')
    for recipe in recipes:
        if (recipe['complete_validation_points'] != original['validation_points']
                or recipe['complete_test_points'] != original['test_points']):
            raise ValueError('All original full points required')
        for tag in ('original', 'replay'):
            if sha(recipe[tag+'_score_path']) != recipe[tag+'_score_sha256']:
                raise ValueError('Preserved complete original/replayed scores changed')
    for artifact in result['artifacts']:
        if sha(ROOT/artifact['path']) != artifact['sha256']:
            raise ValueError('Preserved original trained checkpoint/history/scores changed')


def verify(runtime):
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen full GRASP continuation source changed: ' + pin['path'])
    queue = read(ROOT/runtime['original_queue'])
    originals = validate_scope(queue, runtime)
    template = read(ROOT/runtime['native_replay_template'])
    if template['queue'] != runtime['original_queue'] or any(runtime[k] != template[k] for k in GUARDS):
        raise ValueError('Every original native replay admission/emergency/resource/score/metric guard must remain')
    executions = dict(originals)
    for recovery in runtime['recovery_routes']:
        jobs = read(ROOT/recovery['execution_queue'])['jobs']
        execution = next(j for j in jobs if j['id'] == recovery['execution_id'])
        canonical = recovery['canonical_id']
        check_binding(originals[canonical], execution)
        executions[canonical] = execution
    covered = {}
    for binding in runtime['preserved_verified_models']:
        canonical = binding['canonical_id']
        if canonical in covered or canonical not in originals:
            raise ValueError('Preserved original canonical model proofs must be unique')
        if binding['result_path'] != (Path(executions[canonical]['output_directory'])/'result.json').as_posix():
            raise ValueError('Preserved result must match its explicit original execution')
        check_preserved(binding, originals[canonical], executions[canonical])
        covered[canonical] = binding
    if len(covered) != 22:
        raise ValueError('Preserve all22 previously complete full native/independent original models')
    return originals, executions, template, covered


def registration(runtime, runtime_path, template, original, execution):
    result_path = (Path(execution['output_directory'])/'result.json').as_posix()
    result = read(ROOT/result_path)
    check_complete_result(original, execution, result)
    canonical = original['id']
    native = copy.deepcopy(template)
    native.update(full_results=[dict(canonical_id=canonical, result_path=result_path, result_sha256=sha(ROOT/result_path))],
        full_job_count=1, full_score_recipe_count=len(original['score_recipes']),
        output_directory=runtime['proof_root']+'/'+canonical+'/native',
        state_root=runtime['child_state_root']+'/'+canonical+'/native',
        artifact_root=str(Path(runtime['artifact_root'])/canonical),
        source_receipts=copy.deepcopy(runtime['source_receipts']) + [dict(path=runtime_path, sha256=sha(ROOT/runtime_path)),
            dict(path=result_path, sha256=sha(ROOT/result_path))])
    config_root = Path(runtime['child_configuration_root'])
    native_path = (config_root/(canonical+'_native.json')).as_posix()
    pipeline_path = (config_root/(canonical+'_pipeline.json')).as_posix()
    exclusive_json(ROOT/native_path, native)
    pipeline = {k:copy.deepcopy(native[k]) for k in (*GUARDS, 'original_scope')}
    pipeline.update(canonical_id=canonical, producer_configuration=native_path,
        review_configuration=(config_root/(canonical+'_completed_review.json')).as_posix(),
        review_output_directory=runtime['proof_root']+'/'+canonical+'/independent',
        output_directory=runtime['proof_root']+'/'+canonical+'/completion',
        state_root=runtime['child_state_root']+'/'+canonical+'/pipeline',
        source_receipts=copy.deepcopy(native['source_receipts']) + [dict(path=native_path, sha256=sha(ROOT/native_path))])
    exclusive_json(ROOT/pipeline_path, pipeline)
    return pipeline_path


def completed_pipeline(runtime, canonical):
    from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import terminal, binding
    path = (Path(runtime['child_configuration_root'])/(canonical+'_pipeline.json')).as_posix()
    config = read(ROOT/path)
    producer = binding(config)
    terminal(config, producer)
    receipt = read(ROOT/config['state_root']/'receipt.json')
    proof_path = ROOT/config['output_directory']/'completed_native_and_independent_review.json'
    proof = read(proof_path)
    independent_path = ROOT/config['review_output_directory']/'independent_full_replay_verification.json'
    independent = read(independent_path)
    result = read(ROOT/producer['full_results'][0]['result_path'])
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
            or not receipt['complete_native_and_independent_review_passed']
            or not proof['passed'] or proof['canonical_id'] != canonical
            or proof['configuration_sha256'] != sha(ROOT/path)
            or proof['original_training_seed'] != result['job']['seed']
            or proof['required_original_cohort_repeats'] != 10
            or proof['independent_full_replay_proof_sha256'] != sha(independent_path)
            or not independent['passed'] or not independent['all_original_forward_counts_and_recipe_budgets_verified']
            or not independent['all_saved_full_score_arrays_and_identities_reverified']
            or not independent['all_metrics_independently_recomputed_from_full_replay']
            or independent['actual_completed_recipe_count'] != producer['full_score_recipe_count']
            or proof['additional_training_seed_slots'] != 0):
        raise ValueError('Complete actual native and independent recipe terminal evidence required')
    return dict(canonical_id=canonical, completed_pipeline_configuration=path,
        completed_full_replay_proof=proof_path.relative_to(ROOT).as_posix(), proof_sha256=sha(proof_path))


def live_replay_ids():
    """Inspect actual PID/creation/command/config; stale status and locks prove no liveness."""
    ids = set()
    for process in psutil.process_iter(['name']):
        if 'python' not in (process.info['name'] or '').lower():
            continue
        try:
            command = process.cmdline()
            if '--config' not in command or not any('grasp' in item and any(t in item for t in
                    ('audit_grasp_checkpoint_replay', 'run_grasp_registered_replay_review_pipeline')) for item in command):
                continue
            identity = process.create_time()
            config = read(ROOT/command[command.index('--config')+1])
            if process.create_time() != identity or process.cmdline() != command:
                continue
            ids.update(r['canonical_id'] for r in config.get('full_results', []))
            if 'canonical_id' in config:
                ids.add(config['canonical_id'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError, KeyError):
            continue
    return ids


def controller(runtime, path):
    originals, executions, template, verified = verify(runtime)
    api = resource_api()
    base = ROOT/runtime['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None:
        raise RuntimeError('This original full9120 completion/replay queue already has an owner')
    previous = read(base/'status.json') if (base/'status.json').exists() else {}
    failed = previous.get('failed_preserved', {})
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), configuration_sha256=sha(ROOT/path),
        required_original_models=9120, required_original_cohorts=480, required_original_training_seeds=10,
        verified_original_ids=sorted(verified), failed_preserved=failed, active_pid=None,
        original_scope=runtime['original_scope'], additional_training_seed_slots=0, all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json', state)
    try:
        while len(verified) < len(originals):
            live = live_replay_ids()
            for canonical, original in originals.items():
                if canonical in verified or canonical in failed or canonical in live:
                    continue
                execution = executions[canonical]
                result_path = ROOT/execution['output_directory']/'result.json'
                if not result_path.exists():
                    continue
                try:
                    result = read(result_path)
                except (OSError, ValueError):
                    continue  # A result still being written is not terminal failure.
                if result.get('status') != 'completed':
                    continue
                try:
                    pipeline_path = (Path(runtime['child_configuration_root'])/(canonical+'_pipeline.json')).as_posix()
                    if (ROOT/pipeline_path).exists():
                        verified[canonical] = completed_pipeline(runtime, canonical)
                    else:
                        pipeline_path = registration(runtime, path, template, original, execution)
                        logs = base/'logs'/canonical
                        logs.mkdir(parents=True, exist_ok=False)
                        command = [runtime['python'], '-X', 'utf8', '-u', '-m',
                            'scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2', '--config', pipeline_path]
                        with (logs/'pipeline.stdout.log').open('xb') as out, (logs/'pipeline.stderr.log').open('xb') as err:
                            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), PYTHONUTF8='1',
                                    PYTHONDONTWRITEBYTECODE='1'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        handle = psutil.Process(child.pid)
                        creation, actual = handle.create_time(), handle.cmdline()
                        update(dict(state='waiting_actual_full_native_replay_and_independent_recipe_review',
                            active_pid=child.pid, active_create_time=creation, actual_command=actual, active_job=canonical))
                        while child.poll() is None:
                            try:
                                current = psutil.Process(child.pid)
                                if current.create_time() != creation or current.cmdline() != actual:
                                    raise ValueError('Active full native pipeline identity changed')
                                update(dict(active_controller_cpu_seconds=sum(current.cpu_times()[:2])))
                            except psutil.NoSuchProcess:
                                if child.poll() is None:
                                    raise
                            time.sleep(runtime['poll_seconds'])
                        code = child.wait()
                        update(dict(active_pid=None, active_job=None))
                        if code != 0:
                            raise RuntimeError('Full native replay/review failed; preserve its actual attempt and inputs')
                        verified[canonical] = completed_pipeline(runtime, canonical)
                    exclusive_json(base/'completed'/ (canonical+'.json'), verified[canonical])
                except Exception as error:
                    failed[canonical] = dict(error_type=type(error).__name__, error=str(error), failed_attempt_preserved=True)
                update(dict(verified_original_ids=sorted(verified), failed_preserved=failed,
                    pending_original_models=9120-len(verified), active_pid=None, active_job=None))
            update(dict(state='waiting_actual_original_full_training_completion',
                verified_original_ids=sorted(verified), pending_original_models=9120-len(verified),
                live_other_full_replay_ids=sorted(live), failed_preserved=failed))
            time.sleep(runtime['poll_seconds'])
        update(dict(state='all_registered_grasp_native_replays_and_independent_recipes_verified'))
    finally:
        lane.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    runtime = read(ROOT/args.config)
    if args.probe_only:
        originals, executions, template, verified = verify(runtime)
        print(json.dumps(dict(required_original_models=len(originals), previously_complete_native_reviewed_models=len(verified),
            remaining_original_models=len(originals)-len(verified), required_original_seeds=10,
            numpy_imported='numpy' in __import__('sys').modules, torch_imported='torch' in __import__('sys').modules,
            all_original_experiments_complete=False)))
    else:
        controller(runtime, args.config)
