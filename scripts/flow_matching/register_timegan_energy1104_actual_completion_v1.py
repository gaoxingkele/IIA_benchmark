"""Preserve actual full native outputs and the failed host-dependency exit.

The independent read-only verification has its own exit code; it never rewrites
the original worker's exit1 receipt or claims an independent numerical replay.
"""
import copy
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v3 as dispatch
from scripts.flow_matching import run_timegan_full45_completion_replay_v5 as replay
from scripts.flow_matching.timegan_native_result_verifier_v2 import verify_result

ROOT = dispatch.original.ROOT
read = dispatch.original.read
sha = dispatch.original.sha
exclusive = dispatch.original.exclusive
EVIDENCE = 'projects/flow_matching_research/results/2026-10-13/timegan_energy1104_actual_completion_v1'
IDENTITY = 'timegan_native__gru__energy__seed1104'


def exact_owner(pid):
    import psutil
    process = psutil.Process(pid)
    return dict(pid=pid, create_time=process.create_time(), command=process.cmdline())


def pins(config, added):
    receipts = {item['path']: item['sha256'] for item in config['source_receipts']}
    for path in added:
        receipts[path] = sha(ROOT / path)
    config['source_receipts'] = [dict(path=path, sha256=value) for path, value in sorted(receipts.items())]


def main():
    config = copy.deepcopy(read(ROOT / dispatch.PREVIOUS))
    queue = dispatch.original.verify_runtime(config)
    job = next(j for j in queue['jobs'] if j['id'] == IDENTITY)
    result = verify_result(queue, job)
    prior_base = ROOT / config['roles']['original']['state_root']
    failed_receipt_path = prior_base / 'receipts' / (IDENTITY + '.json')
    failed = read(failed_receipt_path)
    if failed['exit_code'] != 1 or failed['resource_guard_aborted'] is not False:
        raise ValueError('Exact observed terminal verification-only failure required')
    stderr_path = prior_base / 'logs' / (IDENTITY + '.stderr.log')
    stderr = stderr_path.read_text(encoding='utf-8')
    if "ModuleNotFoundError: No module named 'psutil'" not in stderr:
        raise ValueError('Preserved terminal missing host-monitor dependency required')
    evidence = ROOT / EVIDENCE
    evidence.mkdir(parents=True, exist_ok=False)
    sources = {
        'result.json': ROOT / job['output_directory'] / 'result.json',
        'completed_evaluators.json': ROOT / job['output_directory'] / 'completed_evaluators.json',
        'preserved_worker_exit1_receipt.json': failed_receipt_path,
        'preserved_worker_stderr.log': stderr_path,
        'preserved_worker_stdout.log': prior_base / 'logs' / (IDENTITY + '.stdout.log'),
    }
    artifacts = []
    for name, source in sources.items():
        data = source.read_bytes()
        target = evidence / name
        with target.open('xb') as stream:
            stream.write(data)
        artifacts.append(dict(source_path=source.as_posix(), archive_name=name, bytes=len(data), sha256=sha(target)))
    verification = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=0, verification_kind='independent_read_only_exact_native_result_verifier',
        original_training_worker_exit_code=failed['exit_code'],
        original_worker_receipt_path=failed_receipt_path.relative_to(ROOT).as_posix(),
        original_worker_receipt_sha256=sha(failed_receipt_path),
        resource_guard_aborted=False, full_native_result_verified=True,
        result_path=sources['result.json'].as_posix(), result_sha256=sha(sources['result.json']),
        canonical_job=job, optimizer_updates=result['optimizer_updates'],
        full_original_data_shape=result['full_original_data_shape'],
        native_checkpoint_count=len(result['checkpoints']), completed_native_evaluators=len(result['evaluators']),
        scores=result['scores'], numerical_environment_or_author_source_changed=False,
        scientific_training_repeated=False, independent_full_checkpoint_replay_completed=False,
        author_equivalence_certified=False, strict_TAB_result=False,
        original_scope=config['original_scope'], all_original_experiments_complete=False)
    exclusive(evidence / 'independent_completion_verification.json', verification)
    exclusive(evidence / 'completion_evidence.json', dict(artifacts=artifacts,
        native_training_and_all_evaluators_completed=True, original_failed_exit_preserved=True,
        verifier_function_ast_unchanged=True, missing_dependency_is_host_process_monitor=True,
        original_scope=config['original_scope'], all_original_experiments_complete=False))
    added = [dispatch.PREVIOUS, replay.PREVIOUS,
        'scripts/flow_matching/timegan_native_result_verifier_v2.py',
        'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v3.py',
        'scripts/flow_matching/run_timegan_full45_completion_replay_v5.py',
        'scripts/flow_matching/register_timegan_energy1104_actual_completion_v1.py',
        'tests/test_fm_timegan_native_verifier_isolation_v2.py']
    added += [p.relative_to(ROOT).as_posix() for p in evidence.iterdir()]
    config.update(runtime_configuration='configs/runtime/fm_timegan_corrected_observer_dispatch.v3.json',
                  native_result_verification='exact_original_function_without_host_monitor_import')
    route = config['roles']['original']
    route.update(runtime_configuration=config['runtime_configuration'],
        previous_owner=exact_owner(27364),
        previous_state_path='experiments/runs/fm_timegan_corrected_observer_original_v2/status.json',
        state_root='experiments/runs/fm_timegan_corrected_observer_original_v3')
    pins(config, added)
    dispatch.verify_runtime(config)
    exclusive(ROOT / config['runtime_configuration'], config)
    full = copy.deepcopy(read(ROOT / replay.PREVIOUS))
    full.update(runtime_configuration='configs/runtime/fm_timegan_full45_completion_replay.v6.json',
        state_root='experiments/runs/fm_timegan_full45_completion_replay_v6',
        output_root='projects/flow_matching_research/results/2026-10-13/timegan_full45_completion_replay_v6/completed_models',
        routes=replay.expected_routes(full), previous_replay_owner=exact_owner(39264),
        previous_replay_state_path='experiments/runs/fm_timegan_full45_completion_replay_v5/status.json')
    pins(full, added + [config['runtime_configuration']])
    replay.verify(full)
    exclusive(ROOT / full['runtime_configuration'], full)
    print(config['runtime_configuration'])
    print(full['runtime_configuration'])


if __name__ == '__main__':
    main()
