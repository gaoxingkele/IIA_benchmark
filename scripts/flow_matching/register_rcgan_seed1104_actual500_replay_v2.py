"""Archive exact completed training evidence and register all five full native replays."""
import copy
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.run_rcgan_full500_completion_replay_queue_v2 import ROOT, original, verify


def main():
    previous_path = 'configs/runtime/fm_rcgan_full500_completion_replay_queue.v1.json'
    config = copy.deepcopy(original.read(ROOT / previous_path))
    recovery_path = 'configs/reproducibility/fm_rcgan_seed1104_full134_state_recovery.2026-10-12_v3.json'
    recovery = original.read(ROOT / recovery_path)
    route = config['routes'][1]
    route.update(execution_job=recovery['recovery_job'], recovery_configuration=recovery_path,
                 native_receipt=recovery['state_root'] + '/receipt.json',
                 result_path=(Path(recovery['recovery_job']['output_directory']) / 'result.json').as_posix())
    result = original.read(route['result_path'])
    receipt = original.read(ROOT / route['native_receipt'])
    original.complete_metadata(route, result, receipt)
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-13/rcgan_seed1104_actual500_completion_v1'
    evidence.mkdir(parents=True, exist_ok=False)
    paths = [route['result_path'], route['native_receipt'], recovery['state_root'] + '/status.json',
             recovery['state_root'] + '/full_native_resume_proof_binding.json']
    paths += [(Path(recovery['recovery_job']['output_directory']) / name).as_posix() for name in
              ('completed_full_epochs.json', 'completed_full_evaluators.json',
               'actual_full_resume_restoration.json', 'full_native_loader_audit.json')]
    artifacts = []
    for path in paths:
        target = evidence / Path(path).name
        data = (ROOT / path).read_bytes()
        with target.open('xb') as output:
            output.write(data)
        artifacts.append(dict(source_path=path, archive_name=target.name, sha256=original.sha(target), bytes=len(data)))
    original.exclusive(evidence / 'completion_evidence.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts,
        canonical_job=route['canonical_job'], actual_execution_job=route['execution_job'],
        verified_prior_epochs=366, completed_resume_epochs=134, completed_original_epochs=500,
        completed_original_evaluators=500, full_native_training_budget_completed=True,
        independent_full500_native_replay_completed=False, author_equivalence_certified=False,
        strict_TAB_result=False, original_scope=config['original_scope'], all_original_experiments_complete=False))
    config.update(runtime_configuration='configs/runtime/fm_rcgan_full500_completion_replay_queue.v2.json',
                  state_root='experiments/runs/fm_rcgan_full500_completion_replay_queue_v2',
                  output_root='projects/flow_matching_research/results/2026-10-13/rcgan_full500_completion_replay_v2/completed_models')
    pins = {p['path']: p['sha256'] for p in config['source_receipts']}
    for path in [previous_path, recovery_path,
                 'scripts/flow_matching/run_rcgan_full500_completion_replay_queue_v2.py',
                 'scripts/flow_matching/register_rcgan_seed1104_actual500_replay_v2.py',
                 'tests/test_fm_rcgan_seed1104_actual500_replay_v2.py']:
        pins[path] = original.sha(ROOT / path)
    for item in recovery['source_receipts']:
        pins[item['path']] = item['sha256']
    for path in evidence.iterdir():
        pins[path.relative_to(ROOT).as_posix()] = original.sha(path)
    config['source_receipts'] = [dict(path=p, sha256=s) for p, s in sorted(pins.items())]
    verify(config)
    original.exclusive(ROOT / config['runtime_configuration'], config)
    print(config['runtime_configuration'])


if __name__ == '__main__':
    main()
