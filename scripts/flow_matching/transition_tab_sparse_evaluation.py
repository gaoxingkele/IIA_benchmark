"""Replace ancillary metric workers after verifying the exact source defect.

Model training is untouched. Inline incomplete range calculations are stopped
because the verified G*P None matrix is removed, not because a wait expired.
Their full saved model scores and all completed metric artifacts are preserved.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock
from scripts.flow_matching.transition_heavy_scheduler_v2 import launch

ROOT = Path(__file__).resolve().parents[2]


def main():
    registration_path = ROOT / 'docs/reports/fm_tab_sparse_registration_2026-10-09.json'
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    proof = json.loads((ROOT / 'docs/reports/fm_tab_sparse_affiliation_differential_2026-10-09.json').read_text(encoding='utf-8'))
    assert all(r['all_six_reference_outputs_bit_exact'] for r in proof['records'])
    archive = ROOT / 'experiments/runs/fm_tab_sparse_transition_20261009'
    archive.mkdir(parents=True, exist_ok=True)
    lock = exclusive_lock(archive / 'transition.lock')
    if lock is None or (archive / 'launched.json').exists():
        raise RuntimeError('Transition already owned/launched; inspect handles before restart')
    old = []
    for case in registration['configs']:
        parent = json.loads((ROOT / case['parent_config']).read_text(encoding='utf-8'))
        state_path = ROOT / parent['output_root'] / 'status.json'
        state = json.loads(state_path.read_text(encoding='utf-8'))
        process = psutil.Process(state['pid'])
        command = process.cmdline()
        expected = 'run_tab_ranges_when_ready.py' if case['registered_attempts'] == 1750 else 'run_tab_ranges_for_config.py'
        assert any(Path(a).name == expected for a in command), command
        process.suspend()
        try:
            assert not process.children(recursive=True), 'A child experiment must not be stopped'
            record = {'pid': process.pid, 'created_unix': process.create_time(), 'command': command,
                      'state_before_transition': state, 'state_sha256': sha(state_path),
                      'only_saved_score_metric_calculation_not_model_training': True,
                      'reason': 'Verified quadratic outer None-intersection matrix; same original outputs validated before replacement'}
            write_json(archive / (Path(case['parent_config']).stem + '_old_worker.json'), record)
            process.terminate()
            process.wait(timeout=10)
            old.append(record)
        except BaseException:
            if process.is_running():
                process.resume()
            raise
    reused = []
    for case in registration['configs']:
        command = [sys.executable, '-m', 'scripts.flow_matching.run_tab_ranges_sparse_queue', '--config', case['config'], '--reuse-only']
        subprocess.run(command, cwd=ROOT, check=True)
        state_path = ROOT / case['output_root'] / 'status.json'
        state = json.loads(state_path.read_text(encoding='utf-8'))
        reused.append({'config': case['config'], 'completed_original_evaluations_reused': state['completed_jobs']})
    launched = []
    for case in registration['configs']:
        process = launch([sys.executable, '-u', '-m', 'scripts.flow_matching.run_tab_ranges_sparse_queue', '--config', case['config']],
                         ROOT / case['output_root'], 'sparse_queue')
        launched.append(dict(process, config=case['config'], output_root=case['output_root']))
    write_json(archive / 'launched.json', launched)
    write_json(ROOT / 'docs/reports/fm_tab_sparse_transition_2026-10-09.json',
               {'old_workers': old, 'completed_reused': reused, 'launched': launched,
                'model_training_processes_stopped': False, 'old_inputs_results_and_metric_artifacts_preserved': True,
                'all_experiments_complete': False})
    print(json.dumps({'reused': reused, 'launched': launched}, indent=2))


if __name__ == '__main__':
    main()
