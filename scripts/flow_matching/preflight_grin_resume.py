"""Verify both GRIN Air datasets and exact CPU epoch-boundary resume."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Checkpoints identify callback classes as __main__.EpochRNG in CLI execution.
from scripts.flow_matching.run_grin import EpochRNG
import numpy as np
import torch


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def main():
    directory = ROOT / 'configs/experiments/fm_grin_jobs'
    base = ROOT / 'experiments/runs/flow_matching_campaign'
    for variant, stop in (('full_v2', None), ('resumed_v2', 1), ('resumed_v2', None), ('full437', None)):
        config = directory / ('diagnostic_' + variant + '.json')
        command = [sys.executable, str(ROOT / 'scripts/flow_matching/run_grin.py'), '--config', str(config)]
        if stop:
            command += ['--stop_after_epoch', str(stop)]
        stem = 'grin_preflight_' + variant + ('_interrupted' if stop else '')
        with (base / (stem + '.stdout.log')).open('w', encoding='utf-8') as stdout, (base / (stem + '.stderr.log')).open('w', encoding='utf-8') as stderr:
            subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    outputs = [base / ('grin_air36_resume_diagnostic_' + name) for name in ('full_v2', 'resumed_v2')]
    checkpoints = [torch.load(str(path / 'checkpoints/last.ckpt'), map_location='cpu') for path in outputs]
    reports = [json.loads((path / 'diagnostic_report.json').read_text(encoding='utf-8')) for path in outputs]
    checks = {key: equal(checkpoints[0][key], checkpoints[1][key])
              for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'epoch', 'global_step')}
    rng = [[value for key, value in checkpoint['callbacks'].items() if getattr(key, '__name__', '') == 'EpochRNG'][0]
           for checkpoint in checkpoints]
    checks['rng_state'] = equal(*rng)
    checks['selected_model_metrics'] = reports[0]['metrics'] == reports[1]['metrics']
    checks['completed_two_epochs'] = all(report['completed_epochs'] == 2 for report in reports)
    full437 = json.loads((base / 'grin_full437_diagnostic/diagnostic_report.json').read_text(encoding='utf-8'))
    checks['real_full437_finite'] = full437['status'] == 'completed' and all(np.isfinite(list(full437['metrics'].values())))
    checks['both_calendar_splits_disjoint'] = all(not any(report['data_audit']['timestamp_overlap_counts'].values()) for report in [reports[0], full437])
    report = {'status': 'passed' if all(checks.values()) else 'failed', 'diagnostic': True, 'checks': checks,
              'datasets': {name: {'parameters': result['parameters'], 'data_audit': result['data_audit']}
                           for name, result in [('air36', reports[0]), ('air', full437)]},
              'boundary': 'Two real windows per split, two CPU epochs. Exact CPU resume only; no benchmark performance or bit-identical GPU claim.'}
    (base / 'grin_cpu_preflight_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if not all(checks.values()):
        raise AssertionError('GRIN preflight failed')


if __name__ == '__main__':
    main()
