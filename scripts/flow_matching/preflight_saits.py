"""Validate both full author architectures and exact CPU epoch resume on real data."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def equal_state(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def main():
    import torch
    base = ROOT / 'experiments/runs/flow_matching_campaign'
    records = []
    for model in ('saits', 'brits'):
        for mode in ('full', 'resumed'):
            path = ROOT / f'configs/experiments/fm_saits_jobs/diagnostic_{model}_{mode}.json'
            command = [sys.executable, str(ROOT / 'scripts/flow_matching/run_saits.py'), '--config', str(path)]
            output = base / f'saits_{model}_cpu_{mode}'
            output.mkdir(exist_ok=True)
            with (output / 'console.log').open('a', encoding='utf-8') as stdout, (output / 'stderr.log').open('a', encoding='utf-8') as stderr:
                if mode == 'resumed' and not (output / 'resume.pt').exists():
                    subprocess.run(command + ['--diagnostic_stop_after_epoch', '1'], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
                    checkpoint = torch.load(output / 'resume.pt', map_location='cpu', weights_only=False)
                    assert checkpoint['next_epoch'] == 1 and not checkpoint['finished']
                subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        full, resumed = [torch.load(base / f'saits_{model}_cpu_{mode}' / 'resume.pt', map_location='cpu', weights_only=False) for mode in ('full', 'resumed')]
        keys = ('model', 'optimizer', 'controller', 'patience', 'next_epoch', 'finished', 'python_rng', 'numpy_rng', 'torch_rng')
        checks = {key: equal_state(full[key], resumed[key]) for key in keys}
        full_report, resumed_report = [json.loads((base / f'saits_{model}_cpu_{mode}' / 'diagnostic_report.json').read_text(encoding='utf-8')) for mode in ('full', 'resumed')]
        checks['metrics'] = full_report['metrics'] == resumed_report['metrics']
        checks['validation_selection'] = full_report['validation_best'] == resumed_report['validation_best']
        if not all(checks.values()):
            raise AssertionError(f'{model} restart differs from uninterrupted training: {checks}')
        records.append({'model': model, 'checks': checks, 'parameters': full_report['parameters'], 'finite': True,
                        'boundary': 'Four real training patients, two validation/test patients, two CPU epochs; plumbing verification only, no benchmark score.'})
    report = {'status': 'passed', 'diagnostic': True, 'records': records}
    (base / 'saits_cpu_preflight_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
