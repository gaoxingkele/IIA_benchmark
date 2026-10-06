"""Check author architectures on all real time-series data and corrected resume."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.preflight_saits import equal_state


def execute(path, interrupt=False):
    settings = json.loads(path.read_text(encoding='utf-8'))
    output = ROOT / settings['output_root']
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'scripts/flow_matching/run_saits.py'), '--config', str(path)]
    with (output / 'console.log').open('a', encoding='utf-8') as stdout, (output / 'stderr.log').open('a', encoding='utf-8') as stderr:
        if interrupt and not (output / 'resume.pt').exists():
            subprocess.run(command + ['--diagnostic_stop_after_epoch', '1'], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    return json.loads((output / 'diagnostic_report.json').read_text(encoding='utf-8'))


def main():
    import torch
    records = []
    directory = ROOT / 'configs/experiments/fm_saits_timeseries_jobs'
    for path in sorted(directory.glob('diagnostic_*.json')):
        report = execute(path, interrupt='resumed' in path.name)
        records.append({'id': report['config']['id'], 'dataset': report['config']['dataset'], 'variant': report['config']['variant'],
                        'method': report['config']['method'], 'finite': all(torch.isfinite(torch.tensor(x)) for x in report['metrics'].values()),
                        'parameters': report['parameters']})
    base = ROOT / 'experiments/runs/flow_matching_campaign'
    prefix = 'saits_ts_electricity_saits_corrected_cpu'
    states = [torch.load(base / (prefix + suffix) / 'resume.pt', map_location='cpu', weights_only=False) for suffix in ('', '_resumed')]
    checks = {key: equal_state(states[0][key], states[1][key]) for key in ('model', 'optimizer', 'controller', 'numpy_rng', 'torch_rng')}
    if not all(checks.values()) or not all(r['finite'] for r in records):
        raise AssertionError('Real-data preflight or corrected/dropout checkpoint recovery failed')
    result = {'status': 'passed', 'diagnostic': True, 'records': records, 'corrected_dropout_resume_checks': checks,
              'boundary': 'Full author architecture on four real training and two validation/test windows for two CPU epochs. No benchmark performance claim.'}
    (base / 'saits_timeseries_cpu_preflight_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
