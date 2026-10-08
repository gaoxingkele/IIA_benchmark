"""Start idempotent hidden iterative-reflow and additional TAB continuations."""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.start_tsad_execution import launch
from scripts.flow_matching.prepare_tsad_execution import write_json


def main():
    config_path = 'configs/experiments/fm_reflow_tab_range.v1.json'
    config = json.loads((ROOT / config_path).read_text(encoding='utf-8'))
    target = str((ROOT / config_path).resolve()).lower().replace('\\', '/')
    metric_process = None
    for process in psutil.process_iter(['pid', 'cmdline']):
        arguments = process.info['cmdline'] or []
        if any('run_tab_ranges_for_config.py' in a for a in arguments) and '--config' in arguments:
            candidate = str(Path(arguments[arguments.index('--config') + 1]).resolve()).lower().replace('\\', '/')
            if candidate == target:
                metric_process = {'pid': process.pid, 'status': 'already_running'}
                break
    if metric_process is None:
        base = ROOT / config['output_root']
        base.mkdir(parents=True, exist_ok=True)
        environment = {key.upper(): value for key, value in os.environ.items()}
        environment['PYTHONUNBUFFERED'] = '1'
        command = [sys.executable, str(ROOT / 'scripts/flow_matching/run_tab_ranges_for_config.py'), '--config', str(ROOT / config_path)]
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0
        with (base / 'console.log').open('a', encoding='utf-8') as stdout, (base / 'stderr.log').open('a', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, creationflags=flags, start_new_session=sys.platform != 'win32')
        metric_process = {'pid': child.pid, 'status': 'launched'}
    records = {'training': launch('configs/experiments/fm_reflow_queue.v1.json', 'experiments/runs/fm_reflow_execution_v1', 'cpu_reflow'),
               'range_metrics': metric_process}
    write_json(ROOT / 'experiments/runs/fm_reflow_execution_v1/launch.json', records)
    print(json.dumps(records))


if __name__ == '__main__':
    main()
