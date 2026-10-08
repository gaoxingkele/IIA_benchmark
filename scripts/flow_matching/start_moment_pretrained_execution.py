"""Start one hidden complete MOMENT CPU scorer and pinned TAB range follower."""
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]


def launch(script, arguments, logs, python):
    runner = ROOT / script
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            args = process.info['cmdline'] or []
            if any(str(runner).lower() == a.lower() for a in args) and all(str(a) in args for a in arguments):
                return {'status': 'already_running', 'pid': process.pid}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logs.mkdir(parents=True, exist_ok=True)
    env = {k.upper():v for k,v in os.environ.items()}
    env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    flags = subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=='win32' else 0
    with (logs/'console.log').open('a',encoding='utf-8') as out,(logs/'stderr.log').open('a',encoding='utf-8') as err:
        child = subprocess.Popen([str(python),str(runner),*map(str,arguments)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                 stdout=out,stderr=err,creationflags=flags,start_new_session=sys.platform!='win32')
    return {'status':'launched','pid':child.pid}


def main():
    queue = ROOT/'configs/experiments/fm_moment_pretrained_queue.v1.json'
    settings = json.loads(queue.read_text())
    report = json.loads((ROOT/'docs/reports/fm_moment_pretrained_preflight_2026-10-09.json').read_text())
    from scripts.flow_matching.prepare_tsad_execution import sha
    if sha(queue) != report['queue_sha256']:
        raise ValueError('MOMENT preflight belongs to another queue')
    model = launch('scripts/flow_matching/run_moment_pretrained_queue.py',['--queue',queue],ROOT/settings['state_root'],settings['python'])
    ranges = launch('scripts/flow_matching/run_tab_ranges_for_config.py',['--config',ROOT/'configs/experiments/fm_moment_pretrained_tab_range.v1.json'],
                   ROOT/'experiments/runs/fm_moment_pretrained_tab_range_v1',sys.executable)
    print(json.dumps({'model':model,'ranges':ranges,'registered_jobs':len(settings['jobs'])}))


if __name__=='__main__':
    main()
