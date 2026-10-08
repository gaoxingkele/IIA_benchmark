"""Isolated full-native-batch checks with shared heavy-job and commit-memory guards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.heavy_resources import waited_lock, wait_for_headroom, guarded_wait
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--queue', required=True); parser.add_argument('--launch-after', action='store_true')
    args = parser.parse_args(); path = ROOT/args.queue; queue = json.loads(path.read_text()); settings = queue['settings']
    version = str(queue['schema_version'])
    prefix = 'fm_crossad_complete_preflight_v'+version
    base = ROOT/'experiments/runs'/prefix; base.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'queue_sha256': sha(path), 'state': 'starting', 'active_case': None}
    def update(value):
        state.update(value); write_json(base/'status.json', state)
    cases = [(d, None) for d in settings['datasets']] + [(d, r) for r in range(1,6) for d in ('GECCO','MSL','PSM')]
    records = []
    with waited_lock(base/'worker.lock', update):
        for dataset, row in cases:
            name = dataset+('_row'+str(row) if row else '')
            output = base/name; output.mkdir(exist_ok=True)
            case_report = ROOT/'docs/reports'/(prefix+'_'+name+'_2026-10-09.json')
            if case_report.exists():
                report = json.loads(case_report.read_text())
                assert report['queue_sha256'] == sha(path) and len(report['records']) == 1
                records.append(dict(dataset=dataset, row=row, status='completed', report=case_report.relative_to(ROOT).as_posix(), sha256=sha(case_report)))
                continue
            if (output/'failure.json').exists():
                records.append(dict(dataset=dataset, row=row, status='failed_preserved')); continue
            with waited_lock(ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock', update):
                start = wait_for_headroom(settings, update)
                command = [str(ROOT/settings['python']), '-u', '-m', 'scripts.flow_matching.preflight_crossad_author',
                           '--queue', str(path), '--dataset', dataset, '--report-prefix', prefix]
                if row:
                    command += ['--row', str(row)]
                env = {k.upper():v for k,v in os.environ.items()}; env.update(CUDA_VISIBLE_DEVICES='-1', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                with (output/'stdout.log').open('a', encoding='utf-8') as out, (output/'stderr.log').open('a', encoding='utf-8') as err:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags)
                    update({'state':'running', 'active_case':name, 'active_pid':child.pid, 'started_unix':time.time()})
                    code, resources = guarded_wait(child, settings, update)
                write_json(output/'resource_usage.json', dict(resources, start_resources=start))
            if code or not case_report.exists():
                write_json(output/'failure.json', {'exit_code':code, 'resource_usage':resources, 'status':'failed_preserved'})
                records.append(dict(dataset=dataset,row=row,status='failed_preserved'))
            else:
                report = json.loads(case_report.read_text()); assert report['queue_sha256'] == sha(path) and len(report['records']) == 1
                records.append(dict(dataset=dataset,row=row,status='completed',report=case_report.relative_to(ROOT).as_posix(),sha256=sha(case_report)))
            update({'active_case':None,'active_pid':None,'completed_cases':sum(r['status']=='completed' for r in records)})
        complete = len(records)==22 and all(r['status']=='completed' for r in records)
        report = {'queue_sha256':sha(path),'cases':records,'all_native_cases_passed':complete,'benchmark_performance':False}
        write_json(ROOT/settings['preflight_report'],report)
        update({'state':'completed' if complete else 'completed_with_unresolved_cases'})
    if complete and args.launch_after:
        subprocess.run([sys.executable,'-m','scripts.flow_matching.start_crossad_complete','--queue',str(path)],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
