"""Launch full CrossAD queue only after all 22 native full-batch checks pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--queue',required=True); args=parser.parse_args()
    path=ROOT/args.queue; queue=json.loads(path.read_text()); settings=queue['settings']
    if settings.get('preflight_parent_queue'):
        from scripts.flow_matching.run_crossad_case_gated_queue import native_proof
        assert any(native_proof(ROOT,job,settings) for job in queue['jobs'])
    else:
        report=json.loads((ROOT/settings['preflight_report']).read_text())
        assert report['queue_sha256']==sha(path) and report['all_native_cases_passed'] and len(report['cases'])==22
        for case in report['cases']:
            assert sha(ROOT/case['report'])==case['sha256']
    runner=ROOT/'scripts/flow_matching'/queue['worker_script']
    for p in psutil.process_iter(['pid','cmdline']):
        if any(str(runner).lower()==arg.lower() for arg in p.info['cmdline'] or []):
            print(json.dumps({'status':'already_running','pid':p.pid})); return
    base=ROOT/settings['output_root']; base.mkdir(parents=True,exist_ok=True)
    env={k.upper():v for k,v in os.environ.items()}; env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    flags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=='win32' else 0
    with (base/'complete_console.log').open('a',encoding='utf-8') as out,(base/'complete_stderr.log').open('a',encoding='utf-8') as err:
        child=subprocess.Popen([str(ROOT/settings['python']),str(runner),'--queue',str(path)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags,start_new_session=sys.platform!='win32')
    print(json.dumps({'status':'launched','pid':child.pid,'registered_jobs':len(queue['jobs'])}))


if __name__=='__main__':
    main()
