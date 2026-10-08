"""Run preserved exact-budget incident retries under the shared heavy-job guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.heavy_resources import waited_lock,wait_for_headroom,guarded_wait
from scripts.flow_matching.prepare_tsad_execution import sha,write_json
from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts
from scripts.flow_matching.run_tsad_queue import complete_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True);args=parser.parse_args()
    path=ROOT/args.queue;queue=json.loads(path.read_text());settings=queue['settings'];base=ROOT/settings['output_root'];base.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'queue_sha256':sha(path),'active_job':None,'active_pid':None,'jobs':{}}
    def update(value):state.update(value);write_json(base/'status.json',state)
    with waited_lock(base/'worker.lock',update):
        for f in queue['frozen_files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
        while not (ROOT/settings['prerequisite_report']).exists():
            update({'state':'waiting_for_full_native_preflight'});time.sleep(10)
        for case in queue['jobs']:
            job=case['job'];output=ROOT/job['output_directory'];final=output/'result.json'
            if final.exists():
                receipt=json.loads(final.read_text())
                assert receipt['experiment_sha256']==hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
                if case['kind'] in ('strict','industrial'):assert complete_result(ROOT,job)
                else:verify_artifacts(receipt)
                state['jobs'][job['id']]='completed';continue
            if output.exists():state['jobs'][job['id']]='partial_preserved';continue
            # Never retry an original job while a matching original child still runs.
            original_id=case['original_id']
            live=[p.pid for p in psutil.process_iter(['pid','cmdline']) if any(a==original_id for a in p.info['cmdline'] or [])]
            if live:state['jobs'][job['id']]='original_still_live';continue
            with waited_lock(ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock',update):
                start=wait_for_headroom(settings,update)
                logs=base/'logs'/job['id'];logs.mkdir(parents=True,exist_ok=True)
                command=[str(ROOT/case['python']),'-u','-m','scripts.flow_matching.run_resource_recovery_job','--queue',str(path),'--job-id',job['id']]
                env={k.upper():v for k,v in os.environ.items()};env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                with (logs/'stdout.log').open('a',encoding='utf-8') as out,(logs/'stderr.log').open('a',encoding='utf-8') as err:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
                    update({'state':'running','active_job':job['id'],'active_pid':child.pid})
                    code,usage=guarded_wait(child,settings,update)
                output.mkdir(parents=True,exist_ok=True)
                write_json(output/'resource_usage.json',dict(usage,start_resources=start))
                if code or not final.exists():
                    write_json(output/'failure.json',{'status':'failed_preserved','exit_code':code,'original_id':original_id,'resources':usage})
                    state['jobs'][job['id']]='failed_preserved'
                else:
                    receipt=json.loads(final.read_text())
                    if case['kind'] in ('strict','industrial'):assert complete_result(ROOT,job)
                    else:verify_artifacts(receipt)
                    state['jobs'][job['id']]='completed'
                update({'active_job':None,'active_pid':None})
        update({'state':'completed' if all(state['jobs'].get(c['job']['id'])=='completed' for c in queue['jobs']) else 'completed_with_unresolved_jobs'})


if __name__=='__main__':main()
