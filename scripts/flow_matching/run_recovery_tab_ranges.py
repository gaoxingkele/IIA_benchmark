"""Full pinned TAB metrics for exact-budget recovery runs; no extra seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    path=ROOT/args.config
    config=json.loads(path.read_text(encoding='utf-8'))
    for source in config['source_receipts']:
        assert digest(ROOT/source['path'])==source['sha256'],source['path']
    runtime=json.loads((ROOT/config['light_runtime_config']).read_text(encoding='utf-8'))
    _,namespace,_=load_controller(ROOT,runtime,'range_main')
    base=ROOT/config['output_root']
    lock=namespace['exclusive_lock'](base/'evaluation.lock')
    if lock is None:raise RuntimeError('Recovery range controller already live')
    jobs=[j for p in config['queue_paths'] for j in json.loads((ROOT/p).read_text(encoding='utf-8'))['jobs']]
    state={'pid':os.getpid(),'config_sha256':digest(path),'active_pid':None,'active_job':None,'registered_attempts':len(jobs)}
    def update(values):
        state.update(values)
        namespace['write_json'](base/'status.json',state)
    complete,failed={},{}
    while True:
        waiting=[]
        for job in jobs:
            if job['id'] in complete or job['id'] in failed:continue
            model_path=ROOT/job['output_directory']/'result.json'
            if not model_path.exists():waiting.append(job['id']);continue
            assert namespace['complete_result'](ROOT,job)
            final=base/'jobs'/job['id']/'evaluation.json'
            if final.exists():
                value=json.loads(final.read_text(encoding='utf-8'))
                expected=namespace['config_hash'](config)
                identity=__import__('hashlib').sha256((digest(model_path)+expected).encode()).hexdigest()
                assert value['evaluation_identity']==identity
                assert final.with_name('evaluation.sha256').read_text(encoding='ascii').strip()==digest(final)
                complete[job['id']]={'evaluation_path':final.relative_to(ROOT).as_posix(),'sha256':digest(final)}
                continue
            with namespace['resource_slot'](ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock',config['runtime_resources'],update):
                logs=base/'logs'/job['id'];logs.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-u','-m','scripts.flow_matching.evaluate_tab_ranges_sparse_job',
                         '--config',str(path),'--result',str(model_path)]
                env={k.upper():v for k,v in os.environ.items()}
                env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads']))
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                with (logs/'stdout.log').open('a',encoding='utf-8') as out,(logs/'stderr.log').open('a',encoding='utf-8') as err:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
                    update({'state':'evaluating','status':'evaluating','active_job':job['id'],'active_pid':child.pid})
                    code,usage=namespace['guarded_wait'](child,config['runtime_resources'],update)
                namespace['write_json'](logs/'resource_usage.json',usage)
                if code or not final.exists():failed[job['id']]={'exit_code':code,'resources':usage}
                else:
                    assert final.with_name('evaluation.sha256').read_text(encoding='ascii').strip()==digest(final)
                    complete[job['id']]={'evaluation_path':final.relative_to(ROOT).as_posix(),'sha256':digest(final)}
                update({'active_pid':None,'active_job':None,'completed':complete,'failures':failed})
            time.sleep(2)
        update({'state':'waiting_training_results' if waiting else 'completed_with_unresolved_evaluations' if failed else 'completed',
                'status':'waiting_training_results' if waiting else 'completed_with_unresolved_evaluations' if failed else 'completed',
                'completed':complete,'failures':failed,'pending_model_results':waiting})
        if not waiting:return
        time.sleep(config['poll_seconds'])


if __name__=='__main__':main()
