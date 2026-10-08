"""Repeat preserved resource incidents at the original full native batch size."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def verify(config):
    assert digest(ROOT/config['parent_queue'])==config['parent_queue_sha256']
    assert digest(ROOT/config['runtime_config'])==config['runtime_config_sha256']
    for source in config['source_receipts']:
        assert digest(ROOT/source['path'])==source['sha256'], source['path']
    for case in config['cases']:
        assert digest(ROOT/case['original_failure_path'])==case['original_failure_sha256']


def validate_proof(path, config, case):
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['queue_sha256']==config['parent_queue_sha256'] and len(report['records'])==1
    record = report['records'][0]
    parent = json.loads((ROOT/config['parent_queue']).read_text(encoding='utf-8'))
    job = next(j for j in parent['jobs'] if j['mode']=='release_checkpoint' and j['dataset']==case['dataset'])
    assert record['dataset']==case['dataset'] and record['finite_backward'] and record['deterministic_release_inference']
    assert record['native_batch_shape'][:2]==[job['train_parameters']['batch_size'],job['model_parameters']['seq_len']]
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'native_batch_shape':record['native_batch_shape']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    path=ROOT/args.config
    config=json.loads(path.read_text(encoding='utf-8'))
    verify(config)
    if args.verify_only:
        print(json.dumps({'verified_cases':len(config['cases']),'source_files':len(config['source_receipts'])}))
        return
    runtime=json.loads((ROOT/config['runtime_config']).read_text(encoding='utf-8'))
    _, namespace, _=load_controller(ROOT,runtime,'preflight')
    base=ROOT/config['output_root']
    lock=namespace['exclusive_lock'](base/'worker.lock')
    if lock is None: raise RuntimeError('Native retry controller already live')
    state={'pid':os.getpid(),'config_sha256':digest(path),'active_case':None,'active_pid':None,'cases':{}}
    def update(values):
        state.update(values)
        namespace['write_json'](base/'status.json',state)
    for case in config['cases']:
        proof=ROOT/case['canonical_proof_path']
        output=base/case['name']
        if proof.exists():
            state['cases'][case['name']]=dict(status='completed',proof=validate_proof(proof,config,case))
            continue
        if (output/'failure.json').exists():
            state['cases'][case['name']]={'status':'failed_preserved'}
            continue
        with namespace['resource_slot'](ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock',config['settings'],update) as resources:
            # The original producer skips these preserved failures; still reject any duplicate active case.
            original=json.loads((ROOT/'experiments/runs/fm_crossad_complete_preflight_v4/status.json').read_text(encoding='utf-8'))
            if original.get('active_case')==case['name']:
                raise RuntimeError('Original native check is still active; duplicate not launched')
            if proof.exists():
                state['cases'][case['name']]=dict(status='completed',proof=validate_proof(proof,config,case))
                continue
            output.mkdir(parents=True,exist_ok=True)
            command=[str(ROOT/config['settings']['python']),'-u','-m','scripts.flow_matching.preflight_crossad_author',
                     '--queue',config['parent_queue'],'--dataset',case['dataset'],'--report-prefix','fm_crossad_complete_preflight_v4']
            if case['row'] is not None: command.extend(['--row',str(case['row'])])
            env={k.upper():v for k,v in os.environ.items()}
            env.update(CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
            flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
            with (output/'stdout.log').open('a',encoding='utf-8') as out,(output/'stderr.log').open('a',encoding='utf-8') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags)
                update({'state':'running','active_case':case['name'],'active_pid':child.pid})
                code,usage=namespace['guarded_wait'](child,config['settings'],update)
            namespace['write_json'](output/'resource_usage.json',dict(usage,start_resources=resources))
            if code or not proof.exists():
                namespace['write_json'](output/'failure.json',{'exit_code':code,'resource_usage':usage,'original_failure_preserved':case})
                state['cases'][case['name']]={'status':'failed_preserved'}
            else:
                state['cases'][case['name']]=dict(status='completed',proof=validate_proof(proof,config,case))
            update({'active_case':None,'active_pid':None})
        time.sleep(2)
    update({'state':'completed' if all(c['status']=='completed' for c in state['cases'].values()) else 'completed_with_unresolved_cases'})


if __name__=='__main__':main()
