"""Replace only exact idle LS4 controllers before defective full GPU jobs start."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.giflow_native_protocol import ROOT,sha,write_json
from scripts.flow_matching.run_ls4_cauchy_corrected_v2 import verify
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


def idle(parent,state,expected,pid_exists=psutil.pid_exists,helper_check=console_helper):
    return (parent.pid==expected['pid'] and parent.create_time()==expected['create_time']
        and parent.cmdline()==expected['command'] and state.get('pid')==parent.pid
        and state.get('state')=='waiting_full_gpu_resources'
        and not (state.get('active_pid') and pid_exists(state['active_pid']))
        and not any(not helper_check(c) for c in parent.children(recursive=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text())
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen corrected LS4 handoff changed')
    journal=ROOT/config['output']
    if journal.exists():raise FileExistsError('Preserve previous handoff; do not restart same attempt')
    # Verify both full successors and idle predecessor identities before any mutation.
    prepared=[]
    for pair in config['pairs']:
        queue=json.loads((ROOT/pair['successor_queue']).read_text());verify(queue)
        expected=pair['expected_controller'];parent=psutil.Process(expected['pid'])
        state_path=ROOT/pair['predecessor_state'];state=json.loads(state_path.read_text())
        if not idle(parent,state,expected):raise ValueError('Exact idle LS4 predecessor not verified')
        if (ROOT/queue['state_root']/'launch.json').exists():raise FileExistsError('Corrected LS4 successor already launched')
        prepared.append((pair,queue,parent,state_path))
    records=[]
    for pair,queue,parent,state_path in prepared:
        parent.suspend()
        try:
            if not idle(parent,json.loads(state_path.read_text()),pair['expected_controller']):
                raise ValueError('LS4 full model started at handoff boundary; retain it')
            parent.terminate();parent.wait(10)
        except BaseException:
            if parent.is_running():parent.resume()
            raise
        base=ROOT/queue['state_root'];base.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-X','utf8','-u','-m','scripts.flow_matching.run_ls4_cauchy_corrected_queue_v2',
            '--queue',pair['successor_queue']]
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
        with (base/'full_controller.stdout.log').open('xb') as out,(base/'full_controller.stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        record=dict(predecessor=pair['expected_controller'],terminated_only_idle_controller=True,
            successor_pid=child.pid,successor_create_time=psutil.Process(child.pid).create_time(),
            successor_command=command,successor_queue=pair['successor_queue'],
            successor_queue_sha256=sha(ROOT/pair['successor_queue']),full_scope_and_budgets_unchanged=True)
        records.append(record);write_json(base/'launch.json',record)
        write_json(journal,dict(records=records,original_full_model_interrupted=False,
            all_canonical_seed_slots_retained=40,complete_handoff=len(records)==len(prepared)))
    print(json.dumps(dict(full_successors_started=len(records),canonical_slots_retained=40)))


if __name__=='__main__':main()
