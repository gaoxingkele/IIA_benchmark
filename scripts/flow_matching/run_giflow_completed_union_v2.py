"""Whole-array verification of all four completed slots within original140.

Old three-model evidence stays immutable. No model/training is rerun; every
original/replayed batch, checkpoint and data identity is still audited.
"""
import argparse
import ast
from datetime import datetime,timezone
import os
import subprocess
import sys
import psutil
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,write
from scripts.flow_matching.giflow_native_protocol import sha


def verify_ast():
    tree=ast.parse((ROOT/'scripts/flow_matching/audit_giflow_complete_replay_union_v1.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
    changes={};expressions={
        'completed_canonical_models':'len(completed)',
        'full_scoring_batches':"sum(r['full_test_batches'] for r in records)",
        'full_test_window_evaluations':"sum(r['full_test_windows'] for r in records)",
        'full_native_forward_calls':"sum(r['full_native_forward_calls'] for r in records)",
        'released_training_seeds_completed':"sum(j['track']=='released_mirror' for j in completed)",
        'corrected_training_seeds_completed':"sum(j['track']=='reviewed_corrected' for j in completed)"}
    returned=next(n.value for n in function.body if isinstance(n,ast.Return))
    report_keywords={id(n) for n in returned.keywords}
    for node in ast.walk(function):
        if isinstance(node,ast.Compare) and ast.unparse(node)=='len(completed) != 3':
            node.comparators[0]=ast.parse("config['expected_completed_models']",mode='eval').body;changes['scope']=1
        if isinstance(node,ast.keyword) and id(node) in report_keywords and node.arg in expressions:
            node.value=ast.parse(expressions[node.arg],mode='eval').body;changes[node.arg]=1
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            if node.value=='Frozen three completed models within the unchanged140 original slots required':
                node.value='Frozen four completed models within the unchanged140 original slots required';changes['message']=1
            if node.value.startswith('All three previously completed original canonical models;42'):
                node.value=node.value.replace('All three previously completed original canonical models;42',
                    'All four previously completed original canonical models;56');changes['boundary']=1
    if set(changes)!={'scope','message','boundary',*expressions}:raise ValueError('Only snapshot counts and truthful coverage metadata may change')
    return ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))


def worker(config,path):
    from scripts.flow_matching import audit_giflow_complete_replay_union_v1 as original
    from scripts.flow_matching.summarize_giflow_canonical_v1 import aggregate
    if config['expected_completed_models']!=4 or len(config['models'])!=4:
        raise ValueError('Four exact completed model identities within whole140 required')
    namespace=dict(original.__dict__);exec(compile(verify_ast(),__file__,'exec'),namespace)
    proof=namespace['verify'](config)
    if {r['job_id'] for r in proof['records']}!=set(config['expected_completed_ids']):
        raise ValueError('Frozen entire completed scope cannot be replaced by another four slots')
    groups={}
    for track in ('released_mirror','reviewed_corrected'):
        rows=[r for r in proof['records'] if r['track']==track]
        groups[track]=dict(required_seeds=5,completed_seeds=sorted(r['seed'] for r in rows),
            uncertainty={key:aggregate([r['original_metrics'][key] for r in rows]) for key in rows[0]['original_metrics']})
    proof.update(configuration=path,configuration_sha256=sha(ROOT/path),captured_utc=datetime.now(timezone.utc).isoformat(),
        original_scope=config['original_scope'],canonical_pending_training_slots=136,
        actual_new_training_seed_slots=0,no_model_reexecution=True,partial_seed_group_statistics=groups,
        cuda_context_initialized=False)
    if 'torch' in sys.modules:raise ValueError('Array-only review must not load the model backend or CUDA')
    output=ROOT/config['output_proof']
    if output.exists():raise FileExistsError('Keep every complete or failed verification attempt')
    write(output,proof)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args();config=read(ROOT/args.config)
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full completed-model union source changed')
    if args.worker:return worker(config,args.config)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    from scripts.flow_matching.replay_crossad_full_native_model_v1 import capped_wait
    api=resource_api();api['resource_snapshot']=resource_snapshot
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'stdout.log').exists():raise FileExistsError('Preserve every existing complete-array review')
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('This complete union already has a live owner')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[sys.executable,'-X','utf8','-u','-m','scripts.flow_matching.run_giflow_completed_union_v2','--config',args.config,'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                p=psutil.Process(child.pid);update(dict(state='verifying_all56_original_and_replayed_batches',
                    active_pid=p.pid,active_create_time=p.create_time(),actual_command=p.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        passed=code==0 and not usage['resource_guard_aborted'] and (ROOT/config['output_proof']).exists()
        write(base/'receipt.json',dict(exit_code=code,all_completed_model_arrays_verified=passed,**usage))
        update(dict(state='completed' if passed else 'failed_attempt_preserved',active_pid=None))
        if not passed:raise RuntimeError('Preserve the failed whole-array verification without silent retries')
    finally:owner.close()


if __name__=='__main__':main()
