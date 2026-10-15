"""Full readonly audits for the other four original TimesNet SMAP seeds.

Reuse the unchanged v1 numerical worker in an isolated globals dictionary.
Only registration/identity/orchestration is generalized; no live module changes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as original

ROOT, read, write, sha = original.ROOT, original.read, original.write, original.sha
MODULE = 'scripts.flow_matching.run_timesnet_SMAP_remaining_full_audits_v2'
CONFIG = 'configs/reproducibility/fm_timesnet_SMAP_remaining4_full_audits.2026-10-16_v2.json'
TEST = 'tests/test_fm_timesnet_SMAP_remaining_full_audits_v2.py'
QUEUE = 'configs/experiments/fm_tsad_constant_fit_strict_baselines_queue.v2.json'
GUARD = 'configs/reproducibility/fm_saits_full_checkpoint_replay.2026-10-11_v1.json'
SEEDS = (1104, 1105, 1106, 1107)


def identity(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only the four remaining original model seeds are supported')
    return f'fm_strict_timesnet__smap__registered__seed{seed}__constant_fit_v2'


def validate_identity(config, queue, job, result, model, guard, guard_template):
    """Retain every v1 identity check; explicitly project only job seed/id.

    The immutable model template has default seed1103. The real job override,
    result and replay keep their actual seed. This private metadata projection
    never enters the numerical worker or changes stored/source records.
    """
    expected = identity(config['seed'])
    if (config['job_id'] != expected or job['id'] != expected or job['seed'] != config['seed']
            or job['parameter_overrides'] != dict(seed=config['seed'], device='cpu')
            or result['job'] != job or result['id'] != expected or guard['job_id'] != expected
            or result['experiment_sha256'] != hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()):
        raise ValueError('Actual original job id, override, result and fingerprint must agree')
    c,q,j,r,m,g,t = deepcopy((config,queue,job,result,model,guard,guard_template))
    c['job_id'] = original.CANONICAL
    j.update(id=original.CANONICAL, seed=1103, parameter_overrides=dict(seed=1103,device='cpu'))
    q['jobs'] = [j if item['id']==expected else item for item in q['jobs']]
    r.update(id=original.CANONICAL,job=j,
             experiment_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest())
    g['job_id'] = original.CANONICAL
    # The original queue already contains seed1103. Keep all 35 distinct ids
    # when mapping this one private fixture into v1's fixed selector space.
    q['jobs'] = [dict(item,id='retained_seed1103_for_v1_projection')
                 if item['id']==original.CANONICAL and item is not j else item for item in q['jobs']]
    original.validate_identity(c,q,j,r,m,g,t)


def registration():
    queue, template = read(ROOT/QUEUE), read(ROOT/GUARD)
    jobs = [next(j for j in queue['jobs'] if j['id']==identity(seed)) for seed in SEEDS]
    pins = {MODULE.replace('.','/')+'.py',TEST,QUEUE,GUARD,
        'scripts/flow_matching/audit_timesnet_constant_fit_complete_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/verify_tranad_precision_and_smd_completion_v1.py',
        'scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py'}
    for j in jobs:
        pins.update(p['path'] for p in j['frozen_files'])
        pins.update([j['model_config'],j['dataset_manifest_path']])
    return dict(schema_version=2,paper_citation='https://arxiv.org/abs/2210.02186',
        data_citation='https://github.com/decisionintelligence/TAB',
        queue=QUEUE,guard_template=GUARD,python=queue['python'],cpu_threads=2,
        settings=template['settings'],resource_lock=template['resource_lock'],
        original_scope=original.SCOPE,jobs=jobs,remaining_seeds=list(SEEDS),
        required_original_seed_cohort=[1103,*SEEDS],modes=['point','native'],required_new_audit_stages=8,
        existing_seed1103_configuration=dict(point='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.point.2026-10-14_v1.json',
            native='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.native.2026-10-14_v1.json'),
        state_root='experiments/runs/fm_timesnet_SMAP_remaining4_full_audits_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/timesnet_SMAP_remaining4_full_audits_v2',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        boundary='All4remaining original seeds,all51registered SMAP entities/all train-validation-test timestamps. '
            'Full28132633parameter checkpoint,exact original CPU float32/batch128 and bitwise score agreement. '
            'No new fitting,training seeds,threshold selection,retries or resource/tolerance changes. '
            'Original configured3epochs receipt retained;measured epoch/update trace unavailable. '
            'Registered SMAP subset is not certified canonical NASA full dataset. '
            'All35baseline jobs/291data controls/full45paper scope and fullTAB range obligations remain.')


def verify(config):
    if config != registration():
        raise ValueError('All four full original seeds and unchanged audit/resource registration required')
    return {j['id']:j for j in config['jobs']}


def source_ready(config, job):
    result = ROOT/job['output_directory']/'result.json'
    guard = ROOT/'experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/strict_baselines/receipts'/(job['id']+'.json')
    if guard.exists():
        g = read(guard)
        if g['exit_code'] != 0 or g['resource_guard_aborted'] or not g['full_registered_budget_retained']:
            raise ValueError('Original scientific training failed; preserve, do not retry')
    return result.exists() and guard.exists()


def stage_config(config, job, mode):
    if mode not in ['point','native']:
        raise ValueError('Both original full audit modes are required')
    return dict(queue=config['queue'],job_id=job['id'],seed=job['seed'],mode=mode,
        python=config['python'],cpu_threads=config['cpu_threads'],settings=config['settings'],
        resource_lock=config['resource_lock'],guard_template=config['guard_template'],
        original_scope=config['original_scope'],source_receipts=config['source_receipts'],
        terminal_receipt='experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/strict_baselines/receipts/'+job['id']+'.json',
        state_root=config['state_root']+'/attempts/'+job['id']+'/'+mode,
        output_directory=config['artifact_root']+'/'+job['id']+'/'+mode)


def binding(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen full audit source changed: '+pin['path'])
    queue = read(ROOT/config['queue'])
    job = next(j for j in queue['jobs'] if j['id']==config['job_id'])
    if not source_ready(config,job):
        raise ValueError('Complete original scientific result and successful guard required before NN imports')
    result = read(ROOT/job['output_directory']/'result.json')
    model,guard = read(ROOT/job['model_config']),read(ROOT/config['terminal_receipt'])
    validate_identity(config,queue,job,result,model,guard,read(ROOT/config['guard_template']))
    from scripts.flow_matching.run_tsad_queue import complete_result
    if not complete_result(ROOT,job) or sha(ROOT/config['queue']) != guard['queue_sha256']:
        raise ValueError('Actual original result/checkpoint/scores/queue bytes must agree')
    for pin in job['frozen_files']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Original frozen scientific source changed')
    return queue,job,result,model


def numerical_worker():
    """No numerical bytecode, source globals or shared functions are modified."""
    namespace = dict(original.worker.__globals__,binding=binding)
    return FunctionType(original.worker.__code__,namespace,original.worker.__name__,original.worker.__defaults__,original.worker.__closure__)


def worker(config, config_path):
    # Prove the full original model is ready before Torch/NumPy imports.
    binding(config)
    numerical_worker()(config,config_path)


def controller(config,jobs):
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda snapshot:None)
    api=original.resource_api()
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Remaining full TimesNet audit owner already live')
    if (base/'status.json').exists():
        owner.close();raise FileExistsError('Preserve prior terminal or interrupted audit owner')
    completed=[];failed={}
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        required_stages=8,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed=list(completed),failed_preserved=dict(failed),
            captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        for job in jobs.values():
            try:
                while not source_ready(config,job):
                    update(state='waiting_actual_original_training',job=job['id'],active_pid=None);time.sleep(10)
                for mode in config['modes']:
                    stage=stage_config(config,job,mode);attempt=ROOT/stage['state_root'];output=Path(stage['output_directory'])
                    if attempt.exists() or output.exists():raise FileExistsError('Preserve every previous full/partial audit;no automatic retry')
                    binding(stage)
                    with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
                        verify(read(ROOT/CONFIG));binding(stage)
                        attempt.mkdir(parents=True,exist_ok=False)
                        path=attempt/'frozen_stage.json';write(path,stage)
                        command=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker','--config',str(path)]
                        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                            write(attempt/'launch.json',handle);update(state='running_full_'+mode,job=job['id'],mode=mode,active_pid=child.pid)
                            code,usage=original.capped_wait(child,stage,api,update)
                        proof=output/'independent_full_saved_score_verification.json'
                        passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                        write(attempt/'receipt.json',dict(exit_code=code,passed=passed,**usage,exact_child_handle=handle,
                            proof_sha256=sha(proof) if proof.exists() else None))
                        if not passed:raise ValueError('Actual full audit failed;preserve every numerical artifact')
                    completed.append(dict(id=job['id'],mode=mode));update(state='between_full_audits',active_pid=None)
                    time.sleep(2)
            except Exception as error:
                failed[job['id']]=str(error);update(state='failed_original_seed_preserved',active_pid=None)
        update(state='all_remaining_seed_audits_terminal_other_experiments_remain',active_pid=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        path=ROOT/CONFIG
        with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(registration(),stream,indent=2);stream.write('\n')
        print(json.dumps(dict(config=CONFIG,remaining_seeds=SEEDS,full_audit_stages=8)));return
    config=read(ROOT/args.config)
    if args.worker:worker(config,args.config);return
    jobs=verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True,remaining_original_models=4,full_audit_stages=8,
            ready_original_seeds=[j['seed'] for j in jobs.values() if source_ready(config,j)],
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
            scientific_new_audits_complete=False)));return
    controller(config,jobs)


if __name__=='__main__':main()
