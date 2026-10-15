"""All54 Table2 reconstruction jobs,full five-epoch heads and native replay."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from scripts.flow_matching import moment_icm_forecast_finetune_v1 as f

ROOT,read,sha,write=f.ROOT,f.read,f.sha,f.write
MODULE='scripts.flow_matching.run_moment_icm_full_forecast_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_all54_full_forecast_heads.2026-10-16_v1.json'
MODEL='configs/models/fm_moment_ICM_full_PILE_forecast_head.v1.json'
TEST='tests/test_fm_moment_icm_full_forecast_v1.py'


def registration():
    previous=read(ROOT/f.pretrained.pretrain.CONFIG);f.pretrained.pretrain.verify(previous)
    data=read(ROOT/f.inputs.CONFIG);f.inputs.verify(data)
    source=ROOT/f.SOURCE
    defaults=yaml.safe_load((source/'configs/default.yaml').read_text(encoding='utf-8'))
    recipe=dict(defaults,**yaml.safe_load((source/'configs/forecasting/linear_probing.yaml').read_text(encoding='utf-8')))
    fields=['max_epoch','train_batch_size','val_batch_size','num_workers','shuffle','pin_memory','optimizer_name',
        'init_lr','weight_decay','max_norm','pct_start','lr_scheduler_type','use_amp','torch_dtype','head_dropout','add_positional_embedding']
    expected=dict(max_epoch=5,train_batch_size=64,val_batch_size=256,num_workers=5,shuffle=True,pin_memory=True,
        optimizer_name='AdamW',init_lr=5e-5,weight_decay=0,max_norm=5,pct_start=.3,lr_scheduler_type='onecyclelr',
        use_amp=True,torch_dtype='bfloat16',head_dropout=.1,add_positional_embedding=False)
    if {k:recipe[k] for k in fields}!=expected:raise ValueError('Actual frozen related-source forecast recipe required')
    jobs=[dict(id='moment_forecast__'+branch+'__'+j['dataset']+'__h'+str(h)+'__seed13',branch=branch,
        dataset=j['dataset'],horizon=h,seed=13,epochs=5,scientific=True)
        for branch in f.BRANCHES for j in data['jobs'] for h in [96,192,384]]
    files=[MODULE.replace('.','/')+'.py','scripts/flow_matching/moment_icm_forecast_finetune_v1.py',TEST,MODEL,
        f.pretrained.pretrain.CONFIG,f.inputs.CONFIG,'scripts/flow_matching/moment_icm_svm_classification_v1.py']
    files += [str(Path(f.SOURCE)/r) for r in ['configs/default.yaml','configs/forecasting/linear_probing.yaml',
        'moment/tasks/forecast_finetune.py','moment/tasks/base.py','moment/models/moment.py',
        'moment/models/layers/embed.py','moment/utils/forecasting_metrics.py','moment/utils/utils.py','moment/data/dataloader.py']]
    return dict(schema_version=1,paper_citation='https://arxiv.org/abs/2409.13530v1',
        data_citation='https://github.com/thuml/Autoformer',model_configuration=MODEL,jobs=jobs,required_cases=54,
        source_recipe=expected,lookback=256,horizons=[96,192,384],branches=f.BRANCHES,
        both_full_pretrained_models_required=True,python=previous['python'],settings=previous['settings'],resource_lock=previous['resource_lock'],
        prediction_tolerance=dict(rtol=2e-6,atol=2e-6),metric_tolerance=dict(rtol=2e-6,atol=2e-6),
        conditional_uncertainty=dict(resamples=200,seed=10013,block='256+horizon_forecast_origins'),
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_moment_icm_full54_forecast_heads_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_full54_forecast_heads_v1',
        boundary='Exact three Table2 branches/all6datasets/all3horizons. Source full5epoch training,all role shuffles,5workers, '
            'AMPbf16,AdamW/OneCycle and validation-only best selection retained;256lookback and paper-derived local Tiny '
            'core are declared reconstruction choices. No 8fold/4fold identity assumed. All complete final pretrained '
            'states required;no release/random substitutes. Full frozen-backbone/head-beta audit,all window orders, '
            'full saved predictions and full native CUDA replay. All other baselines,Table1/4/5 and periodicAD remain.')


def verify(config):
    if config!=registration():raise ValueError('All54full heads and unchanged source recipe/resources required')
    return {j['id']:j for j in config['jobs']}


def ready_models(config):
    return all(f.pretrained.trained_checkpoint(config,v) is not None for v in ['independent','icm'])


def controller(config,jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();snapshot,guarded_wait=f.pretrained.pretrain.guarded_gpu_functions(api)
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](root/'controller.lock')
    if lane is None:raise RuntimeError('Full forecast owner already live')
    completed=[];failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_cases=54,
        all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw)
        state.update(completed_ids=list(completed),completed_count=len(completed),failed_preserved=dict(failed),
            captured_utc=datetime.now(timezone.utc).isoformat());write(root/'status.json',state)
    def stage(job,phase):
        attempt=root/'attempts'/job['id']/phase;output=Path(config['artifact_root'])/job['id']
        proof=output/('full_forecast_training.json' if phase=='train' else 'full_native_forecast_review.json')
        if attempt.exists() or proof.exists() or (phase=='train' and output.exists()):
            raise FileExistsError('Every previous full/partial forecast attempt preserved;no implicit retry')
        attempt.mkdir(parents=True,exist_ok=False)
        command=[str(ROOT/config['python']),'-B','-X','utf8','-u','-m',MODULE,'--stage',phase,'--job-id',job['id']]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='13',CUDA_VISIBLE_DEVICES=str(config['settings']['gpu_index']),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'))
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(attempt/'launch.json',handle);update(state='running_full_forecast_'+phase,job=job['id'],active_pid=child.pid,active_create_time=handle['create_time'])
            code,usage=guarded_wait(child,config['settings'],api,update)
        passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
        write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
            proof_sha256=sha(proof) if proof.exists() else None))
        return passed
    try:
        while not ready_models(config):
            update(state='waiting_both_actual_full39479update_PILE_models',active_pid=None);time.sleep(10)
        for identity,job in jobs.items():
            try:
                if f.ready_data(job['dataset'],job['horizon']) is None:raise ValueError('Actual complete54role/horizon input proof required')
                while True:
                    gpu=snapshot(config['settings'])
                    if gpu['gpu_free_bytes']<config['settings']['minimum_gpu_free_bytes']:
                        update(gpu,state='waiting_original_full_GPU_headroom',job=identity,active_pid=None);time.sleep(10);continue
                    with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],lambda v:update(v,job=identity)):
                        if snapshot(config['settings'])['gpu_free_bytes']<config['settings']['minimum_gpu_free_bytes']:continue
                        verify(read(ROOT/CONFIG))
                        if not ready_models(config):raise ValueError('Complete scientific model prerequisites changed')
                        if not stage(job,'train') or not stage(job,'replay'):raise ValueError('Full guarded forecast stage failed;preserve')
                    completed.append(identity);break
            except Exception as error:failed[identity]=str(error)
            update(state='between_full_forecast_jobs',job=None,active_pid=None)
            time.sleep(config['settings']['controller_yield_seconds'])
        update(state='all54_forecast_attempts_terminal_other_paper_experiments_remain',active_pid=None,job=None)
    finally:lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--stage',choices=['train','replay']);parser.add_argument('--job-id');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen full forecast registration')
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(registration(),stream,indent=2);stream.write('\n')
        print(json.dumps(dict(configuration=CONFIG,cases=54,epochs_per_model=5)));return
    config=read(ROOT/CONFIG);jobs=verify(config);config['_config_path']=CONFIG
    if args.probe_only:print(json.dumps(dict(passed=True,cases=54,epochs_per_model=5,both_full_models_ready=ready_models(config),
        torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,scientific_models_completed=0)))
    elif args.stage:
        if not ready_models(config):raise ValueError('Both complete full PILE models required before numerical imports')
        job=jobs[args.job_id];output=Path(config['artifact_root'])/job['id']
        (f.worker if args.stage=='train' else f.replay_worker)(config,job,output)
    else:controller(config,jobs)


if __name__=='__main__':main()
