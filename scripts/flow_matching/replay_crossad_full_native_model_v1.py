"""Replay an entire completed native CrossAD model and its original evaluators."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import psutil

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT, read, digest
from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import capped_wait, resource_api


def compare_scores(saved, actual):
    import numpy as np
    for role in ('train','validation','test'):
        if (actual[role].shape != saved[role].shape or not np.isfinite(actual[role]).all()
                or not np.allclose(actual[role],saved[role],rtol=1e-6,atol=1e-7)):
            raise ValueError('Every original full score must replay under the existing native tolerance')


def compare_metrics(original, replay):
    import math
    if set(original) != set(replay):
        raise ValueError('Every original native metric file must be replayed')
    for name, rows in original.items():
        if len(rows)!=len(replay[name]):
            raise ValueError('Every original native metric row must be replayed')
        for a,b in zip(rows,replay[name]):
            if set(a)!=set(b) or any(not math.isclose(float(a[k]),float(b[k]),rel_tol=1e-12,abs_tol=1e-12) for k in a):
                raise ValueError('Original metric or SPOT threshold differs: '+name)


def worker(config,path):
    import numpy as np
    import pandas as pd
    import torch
    from iia_benchmark.models.fm_crossad_author_runtime import import_author, install_wide_cache
    from iia_benchmark.models.fm_crossad_memory_v2 import build_memory_model
    from scripts.flow_matching.run_crossad_author_job import native_scores
    torch.set_num_threads(config['cpu_threads'])
    queue=read(config['queue']); settings=queue['settings']
    job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    result=read(ROOT/job['output_directory']/'result.json')
    completion=read(config['independent_point_completion'])
    if not completion['passed'] or completion['job_id']!=job['id'] or result['status']!='completed':
        raise ValueError('The complete original training and every raw-aligned point must be audited first')
    if result['details']['environment']!=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__):
        raise ValueError('Preserve the exact original numerical environment')
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    target=ROOT/config['artifact_directory']
    if target.exists(): raise FileExistsError('Preserve full native model replay attempts')
    target.mkdir(parents=True)
    configs=target/'workspace/configs'/job['dataset']; configs.mkdir(parents=True)
    for name, data in [('model_configs_0.json',job['model_parameters']),('train_configs.json',job['train_parameters'])]:
        (configs/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    module=import_author(ROOT/settings['runtime_root'])
    install_wide_cache(ROOT/settings['runtime_root'],ROOT/settings['cache_root'])
    module.Exp_Anomaly_Detection._build_model=lambda self:build_memory_model(ROOT/settings['runtime_root'],job['model_parameters']).float()
    args=SimpleNamespace(configs_path=str(target/'workspace/configs'),save_path=str(target/'workspace/test_results'),
        root_path=str(ROOT/settings['raw_root']),data=job['dataset'],data_origin='DADA',use_gpu=False,
        gpu_type='cpu',use_multi_gpu=False,gpu=0,devices='0',device=torch.device('cpu'),
        metrics=settings['author_metrics'],method='spot',t=settings['spot_q'])
    exp=module.Exp_Anomaly_Detection(args,id=0)
    parameters=sum(p.numel() for p in exp.model.parameters())
    if parameters!=result['details']['parameter_count']:
        raise ValueError('The exact full original model architecture must be retained')
    source=next(item for item in result['artifacts'] if item['path'].endswith('/checkpoint.pth'))
    checkpoint=Path(exp.model_save_path)/'checkpoint.pth'; checkpoint.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/source['path'],checkpoint)
    if digest(checkpoint)!=source['sha256']:raise ValueError('Original complete best weights required')
    original_load=torch.load
    def cpu_load(*arguments,**keywords):
        keywords.setdefault('map_location','cpu');keywords.setdefault('weights_only',False)
        return original_load(*arguments,**keywords)
    torch.load=cpu_load
    exp.model.load_state_dict(torch.load(checkpoint),strict=True)
    exp.test()
    exp.evaluate_spot(t=settings['spot_q'])
    replay_dir=Path(exp.rst_save_path)
    metrics={p.stem:pd.read_csv(p).to_dict(orient='records') for p in replay_dir.glob('*.csv')}
    compare_metrics(result['metrics']['original_author'],metrics)
    train_set,_=exp._get_data('train')
    values={'train':train_set.train,'validation':train_set.val,'test':train_set.test}
    actual={role:native_scores(exp.model.eval(),v,job['model_parameters']['seq_len'],job['train_parameters']['batch_size'])
        for role,v in values.items()}
    with np.load(ROOT/job['output_directory']/'native_scores.npz',allow_pickle=False) as saved:
        compare_scores(saved,actual)
        np.savez_compressed(target/'all_native_replayed_scores.npz',**actual,
            labels=saved['labels'],source_rows=saved['source_rows'])
    if torch.cuda.is_initialized():raise ValueError('Original CPU replay must not initialize CUDA')
    replay_outputs={p.relative_to(target).as_posix():digest(p) for p in target.rglob('*') if p.is_file()}
    proof=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=path,
        configuration_sha256=digest(path),job_id=job['id'],original_seed=job['seed'],
        full_original_model_reinference_passed=True,full_model_parameters=parameters,
        full_score_points={k:len(v) for k,v in actual.items()},
        original_native_test_points=result['details']['original_scored_test_points'],
        full_original_author_metric_reexecution_passed=True,original_author_metrics=metrics,
        original_SPOT_init_model_inference_and_threshold_reexecution_passed=True,
        original_range_parameters=dict(slidingWindow=100,thre=250,version='opt'),
        original_inference_score_tolerance=dict(rtol=1e-6,atol=1e-7),
        original_metric_reexecution_tolerance=dict(rtol=1e-12,atol=1e-12),
        source_result_sha256=digest(ROOT/job['output_directory']/'result.json'),
        actual_cpu_threads=torch.get_num_threads(),cuda_context_initialized=False,
        replay_outputs=replay_outputs,additional_training_seed_slots=0,
        original_scope=config['original_scope'],required_original_main_training_seeds=6,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary='Entire original best model, all native train/validation/test scores, original nonoverlap author test, full range/VUS250thresholds100window default, SPOT init model inference and affiliation rerun. Raw labels/tail already independently audited. Original flat-data/transductive-scaler/framework/CPU boundaries retained; no entity-grouped TAB or paper equivalence certification.')
    (target/'full_native_model_replay.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(full_model_parameters=parameters,full_original_native_metric_reexecution_passed=True,
        full_score_points=proof['full_score_points'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();config=read(args.config)
    for item in config['source_receipts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen full native replay source changed')
    if args.worker:return worker(config,args.config)
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('The original full native model replay is already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):
        state.update(values);(base/'status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),
                MKL_NUM_THREADS=str(config['cpu_threads']),OPENBLAS_NUM_THREADS=str(config['cpu_threads']))
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen([config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.replay_crossad_full_native_model_v1','--config',args.config,'--worker'],
                    cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid);update(dict(active_pid=child.pid,active_create_time=handle.create_time()))
                code,usage=capped_wait(child,config,api,update)
        passed=code==0 and (ROOT/config['artifact_directory']/'full_native_model_replay.json').exists()
        receipt=dict(exit_code=code,full_native_model_replay_passed=passed,**usage)
        (base/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,**receipt))
        if not passed:raise RuntimeError('Full original replay failed; preserve the entire attempt')
    finally:owner.close()


if __name__=='__main__':main()
