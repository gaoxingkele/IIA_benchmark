"""Audit every canonical CFM-TS slot, resolving exact retries without double counting."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import psutil
from scipy.stats import t

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT, read, sha
from scripts.flow_matching.run_cfm_ts_original_job import complete, verify, write
from scripts.flow_matching.register_cfm_ts_exact_recovery_v1 import validate_recovery


def stats(values):
    if not values:
        return dict(n=0,mean=None,std=None,se=None,ci95_low=None,ci95_high=None)
    mean=statistics.mean(values)
    std=statistics.stdev(values) if len(values)>1 else None
    se=std/len(values)**.5 if std is not None else None
    margin=float(t.ppf(.975,len(values)-1))*se if se is not None else None
    return dict(n=len(values),mean=mean,std=std,se=se,ci95_low=mean-margin if margin is not None else None,
                ci95_high=mean+margin if margin is not None else None)


def recovery_observation(path,queue):
    state_path=ROOT/queue['state_root']/'status.json'
    state=read(state_path) if state_path.exists() else {}
    observed={'queue':path,'queue_sha256':sha(ROOT/path),'state':state,'controller_verified_live':False,'worker_verified_live':False}
    try:
        parent=psutil.Process(state['pid']);command=parent.cmdline()
        runtime=read(ROOT/command[command.index('--runtime-config')+1])
        selected=runtime['controllers'][command[command.index('--controller')+1]]
        if (any(module in command for module in ['scripts.flow_matching.run_cfm_ts_low_memory_queue_v2',
                'scripts.flow_matching.run_cfm_ts_receipt_runtime_v3']) and selected['queue']==path
                and selected['queue_sha256']==sha(ROOT/path)):
            observed.update(controller_verified_live=True,controller_command=command,controller_create_time=parent.create_time())
            if state.get('active_pid'):
                worker=psutil.Process(state['active_pid']);cmd=worker.cmdline()
                if worker.ppid()==parent.pid and 'scripts.flow_matching.run_cfm_ts_original_job' in cmd and state['active_job'] in cmd:
                    observed.update(worker_verified_live=True,worker_command=cmd,worker_create_time=worker.create_time())
    except (KeyError,ValueError,psutil.NoSuchProcess,psutil.AccessDenied):pass
    return observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',default='configs/experiments/fm_cfm_ts_original_queue.v1.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--recovery-queue',action='append',default=[])
    args=parser.parse_args(); queue=read(ROOT / args.queue); verify(queue)
    recoveries=[]
    for path in args.recovery_queue:
        recovery=read(ROOT/path);validate_recovery(queue,recovery)
        if recovery['recovery_original_queue']!=args.queue:raise ValueError('Recovery belongs to another canonical queue')
        recoveries.append((path,recovery))
    recovery_processes=[recovery_observation(path,q) for path,q in recoveries]
    target=ROOT / args.output
    if target.exists():
        raise FileExistsError('Immutable audit snapshot already exists')
    target.mkdir(parents=True)
    policy=read(ROOT / queue['protocol']); state_path=ROOT / queue['state_root']/'status.json'
    state=read(state_path) if state_path.exists() else {}
    observation={'controller_verified_live':False,'worker_verified_live':False,'state':state}
    try:
        controller=psutil.Process(state['pid'])
        command=controller.cmdline()
        direct='scripts.flow_matching.run_cfm_ts_original_queue' in command
        light='scripts.flow_matching.run_cfm_ts_light_queue_v1' in command
        if light:
            runtime=read(ROOT / command[command.index('--runtime-config')+1])
            light=runtime['queue']==args.queue and runtime['queue_sha256']==sha(ROOT / args.queue)
        if 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2' in command:
            runtime=read(ROOT/command[command.index('--runtime-config')+1])
            selected=runtime['controllers'][command[command.index('--controller')+1]]
            light=selected['queue']==args.queue and selected['queue_sha256']==sha(ROOT/args.queue)
        if direct or light:
            observation.update(controller_verified_live=True,controller_create_time=controller.create_time(),controller_command=controller.cmdline())
            if state.get('active_pid'):
                child=psutil.Process(state['active_pid'])
                if (child.ppid()==controller.pid and 'scripts.flow_matching.run_cfm_ts_original_job' in child.cmdline()
                        and state['active_job'] in child.cmdline()):
                    observation.update(worker_verified_live=True,worker_command=child.cmdline(),worker_create_time=child.create_time())
    except (KeyError,psutil.NoSuchProcess,psutil.AccessDenied):
        pass
    jobs=[]; groups=defaultdict(list)
    for original_job in queue['jobs']:
        job=original_job;done=complete(job);resolved_by=args.queue
        if not done:
            for path,recovery in recoveries:
                candidate=next((j for j in recovery['jobs'] if j.get('recovery_original_job_id')==original_job['id']),None)
                if candidate is not None and complete(candidate):
                    job=candidate;done=True;resolved_by=path;break
        result_path=ROOT / job['output_directory']/'result.json'
        result=read(result_path) if done else None
        status='completed' if done else 'running' if observation['worker_verified_live'] and state['active_job']==job['id'] else 'failed_or_partial_preserved' if (ROOT / job['output_directory']).exists() or state.get('jobs',{}).get(job['id'])=='failed_or_partial_preserved' else 'pending'
        if not done:
            for observed,(path,recovery) in zip(recovery_processes,recoveries):
                candidate=next((j for j in recovery['jobs'] if j.get('recovery_original_job_id')==original_job['id']),None)
                if candidate is not None:
                    if observed['worker_verified_live'] and observed['state']['active_job']==candidate['id']:
                        status='running_recovery';break
                    if not (ROOT/candidate['output_directory']).exists():status='pending_exact_recovery'
        record=dict(original_job,execution_id=job['id'],resolved_by_queue=resolved_by,
                    recovered_original_slot=job['id']!=original_job['id'],
                    original_output_preserved=original_job['output_directory'],
                    actual_output_directory=job['output_directory'],status=status,result_path=result_path.relative_to(ROOT).as_posix() if done else None,
                    result_sha256=sha(result_path) if done else None,metrics=result['metrics'] if done else None,
                    training_seconds=result['training_seconds'] if done else None,score_seconds=result['score_seconds'] if done else None,
                    parameter_count=result['parameter_count'] if done else None)
        jobs.append(record); groups[job['track'],job['dataset'],job['variant'],job['method'],job['epochs']].append(record)
    rows=[]
    for (track,dataset,variant,method,epochs),members in groups.items():
        ready=[r for r in members if r['status']=='completed']
        claim=next(c for c in policy['paper_claims'] if c['method']==method and c['dataset']==dataset)
        row=dict(track=track,dataset=dataset,variant=variant,method=method,epochs=epochs,
                 required_seeds=len(members),completed_seeds=len(ready),paper_MSE_mean=claim['mean'],paper_MSE_std=claim['std'],
                 paper_source_page=claim['source_page'],author_equivalence_certified=False,
                 comparison='unavailable' if not ready else 'partial_seeds_numerical_comparison_only' if len(ready)<len(members) else 'full_frozen_local_protocol_numerical_comparison_only',
                 status_counts=dict(Counter(r['status'] for r in members)))
        for metric in ['MSE','MAE','RMSE']:
            row.update({metric+'_'+key:value for key,value in stats([r['metrics'][metric] for r in ready]).items()})
        row['MSE_mean_minus_paper']=row['MSE_mean']-claim['mean'] if row['MSE_mean'] is not None and claim['mean'] is not None else None
        row['training_seconds_mean']=statistics.mean(r['training_seconds'] for r in ready) if ready else None
        row['score_seconds_mean']=statistics.mean(r['score_seconds'] for r in ready) if ready else None
        rows.append(row)
    captured=datetime.now(timezone.utc).isoformat()
    report={'captured_utc':captured,'queue':args.queue,'queue_sha256':sha(ROOT / args.queue),'job_counts':dict(Counter(r['status'] for r in jobs)),
            'registered_jobs':285,'registered_seed_cohorts':57,'registered_data_cohorts':30,'jobs':jobs,'seed_aggregates':rows,
            'process_observation':observation,'recovery_process_observations':recovery_processes,
            'recovered_canonical_slots':sum(r['recovered_original_slot'] for r in jobs),
            'explicit_interpretations':policy['explicit_interpretations'],
            'author_equivalence_certified':False,'all_direction_experiments_complete':False,
            'uncertainty_boundary':'Five paired data-generation/model seeds, with t intervals. No author-equivalence or statistical-equivalence claim; one Pendulum trajectory cannot establish new-trajectory generalization.'}
    write(target/'execution_audit.json',report)
    for name,values in [('registered_jobs',jobs),('algorithm_dataset_metrics',rows)]:
        keys=list(dict.fromkeys(k for r in values for k in r))
        with (target/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
            writer.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in values])
    lines=['# CFM-TS 原始连续时间实验', '',f'核验时间：{captured}。285项完整预算任务，30组配对原始ODE数据，57组五种子比较。', '',
           f"当前状态：{report['job_counts']}。", '',
           '这是原论文所规定系统上的完整模拟实验，区别于合成烟雾检查；原作者代码和原始随机轨迹未公开，作者等价尚未证明。正文/附录、原文/修正目标公式分别保留；未完成项为空。', '',
           '| 协议 | 数据集 | 方法/公式 | 轮数 | 完成/预定种子 | MSE 均值 | STD | 95% 下界 | 95% 上界 | 论文 MSE | 本地减论文 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.6f}'
    for r in rows:
        lines.append(f"| {r['track']} | {r['dataset']} | {r['variant']} | {r['epochs']} | {r['completed_seeds']}/{r['required_seeds']} | {fmt(r['MSE_mean'])} | {fmt(r['MSE_std'])} | {fmt(r['MSE_ci95_low'])} | {fmt(r['MSE_ci95_high'])} | {fmt(r['paper_MSE_mean'])} | {fmt(r['MSE_mean_minus_paper'])} |")
    lines+=['','## 复现边界','']+['- '+x for x in policy['explicit_interpretations']]
    (target/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':report['job_counts'],'controller_live':observation['controller_verified_live'],
                      'worker_live':observation['worker_verified_live']}))


if __name__=='__main__':
    main()
