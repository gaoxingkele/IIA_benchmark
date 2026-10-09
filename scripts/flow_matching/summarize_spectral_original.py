"""Audit complete original generation runs and nested evaluator uncertainty."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import subprocess

import psutil

from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write
from scripts.flow_matching.run_spectral_original_job import verify


def aggregate(values):
    if not values:return dict(n=0,mean=None,std=None,se=None,ci95_low=None,ci95_high=None)
    mean=statistics.mean(values);std=statistics.stdev(values) if len(values)>1 else None
    se=std/len(values)**.5 if std is not None else None
    if se is not None:
        from scipy.stats import t
        margin=float(t.ppf(.975,len(values)-1))*se
    else:margin=None
    return dict(n=len(values),mean=mean,std=std,se=se,ci95_low=mean-margin if margin is not None else None,ci95_high=mean+margin if margin is not None else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',default='configs/experiments/fm_spectral_mean_flow_original_queue.v2.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();queue=read(ROOT/args.queue);policy=verify(queue)
    target=ROOT/args.output
    if target.exists():raise FileExistsError('Preserve original generation captures')
    state_path=ROOT/queue['state_root']/'status.json';state=read(state_path) if state_path.exists() else {}
    live=False;worker=False;commands={}
    try:
        parent=psutil.Process(state['pid']);cmd=parent.cmdline()
        live='scripts.flow_matching.run_spectral_original_queue' in cmd and args.queue in cmd
        if live:commands={'parent':cmd,'parent_create_time':parent.create_time()}
        if live and state.get('active_pid'):
            child=psutil.Process(state['active_pid']);worker=(child.ppid()==parent.pid and
                'scripts.flow_matching.run_spectral_original_job' in child.cmdline() and state['active_job'] in child.cmdline())
            if worker:commands.update(child=child.cmdline(),child_create_time=child.create_time())
    except (psutil.NoSuchProcess,psutil.AccessDenied,KeyError):pass
    records=[];groups=defaultdict(list)
    for job in queue['jobs']:
        path=ROOT/job['output_directory']/'result.json';result=None
        if path.exists():
            audit=subprocess.run([str(ROOT/queue['python']),'-X','utf8','-m','scripts.flow_matching.run_spectral_original_job',
                '--queue',args.queue,'--job-id',job['id'],'--verify-result'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
            if audit.returncode==0:result=read(path)
        status='completed' if result else 'running' if worker and state['active_job']==job['id'] else 'failed_or_partial_preserved' if (ROOT/job['output_directory']).exists() else 'pending'
        record=dict(job,status=status,result_path=path.relative_to(ROOT).as_posix() if result else None,
            result_sha256=sha(path) if result else None,metrics=result['metrics'] if result else None,
            resource_metrics={k:result[k] for k in ['parameter_count','training_seconds','sampling_seconds','evaluation_seconds']} if result else None)
        records.append(record);groups[job['model'],job['dataset'],job['data_track']].append(record)
    claims=read(ROOT/'configs/reproducibility/spectral_mean_flow_table1_claims.v1.json')['claims']
    summaries=[]
    for (model,dataset,track),members in groups.items():
        done=[r for r in members if r['status']=='completed']
        for metric in ['context_fid','correlational','discriminative','predictive']:
            reference=next(r for r in claims if (r['method'],r['dataset'],r['metric'])==(model,dataset,metric))
            summary=dict(model=model,dataset=dataset,data_track=track,metric=metric,required_seeds=len(members),
                **aggregate([r['metrics'][metric]['mean'] for r in done]),paper_mean=reference['paper_mean'],
                paper_spread=reference['paper_reported_spread'],paper_spread_type=reference['spread_type'],
                uncertainty_unit='independently trained model seed; five evaluator repeats nested within each model',
                status_counts=dict(Counter(r['status'] for r in members)),author_equivalence_certified=False)
            summary['mean_minus_paper']=summary['mean']-summary['paper_mean'] if summary['mean'] is not None else None
            summaries.append(summary)
    target.mkdir(parents=True)
    report={'captured_utc':datetime.now(timezone.utc).isoformat(),'queue':args.queue,'queue_sha256':sha(ROOT/args.queue),
        'registered_jobs':len(records),'job_counts':dict(Counter(r['status'] for r in records)),
        'controller_live':live,'worker_live':worker,'process_commands':commands,'state':state,
        'jobs':records,'summaries':summaries,'remaining_original_paper_obligations':policy['remaining_paper_obligations'],
        'explicit_boundaries':policy['explicit_boundaries'],'all_original_paper_experiments_complete':False}
    write(target/'execution_audit.json',report)
    for name,rows in [('registered_jobs',records),('algorithm_dataset_metrics',summaries)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (target/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
    fmt=lambda value:'—' if value is None else f'{value:.6f}'
    lines=['# Spectral Mean Flow 原始生成实验','',f"捕获时间：{report['captured_utc']}。70项任务、14组五训练种子比较；状态：{report['job_counts']}。",'',
        '作者Table1原始六数据集、完整训练步数、完整生成数组及四个原评价器分别冻结。Sines发布代码和论文分布保留两条协议；预检不计论文成绩。', '',
        '| 方法 | 数据集 | 数据协议 | 指标 | 完成/预定训练种子 | 均值 | STD | 95%下界 | 95%上界 | 论文值 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['data_track']} | {r['metric']} | {r['n']}/{r['required_seeds']} | {fmt(r['mean'])} | {fmt(r['std'])} | {fmt(r['ci95_low'])} | {fmt(r['ci95_high'])} | {fmt(r['paper_mean'])} |")
    lines+=['','## 剩余原论文任务','']+['- '+v for v in policy['remaining_paper_obligations']]
    lines+=['','## 比较边界','']+['- '+v for v in policy['explicit_boundaries']]
    (target/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':report['job_counts'],'controller_live':live,'worker_live':worker}))


if __name__=='__main__':main()
