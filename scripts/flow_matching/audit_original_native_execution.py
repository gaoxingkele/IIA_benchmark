"""Audit actual full native training/scoring, and compare frozen primary PDF tables."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil

from scripts.flow_matching.grasp_protocol_runner import ROOT, sha, metric_records, write_json
from scripts.flow_matching.export_complete_metrics import write_csv


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extract_table(text, spec):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    start = next(i for i,line in enumerate(lines) if line.startswith('Table '+spec['table']+':'))
    records = []
    previous = start
    for model in spec['rows']:
        index = next((i for i in range(previous+1,len(lines)) if lines[i]==model), None)
        if index is None:
            raise ValueError('Primary table row absent: '+model)
        values = lines[index+1:index+1+3*len(spec['datasets'])]
        if len(values)!=3*len(spec['datasets']) or not all(re.fullmatch(r'(?:0|1)\.\d{4}',s) for s in values):
            raise ValueError('Primary table numeric boundary differs: '+model)
        for dataset, offset in zip(spec['datasets'],range(0,len(values),3)):
            for metric,value in zip(['PRC','ROC','Best_F1'],values[offset:offset+3]):
                records.append({'model':model,'dataset':dataset,'metric':metric,'paper_value':float(value),
                                'page':spec['page'],'table':spec['table'],'printed_decimal_places':4,
                                'paper_repetitions':10})
        previous=index+len(values)
    return records


def live_state(root, queue, module, worker):
    import psutil
    path=root/queue['state_root']/'status.json'
    state=read(path) if path.exists() else {}
    verified = {'state':state,'controller_verified_live':False,'worker_verified_live':False}
    for key,name,expected in [('pid','controller',module),('active_pid','worker',worker)]:
        if not state.get(key):
            continue
        try:
            process=psutil.Process(state[key]);command=process.cmdline()
            good=expected in command
            if name=='worker':
                good=good and state.get('active_job') in command
            verified[name+'_verified_live']=good
            if good:
                verified[name]={'pid':process.pid,'create_time':process.create_time(),'command':command}
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    return verified


def audit_grasp_arrays(root, queue, capture):
    import numpy as np
    import torch
    spec=importlib.util.spec_from_file_location('native_grasp_metric_audit',root/queue['native_metrics'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    verified=[]
    for job in capture['jobs']:
        if job['status']!='completed':
            continue
        result=read(root/job['result_path'])
        data=result['data_audit']
        with np.load(root/data['prepared_path'],allow_pickle=False) as prepared:
            actual_labels=prepared['labels'].copy()
        checkpoint=torch.load(root/result['artifacts'][0]['path'],map_location='cpu',weights_only=False)
        if any(checkpoint[key]!=result[key] for key in ['job_sha256','epochs_completed','optimizer_updates']):
            raise ValueError('Saved full checkpoint differs from reported training')
        if checkpoint['parameters']!=read(root/result['job']['model_config'])['parameters']:
            raise ValueError('Actual saved model parameters differ')
        if checkpoint['seed']!=job['seed'] or tuple(checkpoint['adjacency'].shape)!=(data['features'],data['features']):
            raise ValueError('Actual graph/seed coverage differs')
        for record in result['metrics']:
            with np.load(root/record['scores_path'],allow_pickle=False) as arrays:
                if not np.array_equal(arrays['labels'],actual_labels):
                    raise ValueError('Evaluated labels differ from the frozen full test set')
                if not np.array_equal(arrays['test_row_ids'],np.arange(data['test_points'])):
                    raise ValueError('Full test row order differs')
                if not np.array_equal(arrays['validation_raw_row_ids'],np.arange(data['split_cut'],data['raw_train_points'])):
                    raise ValueError('Full validation row order differs')
                if arrays['validation_scores'].shape!=(data['validation_points'],):
                    raise ValueError('Full validation scoring differs')
                recalculated=metric_records(module.basic_metricor(),arrays['labels'],arrays['test_scores'],arrays['validation_scores'])
            for protocol,values in recalculated.items():
                for key,value in values.items():
                    actual=record['metrics'][protocol][key]
                    if isinstance(value,float):
                        if not math.isclose(value,actual,rel_tol=1e-12,abs_tol=1e-12):
                            raise ValueError('Metric recomputation differs: '+key)
                    elif value!=actual:
                        raise ValueError('Metric protocol declaration differs: '+key)
        verified.append({'id':job['id'],'result_path':job['result_path'],'result_sha256':sha(root/job['result_path']),
                         'epochs_completed':result['epochs_completed'],'optimizer_updates':result['optimizer_updates'],
                         'test_points':data['test_points'],'validation_points':data['validation_points'],
                         'anomaly_points':int(actual_labels.sum()),'saved_checkpoint_binding_checked':True,
                         'full_row_and_label_identity_checked':True,'metrics_independently_recomputed':True,
                         'training_seconds':result['training_seconds'],'metrics':result['metrics']})
    return verified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    settings=read(ROOT/args.config)
    target=ROOT/settings['output_directory']
    if target.exists():
        raise FileExistsError('Preserve the existing audit capture')
    started=datetime.now(timezone.utc).isoformat()
    if sha(ROOT/settings['source_pdf'])!=settings['source_sha256']:
        raise ValueError('Primary PDF identity differs')
    cache=read(ROOT/settings['source_cache'])
    if cache['source_sha256']!=settings['source_sha256']:
        raise ValueError('Primary table extraction source differs')
    references=[]
    for spec in settings['tables']:
        references.extend(extract_table(cache['pages'][str(spec['page'])],spec))
    for row in references:
        row.update(source_sha256=settings['source_sha256'],citation=settings['source_citation'])
    from scripts.flow_matching import mtsbench_stat_protocol as stat
    from scripts.flow_matching import giflow_native_protocol as giflow
    from scripts.flow_matching.summarize_grasp_protocol_v3 import summarize
    from scripts.flow_matching.collect_native_metrics import native_tables
    stat_queue=read(ROOT/settings['statistical_queue']);stat.verify(stat_queue)
    state=live_state(ROOT,stat_queue,'scripts.flow_matching.run_mtsbench_stat_queue_v2','scripts.flow_matching.mtsbench_stat_protocol')
    native_jobs=[];results={}
    for job in stat_queue['jobs']:
        done=stat.complete(job)
        result=read(ROOT/job['output_directory']/'result.json') if done else None
        status='completed' if done else 'running' if state['worker_verified_live'] and state['state']['active_job']==job['id'] else 'partial_preserved' if (ROOT/job['output_directory']).exists() else 'pending'
        native_jobs.append({'algorithm':'grasp_mtsbench_statistical_replay','recipe':job['algorithm'],
            'dataset':job['dataset'],'seed':job['seed'],'id':job['id'],'track':job['protocol'],
            'queue':settings['statistical_queue'],'output_directory':job['output_directory'],'status':status,
            'result_sha256':sha(ROOT/job['output_directory']/'result.json') if done else None,
            'metrics':result['metrics'] if done else None})
        if done:
            results[job['id']]=result
    native_summary,numeric,entity_metrics=native_tables(ROOT,native_jobs,settings)
    grasp_queue=read(ROOT/settings['grasp_queue'])
    grasp=summarize(grasp_queue)
    full_audit=audit_grasp_arrays(ROOT,grasp_queue,grasp)
    giflow_queue=read(ROOT/settings['giflow_queue']);giflow.verify(giflow_queue)
    giflow_state=live_state(ROOT,giflow_queue,'scripts.flow_matching.run_giflow_guarded_gpu_queue_v2','scripts.flow_matching.run_giflow_native_job')
    comparisons=[]
    lookup={(r['table'],r['model'],r['dataset'],r['metric']):r for r in references}
    for row in native_summary:
        if row['mean'] is None:
            continue
        reference=lookup['1',row['recipe'],row['dataset'].upper(),row['metric']]
        comparisons.append(dict(reference,local_mean=row['mean'],local_std=row['std'],completed_repetitions=row['n'],
             required_repetitions=row['required_seeds'],signed_difference=row['mean']-reference['paper_value'],
             printed_mean_agreement=abs(row['mean']-reference['paper_value'])<=.00005,
             protocol_certified=False,boundary='Native transductive exact-current-wrapper replay; deterministic actual refits.'))
    profile_labels={'main':'TSMixer','gnn':'GNN','transformer':'Transformer'}
    for group in grasp['seed_aggregates']:
        if group['profile'] not in profile_labels:
            continue
        for metric in group['metrics'] or []:
            if metric['recipe']!={'source_samples':5,'flow_evaluations':10}:
                continue
            for key in ['AP','ROC','Best_F1']:
                reference=lookup['7',profile_labels[group['profile']],group['dataset'].upper(),'PRC' if key=='AP' else key]
                values=metric['ten_seed_metrics'][key]
                comparisons.append(dict(reference,local_mean=values['mean'],local_std=values['std'],completed_repetitions=values['n'],
                    required_repetitions=group['required_seeds'],signed_difference=values['mean']-reference['paper_value'],
                    printed_mean_agreement=abs(values['mean']-reference['paper_value'])<=.00005,
                    protocol_certified=False,boundary='Paper-derived architecture; incomplete repeats cannot establish ten-seed paper agreement.'))
    report={'capture_started_utc':started,'capture_finished_utc':datetime.now(timezone.utc).isoformat(),
        'configuration':args.config,'configuration_sha256':sha(ROOT/args.config),
        'statistical_queue_sha256':sha(ROOT/settings['statistical_queue']),
        'statistical_status_counts':dict(Counter(j['status'] for j in native_jobs)),
        'statistical_jobs':native_jobs,'statistical_summary':native_summary,'statistical_process_audit':state,
        'grasp_full_protocol':grasp,'grasp_full_result_audit':full_audit,
        'giflow_queue_sha256':sha(ROOT/settings['giflow_queue']),'giflow_process_audit':giflow_state,
        'paper_comparison':comparisons,'primary_table_records':len(references),
        'all_experiments_complete':False,'boundary':settings['boundary']}
    target.mkdir(parents=True)
    write_json(target/'execution_audit.json',report)
    write_json(target/'statistical_result_snapshots.json',results)
    write_json(target/'grasp_progress_snapshot.json',grasp)
    shutil.copy2(ROOT/giflow_queue['state_root']/'transition.json',target/'giflow_scheduler_transition.json')
    for name,rows in [('paper_references',references),('paper_comparison',comparisons),('statistical_summary',native_summary),
                      ('statistical_per_entity',entity_metrics),('statistical_numeric_metrics',numeric)]:
        headers=list(dict.fromkeys(k for r in rows for k in r))
        write_csv(target/(name+'.csv'),[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows],headers)
    lines=['# 原生实验执行与论文数值核验', '',f"核验区间：{started} 至 {report['capture_finished_utc']}（UTC）。",'',
        f"HBOS/COPOD 完整原生任务状态：{report['statistical_status_counts']}，原登记共80项。所有已完成实体、原测试点数、标签数、有限分数、原数据和模型/分数哈希以及实体宏平均已核验。",'',
        f"GRASP 完整训练状态：{grasp['entity_job_counts']}，原登记9120个实体任务、480个全实体组；已完成{grasp['completed_cohorts']}组。",'',
        '完整GRASP产物另外核验保存的1500轮检查点、实际更新数、全验证/测试行身份与标签，并从保存的分数独立重算指标。单种子不能证明与论文十种子均值一致。', '',
        '| 方法 | 数据集 | 指标 | 本地均值 | 论文均值 | 差值 | 完成/预定重复 |', '|---|---|---|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['metric']} | {row['local_mean']:.6f} | {row['paper_value']:.4f} | {row['signed_difference']:+.6f} | {row['completed_repetitions']}/{row['required_repetitions']} |")
    lines += ['', '测试特征拟合的统计方法、测试最优 Best-F1 和附加验证阈值对照分别解释。论文数字来自同一已核验PDF的表1（第8页）及架构表7（第25页），保留全部204个表值和来源哈希；四位小数吻合不是作者协议等价认证。', '',
        'GiFlow 的140个任务、原输出路径与训练预算保持不变。原等待控制器经过PID/创建时间/命令/无模型子进程核验后切换到显卡资源受控队列，与完整GRASP共用CUDA训练锁。原SAITS/GRIN任务和活跃模型未停止。', '',
        f"GiFlow控制器已核验存活：{giflow_state['controller_verified_live']}；当前状态：{giflow_state['state'].get('state')}。资源达到原CPU/commit门限且取得共享GPU锁后执行原始完整任务，不用缩小batch或训练轮数。", '',
        '[完整执行审计](execution_audit.json)；[完整论文表值](paper_references.csv)；[逐项数值比较](paper_comparison.csv)；[统计方法逐实体指标](statistical_per_entity.csv)。', '',settings['boundary'], '']
    (target/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'statistical_counts':report['statistical_status_counts'],'grasp_counts':grasp['entity_job_counts'],
                      'completed_grasp_cohorts':grasp['completed_cohorts'],'giflow_live':giflow_state['controller_verified_live']}))


if __name__=='__main__':
    main()
