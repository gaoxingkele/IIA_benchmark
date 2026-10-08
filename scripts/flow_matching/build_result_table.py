"""Snapshot all registered FM research results without training or test selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t
from scripts.flow_matching.compare_results import compare
from scripts.literature.build_flow_matching_ara import ROOT, write
from scripts.literature.import_flow_matching_archives import contained


class Snapshot:
    def __init__(self,root):self.root=root; self.values={}; self.sources=[]
    def read(self,path):
        path=Path(path).as_posix()
        if path not in self.values:
            raw=contained(self.root,path).read_bytes()
            self.values[path]=json.loads(raw.decode('utf-8-sig'))
            self.sources.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        return self.values[path]
    def sha(self,path):
        self.read(path)
        return next(r['sha256'] for r in self.sources if r['path']==Path(path).as_posix())


def summarize(values):
    if not values:return {'local_mean':None,'local_std':None,'local_se':None,'ci95_low':None,'ci95_high':None}
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in values):raise ValueError('Nonfinite metric')
    mean=statistics.mean(values); std=statistics.stdev(values) if len(values)>1 else None
    se=std/math.sqrt(len(values)) if std is not None else None
    margin=float(t.ppf(.975,len(values)-1))*se if se is not None else None
    return {'local_mean':mean,'local_std':std,'local_se':se,
            'ci95_low':mean-margin if margin is not None else None,'ci95_high':mean+margin if margin is not None else None}


def display_missing_ratio(config,family,dataset):
    # Air-36 evaluates its persisted historical mask. The CSDI zero setting
    # disables newly sampled masks and is not a measured zero missing rate.
    return None if family=='csdi_cfmi' and dataset=='air36' else config.get('missing_ratio')


def validate_result(config,expected_sha,result):
    if result.get('config_sha256')!=expected_sha:raise ValueError('Result/config checksum mismatch: '+config['id'])
    actual=result.get('config',{})
    for key in ('id','dataset','method','fold','seed','missing_ratio','missing_pattern','variant','reference_id','num_samples'):
        if key in config and actual.get(key)!=config[key]:raise ValueError('Result identity mismatch: '+key)
    if config.get('diagnostic') or config.get('limit_test_cases') or actual.get('diagnostic') or actual.get('limit_test_cases') or result.get('diagnostic'):
        raise ValueError('Diagnostic result cannot enter a formal group')


def reference_for(snapshot,config,family,method,formal,settings):
    if family=='csdi_cfmi':
        refs=[r for r in formal['references'] if r['id']==config['reference_id']]
        return {r['metric']:r for r in refs}
    if family=='transfer':return {}
    if family=='grin':
        book=snapshot.read(settings['grin_references'])
        return {metric:{'mean':r['mean'],'reported_spread':r['reported_plus_minus'],
            'source':settings['grin_references'],'citation':book.get('citation'),'metric_scale':'original Air physical scale'}
            for metric,r in book['datasets'][config['dataset']].items()}
    if family=='saits_physio':
        book=snapshot.read(settings['saits_references']); point=book['datasets']['physio'][method]
        source=settings['saits_references']
    else:
        book=snapshot.read(settings['saits_timeseries_references'])
        point=book['datasets'][config['dataset']][str(config['missing_ratio'])][method]
        source=settings['saits_timeseries_references']
    return {metric:{'mean':value,'source':source,'citation':book.get('citation'),
            'metric_scale':'author standardized variables; MRE fraction'} for metric,value in point.items()}


def build(root,settings):
    start=datetime.now(timezone.utc).isoformat(); snapshot=Snapshot(root)
    formal=snapshot.read(settings['formal_references']); ara=snapshot.read(settings['ara_config'])
    groups={};jobs=[];run_details=[]
    for queue_spec in settings['queues']:
        queue=snapshot.read(queue_spec['path'])
        for job in queue['jobs']:
            cfg=snapshot.read(job['config']);sha=snapshot.sha(job['config'])
            if sha!=job['config_sha256']:raise ValueError('Frozen job changed: '+job['config'])
            method=settings['runner_methods'][Path(job['runner']).name]
            if method=='from_config':method=cfg['method']
            family=queue_spec['family']; reference_id=cfg.get('reference_id')
            dataset=cfg.get('dataset')
            if not dataset:dataset='physionet2012' if reference_id and 'physio' in reference_id else 'air36' if reference_id and 'air36' in reference_id else 'physio'
            track=cfg.get('variant') if queue_spec['track']=='config_variant' else queue_spec['track']
            ratio=display_missing_ratio(cfg,family,dataset);pattern=cfg.get('missing_pattern','historical_air36' if dataset=='air36' and family=='csdi_cfmi' else 'paper_mask')
            key=(method,dataset,track,pattern,ratio)
            if key not in groups:
                refs=reference_for(snapshot,cfg,family,method,formal,settings)
                metrics=settings['metric_families']['transfer' if family=='transfer' else method]
                groups[key]={'cfg':cfg,'missing_ratio':ratio,'family':family,'refs':refs,'metrics':metrics,'records':[], 'expected':[], 'sources':[]}
            group=groups[key];repetition=cfg.get('fold',cfg.get('seed'))
            if repetition in group['expected']:raise ValueError('Duplicate group repetition: '+str(key))
            group['expected'].append(repetition)
            result_path=(Path(cfg['output_root'])/'result.json').as_posix()
            state='not_run';record=None
            if contained(root,result_path).exists():
                record=snapshot.read(result_path);validate_result(cfg,sha,record)
                state=record['status']
                if state=='completed':
                    if family=='transfer' and record.get('metric_protocol')!='transfer_ensemble_v1':raise ValueError('Transfer protocol differs')
                    group['records'].append(record);group['sources'].append(result_path)
                    for metric in group['metrics']:
                        if metric not in record['metrics']:raise ValueError('Missing declared metric: '+metric)
                        value=record['metrics'][metric];summarize([value])
                        run_details.append({'algorithm':method,'dataset':dataset,'track':track,'pattern':pattern,'missing_ratio':ratio,
                            'configured_missing_ratio':cfg.get('missing_ratio'),
                            'fold':cfg.get('fold'),'seed':cfg.get('seed'),'metric':metric,'value':value,'run_id':cfg['id'],
                            'training_seconds':record.get('training_seconds'),'evaluation_seconds':record.get('evaluation_seconds'),
                            'eval_count':record.get('eval_count'),'test_cases':record.get('test_cases'),'num_samples':cfg.get('num_samples'),
                            'epochs':cfg.get('epochs'),'config':job['config'],'config_sha256':sha,'result':result_path,
                            'result_sha256':snapshot.sha(result_path),'processed_sha256':record.get('processed_sha256',cfg.get('processed_sha256')),
                            'checkpoint_sha256':record.get('checkpoint_sha256'),'environment':record.get('environment')})
            jobs.append({'id':cfg['id'],'algorithm':method,'dataset':dataset,'track':track,'pattern':pattern,'missing_ratio':ratio,
                'configured_missing_ratio':cfg.get('missing_ratio'),
                'fold':cfg.get('fold'),'seed':cfg.get('seed'),'status':state,'config':job['config'],'config_sha256':sha,
                'result_path':result_path,'queue':queue_spec['path'],'runner':job['runner']})
    rows=[]
    for key,group in groups.items():
        method,dataset,track,pattern,ratio=key; records=group['records'];cfg=group['cfg']
        for metric in group['metrics']:
            ref=group['refs'].get(metric); values=[r['metrics'][metric] for r in records]
            stats=summarize(values); observed=[r['config'].get('fold',r['config'].get('seed')) for r in records]
            verdict='not_run' if not records else 'incomplete_repeats' if len(records)!=len(group['expected']) else 'no_paper_reference'
            reason='缺少结果文件' if not records else '尚未完成全部预定折/种子' if verdict=='incomplete_repeats' else '该指标未登记论文参考值'
            if ref and group['family']=='csdi_cfmi':
                outcome=compare(records,ref); verdict=outcome['verdict'];reason=outcome.get('reason',outcome.get('boundary',''))
            elif ref and records and len(records)==len(group['expected']):
                verdict='numeric_comparison_only_protocol_alignment_unverified';reason='缺少完整历史协议或论文误差定义，不判定统计等效'
            if group['family']=='transfer':reason='新数据插补迁移；不属于异常检测或原论文复现'
            paper=ref.get('mean') if ref else None
            rows.append({'algorithm':method,'dataset':dataset,'dataset_label':settings['dataset_labels'].get(dataset,dataset),
                'task':'imputation','track':track,'pattern':pattern,'missing_ratio':ratio,'metric':metric,'direction':'lower',
                **stats,'completed_repeats':len(values),'required_repeats':len(group['expected']),
                'completed_fold_or_seed':observed,'required_fold_or_seed':group['expected'],'paper_mean':paper,
                'paper_se':ref.get('standard_error') if ref else None,'paper_reported_spread':ref.get('reported_spread') if ref else None,
                'signed_delta':stats['local_mean']-paper if stats['local_mean'] is not None and paper is not None else None,
                'relative_delta':(stats['local_mean']-paper)/paper if stats['local_mean'] is not None and paper else None,
                'comparison_band':outcome.get('comparison_band') if ref and group['family']=='csdi_cfmi' else None,
                'verdict':verdict,'reason':reason,'reference':ref,
                'uncertainty_unit':'fold' if 'fold' in cfg else 'model_seed','metric_scale':ref.get('metric_scale') if ref else ('normalized PhysioNet' if 'physio' in dataset else 'see run protocol'),
                'result_paths':group['sources']})
    claims=[];scope=[]
    reference_settings={group['cfg']['reference_id']:group['missing_ratio']
        for group in groups.values() if group['cfg'].get('reference_id')}
    for paper in ara['papers']:
        for index,point in enumerate(paper['reported_results']):
            claims.append({'paper_id':paper['id'],'algorithm':point.get('method_row',paper['id']),
                'dataset':point['dataset'],'metric':point['metric'],'paper_value':point.get('value',point.get('mean')),
                'reference_id':point.get('id'),'missing_ratio':point.get('missing_ratio',reference_settings.get(point.get('id'))),
                'paper_se':point.get('standard_error'),'paper_std':point.get('standard_deviation'),
                'paper_runs':point.get('runs'),'page':point.get('pdf_page'),'table':str(point.get('table')),
                'protocol':point.get('protocol','see original paper'),'status':'author_reported',
                'local_value':None,'citation':paper['citation'],'source_sha256':paper['primary_source_sha256'],
                'ara_record':index,'verification':point.get('verification'),'boundary':'论文值；本地原协议结果见正式结果页，不自动绑定TAB'})
        for dataset in paper['original_datasets']:
            mapped=settings.get('dataset_aliases',{}).get(dataset,dataset)
            local=[r for r in rows if r['algorithm']==paper['id'] and (r['dataset']==mapped or mapped=='physionet2012' and r['dataset']=='physio')]
            scope.append({'paper_id':paper['id'],'task':paper['task'],'dataset_description':dataset,
                'registered_local_metric_rows':len(local),'numeric_local_metric_rows':sum(r['local_mean'] is not None for r in local),
                'status':'formal_result_registered' if any(r['local_mean'] is not None for r in local) else 'formal_result_not_established_for_this_source_description',
                'source_sha256':paper['primary_source_sha256'],'citation':paper['citation'],
                'reproduction_gap':paper['reproduction_gap']})
    preflight=snapshot.read(settings['preflight_report']); diagnostics=[]
    for path in settings.get('diagnostic_results',[]):
        if not contained(root,path).exists():continue
        result=snapshot.read(path)
        if not result.get('config',{}).get('diagnostic'):raise ValueError('Expected diagnostic source')
        for metric,value in result.get('metrics',{}).items():
            diagnostics.append({'run_id':result['id'],'task':'imputation','dataset':'PhysioNet2012','metric':metric,'value':value,
                'status':'diagnostic_only','result':path,'boundary':'预训练诊断；不加入正式算法比较'})
    summary={'registered_jobs':len(jobs),'completed_jobs':sum(j['status']=='completed' for j in jobs),
        'job_status_counts':dict(Counter(j['status'] for j in jobs)), 'formal_setting_groups':len(groups),
        'aggregate_metric_rows':len(rows),'numeric_aggregate_metric_rows':sum(r['local_mean'] is not None for r in rows),
        'per_run_metric_rows':len(run_details),'verdict_counts':dict(Counter(r['verdict'] for r in rows)),
        'ara_author_claim_records':len(claims),'original_dataset_description_records':len(scope),
        'preflight_profiles':len(preflight['records']),'diagnostic_metric_rows':len(diagnostics),
        'formal_anomaly_detection_result_rows':0}
    return {'schema_version':1,'capture_started_utc':start,'capture_finished_utc':datetime.now(timezone.utc).isoformat(),
        'summary':summary,'formal_results':rows,'per_run_metrics':run_details,'jobs':jobs,'author_claims':claims,
        'original_dataset_scope':scope,'preflights':preflight['records'],'diagnostics':diagnostics,
        'source_snapshots':snapshot.sources,'comparison_plan':snapshot.read(settings['comparison_plan']),
        'boundary':'Complete snapshot of registered queues/metrics and existing ARA transcriptions; not every untranscribed paper table. No training, leaderboard or test tuning performed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reproducibility/fm_result_table.v1.json')
    args=parser.parse_args(); settings=json.loads(args.config.read_text(encoding='utf-8'))
    report=build(ROOT,settings);target=contained(ROOT,settings['output_directory']);target.mkdir(parents=True,exist_ok=True)
    report['report_configuration']={'path':args.config.relative_to(ROOT).as_posix(),
        'sha256':hashlib.sha256(args.config.read_bytes()).hexdigest()}
    report['report_builder']={'path':Path(__file__).relative_to(ROOT).as_posix(),
        'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(target/'experiment_results.json',report)
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
