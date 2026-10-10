"""Publish all captured algorithm/dataset metrics, with protocols and empty slots."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import statistics

from scripts.flow_matching.capture_result_listing_v4 import ROOT, read, sha, write, csv_write
from scripts.flow_matching.publish_algorithm_dataset_catalog import strict_metric_rows


def fmt(value):
    return '—' if value in (None, '') else f'{float(value):.6f}'


def table(headers, records):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ') if value is not None else '—'
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in records])


def validate_rows(rows):
    seen = set()
    for row in rows:
        key = tuple(str(row.get(k, '')) for k in ('task','algorithm','dataset','protocol','metric',
                                                 'epochs','pattern','missing_ratio','score_recipe'))
        if key in seen:
            raise ValueError('Duplicate algorithm/dataset/protocol metric: '+str(key))
        seen.add(key)
        n, required = int(row['n']), int(row['required_repeats'])
        missing = row['mean'] in (None, '')
        if missing != (n == 0) or not 0 <= n <= required:
            raise ValueError('Missing value or repeat count changed')
        if not missing and not math.isfinite(float(row['mean'])):
            raise ValueError('Nonfinite measured value')
        if n < 2 and any(row.get(k) not in (None, '') for k in ('std','se','ci95_low','ci95_high')):
            raise ValueError('Single-seed uncertainty manufactured')


def captured_rows(report, base, cfm, cfm_directory, grasp):
    rows = strict_metric_rows(report['strict_summary'], base+'/strict_summary.csv', report['tsad_capture_utc'])
    for record in report['imputation']['formal_results']:
        rows.append(dict(algorithm=record['algorithm'], dataset=record['dataset'], task='imputation',
            protocol=record['track'], pattern=record['pattern'], missing_ratio=record['missing_ratio'],
            metric=record['metric'], n=record['completed_repeats'], required_repeats=record['required_repeats'],
            mean=record['local_mean'], std=record.get('local_std'), se=record.get('local_se'),
            ci95_low=record.get('ci95_low'), ci95_high=record.get('ci95_high'), paper_mean=record.get('paper_mean'),
            comparison=record['verdict'], captured_utc=report['captured_utc'], source=base+'/imputation_summary.csv',
            author_equivalence_certified=False))
    for record in report['native_summary']:
        rows.append(dict(algorithm=record['recipe'], dataset=record['dataset'], task='anomaly_detection',
            protocol=record['protocol'], metric=record['metric'], n=record['n'],
            required_repeats=record['required_seeds'],
            **{k:record.get(k) for k in ('mean','std','se','ci95_low','ci95_high')},
            paper_mean=None, comparison=record['repeat_interpretation'], captured_utc=report['captured_utc'],
            source=base+'/native_summary.csv', author_equivalence_certified=False))
    rows.extend(grasp)
    for record in cfm['seed_aggregates']:
        for metric in ('MSE','MAE','RMSE'):
            rows.append(dict(algorithm='CFM-TS/'+record['variant'], dataset=record['dataset'],
                task='continuous_time_dynamics', protocol=record['track'], epochs=record['epochs'],
                metric=metric, n=record[metric+'_n'], required_repeats=record['required_seeds'],
                **{k:record[metric+'_'+k] for k in ('mean','std','se','ci95_low','ci95_high')},
                paper_mean=record['paper_MSE_mean'] if metric=='MSE' else None,
                paper_std=record['paper_MSE_std'] if metric=='MSE' else None,
                comparison=record['comparison'], captured_utc=cfm['captured_utc'],
                source=cfm_directory+'/algorithm_dataset_metrics.csv', author_equivalence_certified=False))
    return rows


def generation_rows(config):
    from scripts.flow_matching.summarize_spectral_original import aggregate
    from scripts.flow_matching.run_spectral_original_job import verify, complete
    qpath = config['generation_queue']
    queue = read(ROOT/qpath)
    verify(queue)
    groups, jobs = defaultdict(list), []
    for job in queue['jobs']:
        done = complete(job)
        path = ROOT/job['output_directory']/'result.json'
        result = read(path) if done else None
        jobs.append(dict(job, status='completed' if done else 'no_complete_result',
                         result_path=path.relative_to(ROOT).as_posix() if done else None,
                         result_sha256=sha(path) if done else None))
        groups[job['model'], job['dataset'], job['data_track']].append(result)
    # The original frozen summary specifies metric names, claims and seed budgets.
    template = read(ROOT/config['generation_template'])
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for record in template['summaries']:
        members = groups[record['model'], record['dataset'], record['data_track']]
        if len(members) != record['required_seeds']:
            raise ValueError('Original generation repeat scope changed')
        available = [r for r in members if r is not None]
        if available:
            raise ValueError('New generation results need nested evaluator recipe review before publication')
        rows.append(dict(algorithm='Spectral-author/'+record['model'], dataset=record['dataset'],
            task='unconditional_time_series_generation', protocol=record['data_track'], metric=record['metric'],
            required_repeats=len(members), **aggregate([]), paper_mean=record['paper_mean'],
            paper_spread=record['paper_spread'], comparison='no_complete_result', captured_utc=stamp,
            source=qpath, author_equivalence_certified=False))
    for name in ('table2_queue', 'table3_queue'):
        path = config[name]
        queue = read(ROOT/path)
        groups = defaultdict(list)
        for job in queue['jobs']:
            model = read(ROOT/job['model_config'])
            method = job.get('method', model.get('method', model.get('parameters', {}).get('method')))
            dataset = job.get('dataset', model.get('dataset', model.get('parameters', {}).get('dataset')))
            if method is None or dataset is None:
                raise ValueError('Original generation method/dataset identity missing')
            if (ROOT/job['output_directory']/'result.json').exists():
                raise ValueError('New Table2/3 completion requires full metric receipt verification')
            groups[method, dataset].append(job)
            jobs.append(dict(job, method=method, dataset=dataset, queue=path, status='no_complete_result'))
        for (method,dataset), members in groups.items():
            metrics = ('context_fid','discriminative','predictive') if name=='table2_queue' else ('marginal_score','clf_score','predictive_score')
            model = read(ROOT/members[0]['model_config'])
            claims = model.get('paper_claims', {})
            for metric in metrics:
                rows.append(dict(algorithm='Spectral-author/'+method, dataset=dataset,
                    task='unconditional_time_series_generation', protocol=name.removesuffix('_queue')+'_original',
                    metric=metric, n=0, required_repeats=len(members), mean=None, std=None, se=None,
                    ci95_low=None, ci95_high=None, paper_mean=claims.get(metric+'_mean'),
                    paper_spread=claims.get(metric+'_std'), comparison='no_complete_result',
                    captured_utc=stamp, source=path, author_equivalence_certified=False))
    return rows, jobs


def interactive_html(rows):
    fields = ['task','algorithm','dataset','protocol','epochs','pattern','missing_ratio','score_recipe',
              'metric','n','required_repeats','mean','std','se','ci95_low','ci95_high','paper_mean','captured_utc','source']
    payload = json.dumps([[r.get(k) for k in fields] for r in rows], ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    labels = ['任务','算法 / 消融','数据集','协议','轮数','掩码','缺失率','推断配方','指标','已完成数','预定数',
              '均值','STD','SE','95%下界','95%上界','论文值','核验时间UTC','来源']
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整算法—数据集指标</title>
<style>body{font:14px system-ui;margin:24px;color:#172c42}h1{font-size:25px}input,select{padding:9px;margin:6px 8px 6px 0}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #dbe3ea;white-space:nowrap}th{background:#eaf1f7;position:sticky;top:0}td:nth-child(n+10):nth-child(-n+17){text-align:right}tr:nth-child(even){background:#f7f9fc}.scroll{max-height:75vh;overflow:auto}a{color:#1766ac}</style>
<h1>算法—数据集完整技术指标</h1><p>全量已登记指标及空缺。概率指标0–1，误差保留原量纲；—为缺失，0为实测零值。不同协议、任务和预算分别比较。尚未全部完成。</p>
<p>验证分数99百分位、无PA的严格主表与测试标签最优阈值、作者PA、生成和插补协议单列。n=1不生成种子置信区间。</p>
<input id="query" placeholder="搜索算法、数据集、消融或协议" size="45"><select id="task"><option value="">全部任务</option></select><select id="dataset"><option value="">全部数据集</option></select><label><input type="checkbox" id="measured">仅有实测值</label><span id="count"></span><div class="scroll"><table><thead><tr>'''+''.join('<th>'+html.escape(s)+'</th>' for s in labels)+'''</tr></thead><tbody id="body"></tbody></table></div>
<script>const rows='''+payload+''';const q=document.querySelector('#query'),t=document.querySelector('#task'),d=document.querySelector('#dataset'),m=document.querySelector('#measured'),b=document.querySelector('#body');
for(const [select,index] of [[t,0],[d,2]])for(const x of [...new Set(rows.map(r=>r[index]))].sort()){const o=document.createElement('option');o.value=x;o.textContent=x;select.append(o)}
function show(){const query=q.value.toLowerCase();const selected=rows.filter(r=>(!t.value||r[0]===t.value)&&(!d.value||r[2]===d.value)&&(!m.checked||r[9]>0)&&(!query||r.join(' ').toLowerCase().includes(query)));b.replaceChildren();const fragment=document.createDocumentFragment();for(const r of selected){const tr=document.createElement('tr');for(let i=0;i<r.length;i++){const td=document.createElement('td');const v=r[i];td.textContent=v===null||v===''?'—':typeof v==='number'&&i>=11&&i<=16?v.toFixed(6):String(v);tr.append(td)}fragment.append(tr)}b.append(fragment);document.querySelector('#count').textContent=selected.length+' / '+rows.length+' 条指标'}for(const x of [q,t,d,m])x.addEventListener('input',show);show();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT/args.config)
    target = ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Published result captures are immutable')
    base = config['complete_capture']
    report = read(ROOT/base/'complete_results.json')
    validation = read(ROOT/base/'validation.json')
    for name, expected in validation['outputs'].items():
        if sha(ROOT/base/name) != expected:
            raise ValueError('Captured export changed: '+name)
    cfm = read(ROOT/config['cfm_capture']/'execution_audit.json')
    grasp = list(csv.DictReader((ROOT/base/'grasp_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='')))
    for row in grasp:
        row['source'] = base+'/grasp_algorithm_dataset_metrics.csv'
        for key in ('n','required_repeats'):
            row[key] = int(row[key])
        for key in ('mean','std','se','ci95_low','ci95_high','paper_mean'):
            row[key] = None if row.get(key,'')=='' else float(row[key])
    rows = captured_rows(report, base, cfm, config['cfm_capture'], grasp)
    generated, jobs = generation_rows(config)
    rows.extend(generated)
    validate_rows(rows)
    target.mkdir(parents=True)
    csv_write(target/'all_algorithm_dataset_metrics.csv', rows)
    csv_write(target/'original_generation_jobs.csv', jobs)
    strict = report['strict_summary']
    strict_headers = ['算法配置','数据集','完成/预定','状态','P','R','点级F1','AUROC','AP',
                      'VUS-ROC(n)','VUS-PR(n)','Affiliation F(n)','F1 STD','F1 95%CI']
    strict_values = [[r['algorithm_config'],r['dataset'],str(r['completed_seeds'])+'/'+str(r['required_seeds']),r['status'],
        *[fmt(r[k+'_mean']) for k in ('precision','recall','f1','auroc','average_precision')],
        *[fmt(r[k+'_mean'])+' ('+str(r.get(k+'_n') or 0)+')' for k in ('VUS_ROC','VUS_PR','affiliation_f')],
        fmt(r['f1_std']),fmt(r['f1_ci95_low'])+' / '+fmt(r['f1_ci95_high'])] for r in strict]
    (target/'strict_algorithm_dataset.md').write_text('# 全部严格异常检测组合\n\n验证分数第99百分位，无PA；0–1。完整TAB作者训练等价未认证。\n\n'+table(strict_headers,strict_values)+'\n', encoding='utf-8')
    headers = ['任务','算法 / 消融','数据集','协议','轮数','掩码 / 缺失率','推断配方','指标',
               '完成/预定','均值','STD','SE','95%下界','95%上界','论文值','核验时间UTC']
    values = [[r['task'],r['algorithm'],r['dataset'],r['protocol'],r.get('epochs',''),
        str(r.get('pattern',''))+' / '+str(r.get('missing_ratio','')),r.get('score_recipe',''),r['metric'],
        str(r['n'])+'/'+str(r['required_repeats']),*[fmt(r.get(k)) for k in ('mean','std','se','ci95_low','ci95_high','paper_mean')],r['captured_utc']] for r in rows]
    (target/'all_metrics.md').write_text('# 全部算法—数据集指标及空缺\n\n各协议单列，—为缺失而非零。来源与SHA见CSV和publication_validation.json。\n\n'+table(headers,values)+'\n', encoding='utf-8')
    (target/'results.html').write_text(interactive_html(rows), encoding='utf-8')
    counts = Counter(r['task'] for r in rows)
    measured = sum(int(r['n'])>0 for r in rows)
    full = sum(r['status']=='completed' for r in strict)
    partial = sum(r['status']=='partial' for r in strict)
    cfmfull = sum(g['completed_seeds']==g['required_seeds'] for g in cfm['seed_aggregates'])
    def link(path):
        import os
        return os.path.relpath(ROOT/path,target).replace('\\','/')
    grasp_audit = read(ROOT/base/'grasp_execution_audit.json')
    lines = ['# 完整算法—数据集实验结果技术指标','',
        f"发布：{datetime.now(timezone.utc).isoformat()}。各轨道独立核验时间保留在CSV；持续运行期间的不可变快照。",'',
        f"**{len(rows)}条汇总指标：{measured}条有实测值，{len(rows)-measured}条暂无完整结果。指标行数不等于训练次数。**",'',
        '[可搜索、筛选的全部指标](results.html)；[全量CSV](all_algorithm_dataset_metrics.csv)；[全量可阅读表](all_metrics.md)。','',
        '| 任务 | 汇总指标行数（含缺失） |','|---|---:|',
        *[f'| {name} | {n} |' for name,n in counts.items()], '',
        f"严格异常检测：{len(strict)}个配置—数据集组合，{full}个完成预定重复、{partial}个部分完成、{len(strict)-full-partial}个暂无完整结果；完整运行{report['summary']['complete_tsad_runs']}次。核验时间{report['tsad_capture_utc']}。",'',
        f"CFM-TS连续动力学：{cfm['job_counts']}，{cfmfull}/57个五种子组完整，原始种子槽位共285。已有83个早先完成槽的检查点重推断逐位一致；本次新完成槽未自动获得这一复核标记。",'',
        f"GRASP：{grasp_audit['completed_cohorts']}/{grasp_audit['registered_cohorts']}个全实体种子组完整，精确恢复不增加种子。只纳入全实体完成组，不从部分实体推算论文宏平均。独立数组审计仍有待执行项。",'',
        f"插补：{report['summary']['imputation']['completed_jobs']}/540项完整；MAE/RMSE/CRPS/MRE与缺失率、掩码分别列行。",'',
        '概率指标0–1，误差保留原量纲；—为缺失、0为实测零。n为每一指标的实测重复数。单种子无种子STD/区间；确定性统计重复拟合不解释成随机训练不确定性。', '',
        '严格主表：验证分数第99百分位阈值，无PA。测试标签最优阈值、作者PA、发布权重评价、历史记录与严格主表分别列出。非重叠窗口与部分作者stride=1的更新次数尚不一致；没有作者等价认证。PRONTO正常段选择使用标签，SKAB校准含异常，TEP为经典单工况整运行。', '',
        '完整表和逐次技术明细：','',
        '- [全部严格异常检测组合：P/R/F1、AUROC、AP、VUS、Affiliation、STD/CI](strict_algorithm_dataset.md)',
        f"- [逐种子三档阈值、事件召回、漏检、延迟、误报率、参数量、训练/评分耗时、CUDA峰值]({link(base+'/strict_per_seed.csv')})",
        f"- [逐实体结果]({link(base+'/strict_per_entity.csv')})",
        f"- [全部插补指标与完成状态]({link(base+'/imputation_summary.csv')})",
        f"- [CFM-TS全部57组、论文MSE及差值]({link(config['cfm_capture']+'/README.md')})",
        f"- [作者完整流程及其协议数值]({link(base+'/author_pipeline_numeric_metrics.csv')})",
        f"- [全部作者流程任务状态]({link(base+'/author_pipeline_jobs.csv')})",
        f"- [历史协议结果]({link(base+'/historical_protocol_results.csv')})",
        f"- [TAB比例与PA诊断；测试标签选比例，不用于严格泛化排名]({link(base+'/tab_ratio_diagnostic.csv')})",
        f"- [71条已核验论文报告值]({link(base+'/paper_claims.csv')})",
        f"- [45篇论文273条原始数据/任务范围与未完成义务]({link(base+'/original_dataset_scope.csv')})",
        f"- [681条方法/基线/消融库存]({link('configs/reproducibility/fm_paper_method_inventory.v1.json')})",
        '- [生成原始任务状态，包括新增Table2/3待完成项](original_generation_jobs.csv)', '',
        '表完整列出了已登记指标及其空缺。681条文献方法记录不等于681个已复现算法；未登记或未完成的原始任务继续保留在范围清单中。全部论文、全部方法和全部消融实验尚未完成。','']
    (target/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    source_paths = [args.config, base+'/validation.json',base+'/complete_results.json',base+'/capture_receipt.json',
                    base+'/grasp_execution_audit.json',base+'/grasp_algorithm_dataset_metrics.csv',
                    config['cfm_capture']+'/execution_audit.json',config['generation_template'],
                    config['generation_queue'],config['table2_queue'],config['table3_queue'],
                    'scripts/flow_matching/publish_result_listing_v4.py','tests/test_fm_result_listing_publication_v4.py']
    receipt = dict(published_utc=datetime.now(timezone.utc).isoformat(), aggregate_metric_rows=len(rows),
        measured_metric_rows=measured, missing_metric_rows=len(rows)-measured,task_rows=dict(counts),
        strict_groups=len(strict),strict_complete_groups=full,strict_partial_groups=partial,
        cfm_completed_canonical_slots=cfm['job_counts'].get('completed',0),cfm_complete_groups=cfmfull,
        grasp_complete_cohorts=grasp_audit['completed_cohorts'],all_original_experiments_complete=False,
        checks=dict(missing_not_zero=True,protocols_separate=True,no_duplicate_canonical_metric=True,
                    no_extra_recovery_seeds=True,capture_export_sha_verified=True,single_seed_uncertainty_absent=True),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in source_paths],
        outputs={p.name:sha(p) for p in target.iterdir() if p.is_file()})
    write(target/'publication_validation.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_receipts','outputs')},ensure_ascii=False))


if __name__=='__main__':
    main()
