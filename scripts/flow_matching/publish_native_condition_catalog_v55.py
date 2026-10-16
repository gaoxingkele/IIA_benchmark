"""Preserve the complete legacy catalog and append all139/all70 native cases.

Case-level checkpoint results are distinct from training-seed aggregates.
Only a frozen set of fully guarded, independently verified cases has numbers.
No training/inference is performed; no old numerical row is rewritten.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import statistics

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha

CONFIG = 'configs/reproducibility/fm_native_condition_catalog.2026-10-16_v55.json'
MODULE = 'scripts.flow_matching.publish_native_condition_catalog_v55'
TEST = 'tests/test_fm_native_condition_catalog_v55.py'


def blank(headers):
    row = dict.fromkeys(headers, '')
    row.update(n='0', required_repeats='1', author_equivalence_certified='False', strict_TAB_result='False')
    return row


def key(row):
    return tuple(row.get(k, '') for k in ['algorithm', 'dataset', 'protocol', 'metric',
                 'pattern', 'missing_ratio', 'score_recipe'])


def cross_rows(headers, jobs, reports, captured, config):
    rows = []
    for job in jobs:
        identity = job['canonical_CPU_condition_reference']
        if job['reconstructed_ablation']:
            model = read(ROOT/job['model_config'])
            algorithm = 'CrossAD/reconstructed_row'+str(model['parameters']['row'])
        elif job['mode'] == 'release_checkpoint':
            algorithm = 'CrossAD/released_checkpoint'
        else:
            algorithm = 'CrossAD/full_model'
        report = reports.get(identity)
        for ratio in ['0.1', '1', '5']:
            item = report['independent_strict_metrics'][ratio] if report else None
            for metric, field in [('precision','precision'), ('recall','recall'), ('F1','f1'),
                                  ('AUROC','auroc'), ('AP','average_precision')]:
                row = blank(headers)
                row.update(algorithm=algorithm, dataset=job['dataset'], task='anomaly_detection',
                    protocol=f'original_CUDA_FP32_full139_v2/validation_{ratio}percent_no_PA/{identity}',
                    metric=metric, metric_unit='fraction', captured_utc=captured,
                    source=str(Path(config['cross_independent_artifact_root'])/'conditions'/(identity+'.json'))
                        if report else config['cross_queue'],
                    score_recipe=json.dumps(dict(canonical_id=identity, training_or_eval_seed=job['seed'],
                        mode=job['mode'], reconstructed_ablation=job['reconstructed_ablation']), sort_keys=True),
                    comparison='full_condition_verified_one_checkpoint_not_a_training_seed_cohort' if report
                        else 'pending_full_native_source_replay_and_independent_metric_guards',
                    selection_boundary='Frozen validation score percentile only; strict >; no point adjustment; no test-label selection.',
                    uncertainty_scope='One fixed checkpoint per case; no between-training-seed aggregation/SD/SE/CI. '
                        'Required_repeats=1 denotes one complete case only; original primary cohorts still require five seeds. '
                        'Release-checkpoint evaluation is not a newly trained seed.')
                if item:
                    row.update(n='1', mean=repr(item['micro'][field]))
                rows.append(row)
    return rows


def spectral_rows(headers, cases, reports, captured, config):
    rows=[]
    for identity, case in cases.items():
        job=case['job']; report=reports.get(identity)
        for metric in ['context_fid','correlational','discriminative','predictive']:
            row=blank(headers)
            row.update(algorithm='Spectral_original/'+job['model'], dataset=job['dataset'], task='generation',
                protocol='full_generation_and_original_evaluators_reexecution_v1/'+identity,
                metric=metric, captured_utc=captured, metric_unit='original_evaluator_scale',
                source=str(Path(config['spectral_artifact_root'])/identity/'full_generation_and_evaluator_reexecution.json')
                    if report else config['spectral_configuration'],
                score_recipe=json.dumps(dict(canonical_id=identity, training_seed=job['seed'],
                    data_track=job['data_track'], full_sampling_calls=2, runs_per_original_evaluator=5), sort_keys=True),
                comparison='two_full_samplers_and_all_original_evaluators_verified_no_new_generator_training' if report
                    else 'pending_two_full_samplers_and_all_original_evaluator_guards',
                evaluator_repeats='5' if report else '0',
                selection_boundary='Full frozen original generation/reference; two equal new samplings with RNG restoration; '
                    'sampling seed=training seed+1000000. Historical sampling RNG unknown.',
                uncertainty_scope='One trained generator per case. Five evaluator fits are within-model repeats; '
                    'their SD/SE/t95 are in evaluator_* only. Training-seed std/SE/CI remain blank. '
                    'Original five-seed cohorts require all five trained generators; zero new generator updates.')
            if report:
                item=report['metrics'][metric]
                row.update(n='1', mean=repr(item['mean']), evaluator_std=repr(item['std']),
                    evaluator_se=repr(item['se']), evaluator_ci95=json.dumps([item['ci95_low'],item['ci95_high']]))
            rows.append(row)
    return rows


def validate(before, after, cross_complete, spectral_complete):
    if len(before)!=8626 or len(after)!=10991 or after[:len(before)]!=before:
        raise ValueError('Preserve every original8626 row and append exactly all2365 native metric cells')
    extra=after[len(before):]
    if len({key(r) for r in extra})!=len(extra) or {key(r) for r in before}&{key(r) for r in extra}:
        raise ValueError('New condition-specific tracks must be unique and separate from old tracks')
    cross=[r for r in extra if r['task']=='anomaly_detection']
    spectral=[r for r in extra if r['task']=='generation']
    if len(cross)!=139*3*5 or len(spectral)!=70*4:
        raise ValueError('Every original139/70 condition must have all required metric slots')
    if sum(bool(r['mean']) for r in cross)!=15*cross_complete or sum(bool(r['mean']) for r in spectral)!=4*spectral_complete:
        raise ValueError('Measured cells must match exact frozen full guarded condition sets')
    for row in extra:
        if (row['author_equivalence_certified']!='False' or row['strict_TAB_result']!='False'
                or row['required_repeats']!='1' or any(row[k] for k in ['std','se','ci95_low','ci95_high','paper_mean'])):
            raise ValueError('Never invent author/TAB equivalence, paper targets or training-seed uncertainty')
        if row['mean']:
            if row['n']!='1' or not math.isfinite(float(row['mean'])):
                raise ValueError('One complete finite checkpoint case required')
        elif row['n']!='0' or any(row[k] for k in ['evaluator_std','evaluator_se','evaluator_ci95']):
            raise ValueError('Pending cells cannot inherit numbers or fake uncertainty')


def independent_evidence(config):
    from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
    from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
    from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
    from scripts.flow_matching import reexecute_spectral_full70_generation_evaluators_v1 as spectral
    cc=read(ROOT/cold.CONFIG); original,queue=cold.verify(cc)
    mc=read(ROOT/metrics.CONFIG); _,upstream=metrics.verify(mc)
    cb=read(ROOT/config['cross_frozen_bindings'])['records']
    if set(cb)!=set(config['completed_cross_ids']):
        raise ValueError('Exact frozen CrossAD condition roster required')
    proofs, cross_reports = {}, {}
    for identity in config['completed_cross_ids']:
        binding=cb[identity]
        for field in ['full_array_proof','independent_metric_report']:
            if sha(binding[field+'_path'])!=binding[field+'_sha256']:
                raise ValueError('Frozen full CrossAD proof changed')
        job=next(j for j in queue['jobs'] if j['canonical_CPU_condition_reference']==identity)
        proof=read(binding['full_array_proof_path']); report=read(binding['independent_metric_report_path'])
        if groups.condition(original,queue,job)!=proof or metrics.audit(mc,queue,upstream,job)!=report:
            raise ValueError('Full checkpoint source/replay and independent entire-array metrics must agree')
        guard=read(ROOT/mc['state_root']/'guards'/(identity+'.json'))
        if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
                or guard['report_sha256']!=binding['independent_metric_report_sha256']):
            raise ValueError('Actual independent metric guard required')
        cross_reports[identity],proofs[identity]=report,proof
    cross_cohorts=groups.cohorts(queue['jobs'],proofs)
    if json.loads(json.dumps(cross_cohorts))!=read(ROOT/config['cross_frozen_cohorts']):
        raise ValueError('Original primary cohort completion must remain unchanged')
    sc=read(ROOT/spectral.CONFIG); parent,cases=spectral.verify(sc)
    sb=read(ROOT/config['spectral_frozen_catalog'])['records']
    if set(sb)!=set(config['completed_spectral_ids']):
        raise ValueError('Exact frozen full-generation condition roster required')
    spectral_reports={}
    for identity in config['completed_spectral_ids']:
        binding=sb[identity]
        if sha(binding['report_path'])!=binding['report_sha256'] or sha(binding['guard_path'])!=binding['guard_sha256']:
            raise ValueError('Frozen full-generation report or guard changed')
        report,guard=read(binding['report_path']),read(binding['guard_path'])
        if (not report['passed'] or not guard['passed'] or guard['exit_code']!=0
                or guard['resource_guard_aborted'] or guard['report_sha256']!=binding['report_sha256']
                or report['full_original_sampling_calls']!=2 or report['original_evaluation_repeats']!=5
                or report['actual_new_generator_training_updates']!=0):
            raise ValueError('Full original generation budget and successful guard required')
        for field in ['original_checkpoint_unchanged','original_source_and_frozen_dataset_verified',
                      'every_model_EMA_optimizer_state_unchanged','all_generated_payload_bytes_replayed',
                      'complete_RNG_restoration_and_equal_consumption','original_sampler_grid_and_solver_unchanged',
                      'all_four_original_evaluators_reexecuted','all_learned_evaluators_independently_retrained']:
            if not report[field]:raise ValueError('Full original numerical proof missing: '+field)
        for artifact in report['artifacts']:
            if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Full original generation/RNG payload changed')
        _,proof_path=spectral.ready_proof(parent,cases[identity],identity)
        if report['original_artifact_proof_sha256']!=sha(proof_path):raise ValueError('Original source artifact proof changed')
        for item in report['metrics'].values():
            values=item['raw_repeats'];mean=statistics.mean(values)
            sd=statistics.stdev(values);se=sd/math.sqrt(5);half=2.7764451051977987*se
            if len(values)!=5 or not all(math.isfinite(x) for x in values):raise ValueError('All5 real evaluator repeats required')
            for name,value in dict(mean=mean,std=sd,se=se,ci95_low=mean-half,ci95_high=mean+half).items():
                if not math.isclose(item[name],value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Full evaluator statistics differ')
        spectral_reports[identity]=report
    spectral_cohorts=spectral.full_cohort_comparisons(cases,spectral_reports)
    if spectral_cohorts!=read(ROOT/config['spectral_frozen_cohorts']):
        raise ValueError('Full14 cohort status must remain unchanged')
    return queue['jobs'],cross_reports,cases,spectral_reports,cross_cohorts,spectral_cohorts


def interactive(rows, headers):
    payload=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    columns=json.dumps(headers)
    measured=sum(bool(r['mean']) for r in rows)
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验指标 v55</title>'
      '<style>body{font:14px system-ui;margin:24px;color:#18232e}input{padding:9px;width:min(650px,85vw)}'
      'button{padding:8px;margin:8px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #d4dce3;white-space:nowrap}'
      'th{position:sticky;top:0;background:#edf3fa}.scroll{overflow:auto;max-height:72vh}small{display:block;margin:10px 0}</style>'
      f'<h1>算法—数据集—指标 v55</h1><p>{len(rows):,} 行；{measured:,} 行有数值。'
      '空白包含待完成和原表已完成但指标未定义的情况，请查看 comparison。</p>'
      '<p>新增轨道按固定检查点条件列出，不是多训练种子汇总；训练种子不确定性与评估器内部重复分开。'
      '原论文、严格控制、完整TAB及新数据任务均不能由此自动认证完成。</p>'
      '<input id="filter" placeholder="筛选算法、数据集、协议、状态或指标"><small id="count"></small>'
      '<button id="previous">上一页</button><button id="next">下一页</button>'
      '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in headers)
      +'</tr></thead><tbody id="body"></tbody></table></div><script>const rows='+payload+',cols='+columns+';'
      'const search=rows.map(r=>Object.values(r).join(" ").toLowerCase());let page=0,kept=rows;const size=100;'
      'const input=document.getElementById("filter"),body=document.getElementById("body");'
      'function render(){body.replaceChildren();const frag=document.createDocumentFragment();'
      'for(const r of kept.slice(page*size,(page+1)*size)){const tr=document.createElement("tr");'
      'for(const c of cols){const td=document.createElement("td");td.textContent=r[c]||"—";tr.append(td)}frag.append(tr)}'
      'body.append(frag);document.getElementById("count").textContent="筛选 "+kept.length+" / "+rows.length+" 行；第 "+(page+1)+" 页";'
      'document.getElementById("previous").disabled=page===0;document.getElementById("next").disabled=(page+1)*size>=kept.length;}'
      'input.addEventListener("input",()=>{const q=input.value.toLowerCase();kept=rows.filter((_,i)=>search[i].includes(q));page=0;render()});'
      'document.getElementById("previous").onclick=()=>{if(page>0){page--;render()}};'
      'document.getElementById("next").onclick=()=>{if((page+1)*size<kept.length){page++;render()}};render();</script></html>')


def publish(config_path=CONFIG):
    config=read(ROOT/config_path)
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen publication input changed: '+item['path'])
    with (ROOT/config['base_csv']).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    jobs,cross,cases,spectral,cc,sc=independent_evidence(config)
    captured=datetime.now(timezone.utc).isoformat()
    extra=cross_rows(headers,jobs,cross,captured,config)+spectral_rows(headers,cases,spectral,captured,config)
    rows=before+extra
    validate(before,rows,len(cross),len(spectral))
    output=ROOT/config['output_directory'];output.mkdir(parents=True,exist_ok=False)
    with (output/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    (output/'results.html').write_text(interactive(rows,headers),encoding='utf-8')
    (output/'README.md').write_text('''# 算法—数据集—指标 v55

保留v54全部8626行及原字段、捕获时间不变；增加CrossAD全139条件×3个验证阈值×5指标=2085行，以及生成全70条件×4指标=280行，总计10991行。新轨道只展示冻结的11个CrossAD完整条件和6个完整生成重跑条件，共189个已有实测单元格；没有新增模型训练或推理。

每行是一个固定检查点条件，n=1/required_repeats=1仅指该条件已完整，不代表五/十训练种子组完成。CrossAD仍只有1/22完整主要五种子组，生成重跑仍只有1/14完整五种子组。未完成条件均留空。原表已完成但未定义的指标及其状态保留，不能把空白数当作未运行实验数。

CrossAD的0.1%、1%、5%阈值仅使用验证分数，严格>、无PA；测试标签oracle与完整TAB评估不能混入此轨道。论文重建消融保留未获原作者架构认证的边界；发布权重评估不算新训练。

生成重跑的5次评估器重复仅填入evaluator_*，训练种子SD/SE/CI保持空白。两次完整新采样、全部生成/RNG文件及四类各5次原评估守卫重新核验；历史采样RNG未知，不声称历史生成字节或论文成绩已重现。原始源产物轨道与本重跑轨道不能相加计为训练完成数。

CSV可供程序读取，HTML支持搜索并每页显示100行，避免一次渲染所有指标。新记录均不认证作者等价或严格TAB。各旧轨道原有认证/状态按原表保留。

完整45论文/681方法记录/273数据任务/702既有论文指标以及原论文、TAB、新数据任务范围继续保留；本表更新不是整体实验完成证明。
''',encoding='utf-8')
    status=dict(captured_utc=captured,configuration=config_path,configuration_sha256=sha(ROOT/config_path),
        original_scope=config['original_scope'],legacy_rows_preserved=len(before),new_condition_rows=len(extra),
        metric_rows=len(rows),numeric_rows=sum(bool(r['mean']) for r in rows),
        blank_numeric_rows=sum(not r['mean'] for r in rows),
        CrossAD_verified_conditions=len(cross),CrossAD_required_conditions=139,
        spectral_verified_conditions=len(spectral),spectral_required_conditions=70,
        new_numeric_cells=15*len(cross)+4*len(spectral),new_training_updates=0,
        original_CPU_completion_credit=0,old8626_rows_and_capture_times_unchanged=True,
        all_original_experiments_complete=False)
    (output/'publication_validation.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'frozen_full_cohort_status.json').write_text(json.dumps(dict(CrossAD=cc,Spectral=sc),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args();publish(args.config)
