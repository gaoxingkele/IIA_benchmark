import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
const runtimeRequire=createRequire(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json'));
const {Workbook, SpreadsheetFile}=await import(runtimeRequire.resolve('@oai/artifact-tool'));

const root=process.cwd();
const source=path.join(root,'projects/flow_matching_research/results/2026-10-09/complete_metrics');
const outputDir=path.join(root,'outputs/fm_complete_metrics_20261009');
await fs.mkdir(outputDir,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(source,'complete_results.json'),'utf8'));
const workbook=Workbook.create();
const sheets=[];
const col=n=>{let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;};
const cell=v=>v==null?null:typeof v==='object'?JSON.stringify(v):v;
function add(name,title,note,headers,rows,numeric=[],widths={}){
 const sheet=workbook.worksheets.add(name);sheets.push({sheet,headers,rows});
 const end=rows.length+5,last=col(headers.length);
 sheet.showGridLines=false;
 sheet.getRange(`A1:${last}${end}`).format.font={name:'Arial',size:10,color:'#172B4D'};
 sheet.getRange('A2').values=[[title]];sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true};
 sheet.getRange('A3').values=[[note]];
 sheet.getRange(`A5:${last}5`).values=[headers];
 if(rows.length)sheet.getRange(`A6:${last}${end}`).values=rows.map(r=>r.map(cell));
 sheet.getRange(`A5:${last}${end}`).format.columnWidth=18;
 sheet.getRange(`A5:${last}5`).format={fill:'#23395D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:28,horizontalAlignment:'center',verticalAlignment:'center'};
 if(rows.length)sheet.getRange(`A6:${last}${end}`).format.rowHeight=22;
 for(const [c,w] of Object.entries(widths))sheet.getRange(`${c}5:${c}${end}`).format.columnWidth=w;
 for(const n of numeric)sheet.getRange(`${col(n)}6:${col(n)}${end}`).setNumberFormat('0.000000');
 if(rows.length){const table=sheet.tables.add(`A5:${last}${end}`,true,`MetricsTable${sheets.length}`);table.showFilterButton=true;}
 sheet.freezePanes.freezeRows(5);sheet.freezePanes.freezeColumns(2);
 return sheet;
}
const summary=data.strict_summary;
const stats=['precision','recall','f1','auroc','average_precision'];
const mainHeaders=['算法配置','数据集','完成种子数','预定种子数','Precision均值','Recall均值','点级F1均值','AUROC均值','AP均值','VUS ROC均值','VUS PR均值','Affiliation F均值','VUS种子数','Affiliation种子数','状态','运行中','失败/部分产物','待运行'];
for(const s of stats)mainHeaders.push(s+' STD',s+' SE',s+' CI95下界',s+' CI95上界');
mainHeaders.push('协议','模型配置文件','作者等价已证明');
add('严格结果','算法—数据集严格异常检测结果',
 '验证段1%阈值，无PA。缺失值留空。非重叠窗口训练预算未与作者stride=1对齐。',mainHeaders,
 summary.map(r=>[r.algorithm_config,r.dataset,r.completed_seeds,r.required_seeds,...stats.map(s=>r[s+'_mean']),r.VUS_ROC_mean,r.VUS_PR_mean,r.affiliation_f_mean,r.VUS_ROC_n,r.affiliation_f_n,r.status,r.running,r.partial_or_failed,r.pending,
 ...stats.flatMap(s=>['std','se','ci95_low','ci95_high'].map(t=>r[s+'_'+t])),r.protocol,r.model_config,false]),
 [5,6,7,8,9,10,11,12,...Array.from({length:20},(_,i)=>i+19)],{A:55,B:14,C:16,D:16,E:20,F:20,G:20,H:20,I:20,J:20,K:20,L:24,O:25,AN:65});
const runKeys=['algorithm_config','dataset','seed','calibration_false_alarm_percent','precision','recall','f1','auroc','average_precision','macro_f1','VUS_ROC','VUS_PR','affiliation_f','affiliation_precision','affiliation_recall','event_recall','events','detected_events','mean_detection_delay_points','false_alarms_per_1000_normal_points','true_positives','false_positives','false_negatives','f1_ci95_low','f1_ci95_high','ci_unit','threshold','parameter_count','epochs_completed','training_seconds','score_seconds','peak_cuda_allocated_bytes','VUS_ROC_defined_entities','VUS_ROC_undefined_entities','affiliation_f_defined_entities','affiliation_f_undefined_entities','run_id','protocol','result_path','result_sha256','scores_sha256','checkpoint_sha256'];
const runHeaders=['算法配置','数据集','种子','校准分位数(%)','Precision','Recall','点级F1','AUROC','AP','实体macro F1','VUS ROC macro','VUS PR macro','Affiliation F macro','Affiliation P macro','Affiliation R macro','事件召回','事件数','发现事件数','平均延迟(点)','误报/千正常点','TP','FP','FN','F1 CI95下界','F1 CI95上界','区间单位','阈值','参数量','训练轮数(有记录)','训练秒','评分秒','CUDA峰值(字节)','VUS有定义实体数','VUS未定义实体数','Aff.F有定义实体数','Aff.F未定义实体数','运行ID','协议','结果来源','结果SHA256','分数SHA256','模型SHA256'];
add('逐种子指标','每个完整运行的全部严格技术指标','三档验证阈值分别列行。延迟来自离线窗口分数，并非因果响应延迟。',runHeaders,
 data.strict_runs.map(r=>runKeys.map(k=>r[k])),[5,6,7,8,9,10,11,12,13,14,15,16,19,20,24,25,27,30,31],{A:55,D:20,J:23,K:23,L:23,M:25,N:25,O:25,S:22,T:24,Z:32,AC:26,AK:80,AM:115});
const tabKeys=['algorithm_config','dataset','seed','ratio_percent','f_score','adjust_f_score','auc_roc','auc_pr','threshold','test_scores_used_in_calibration','test_labels_used_to_select_best_grid','run_id','result_path','result_sha256'];
add('TAB比例诊断','TAB比例诊断与PA对照','合并训练/测试分数校准。挑选最高比例分数依赖测试标签。与严格主表分开解释。',
 ['算法配置','数据集','种子','比例(%)','点级F1','PA-F1','AUROC','AP','阈值','测试分数参与校准','测试标签选最优比例','运行ID','结果来源','结果SHA256'],
 data.tab_ratio_diagnostic.map(r=>tabKeys.map(k=>r[k])),[5,6,7,8,9],{A:55,J:30,K:33,L:85,M:120,N:70});
const historyKeys=['model','dataset','seed','protocol','pointwise_precision','pointwise_recall','pointwise_f1','adjusted_precision','adjusted_recall','adjusted_f1','range_precision','range_recall','range_f1','reported_protocol_f1','threshold','scored_points','unscored_points','anomaly_rate_scored','layout','threshold_rule','point_adjustment','parameters','record_id','source_sha256','status'];
add('历史协议结果','全部历史异常检测记录','同种子可存在不同参数或布局，不能合并为新严格成绩。window_flatten可能重复时间点。',
 ['算法','数据集','种子','协议','点级Precision','点级Recall','点级F1','PA-Precision','PA-Recall','PA-F1','范围Precision','范围Recall','范围F1','原协议报告F1','阈值','评分点数','未评分点数','评分异常占比','分数布局','阈值规则','PA开关','原参数','原始记录来源','来源SHA256','历史状态'],
 data.historical_runs.map(r=>historyKeys.map(k=>r[k])),[5,6,7,8,9,10,11,12,13,14,15,18],{A:27,D:30,H:22,I:22,J:22,K:24,L:24,M:22,N:26,S:24,T:28,V:125,W:125,X:70,Y:75});
const imputation=data.imputation;
const formal=[...imputation.formal_results].sort((a,b)=>Number(b.local_mean!==null)-Number(a.local_mean!==null)||a.algorithm.localeCompare(b.algorithm)||a.dataset.localeCompare(b.dataset));
add('插补结果','原协议和迁移插补结果','误差越低越好。掩码、缺失率、单位和折/种子匹配后才能与论文比较。',
 ['算法','数据集','缺失率','掩码','协议','指标','本地均值','STD','SE','CI95下界','CI95上界','完成数','预定数','论文值','论文SE','本地减论文','相对偏差','比较状态','尺度','完成折/种子','结果来源','原因'],
 formal.map(r=>[r.algorithm,r.dataset_label,r.missing_ratio,r.pattern,r.track,r.metric,r.local_mean,r.local_std,r.local_se,r.ci95_low,r.ci95_high,r.completed_repeats,r.required_repeats,r.paper_mean,r.paper_se,r.signed_delta,r.relative_delta,r.verdict,r.metric_scale,r.completed_fold_or_seed,r.result_paths,r.reason]),
 [3,7,8,9,10,11,14,15,16,17],{A:21,B:21,D:22,E:34,F:24,R:65,S:58,T:30,U:130,V:130});
const perKeys=['algorithm','dataset','missing_ratio','track','pattern','fold','seed','metric','value','training_seconds','evaluation_seconds','eval_count','test_cases','num_samples','epochs','run_id','config','config_sha256','result','result_sha256','processed_sha256','checkpoint_sha256'];
add('插补每次运行','插补每个折或种子的原始指标','耗时与checkpoint可能跨多个指标行共享。不能按指标行累加训练耗时。',
 ['算法','数据集','缺失率','协议','掩码','折','种子','指标','数值','训练秒','评估秒','评估点数','测试样本','生成样本','轮数','运行ID','配置','配置SHA256','来源','结果SHA256','数据SHA256','模型SHA256'],
 imputation.per_run_metrics.map(r=>perKeys.map(k=>r[k])),[3,9,10,11],{A:23,B:23,D:35,H:26,P:80,Q:110,S:130});
const allJobs=[...data.tsad_jobs.map(r=>[path.basename(r.model_config,'.json'),r.dataset.toUpperCase(),'异常检测',r.lane,null,null,r.seed,r.status,r.id,r.model_config,null]),
 ...imputation.jobs.map(r=>[r.algorithm,r.dataset,'插补',r.track,r.missing_ratio,r.fold,r.seed,r.status,r.id,r.config,r.pattern])];
add('全部登记任务','全部登记实验任务','异常检测1750项含105次修复重跑。另有540项插补任务。无结果项不计零分。',
 ['算法配置','数据集','任务','执行轨道','缺失率','折','种子','状态','任务ID','配置来源','掩码'],allJobs,[5],{A:55,D:35,H:25,I:100,J:110,K:24});
add('论文报告值','ARA已核验转录的论文指标','论文原单位保留。F1_percent为0–100，F1为0–1。未转录表格仍待核验。',
 ['算法/表行','数据集','缺失率','指标','论文数值','SE','STD','重复数','PDF页','表号','原协议','转录核验','论文ID','参考ID','论文URL','全文SHA256'],
 imputation.author_claims.map(r=>[r.algorithm,r.dataset,r.missing_ratio,r.metric,r.paper_value,r.paper_se,r.paper_std,r.paper_runs,r.page,r.table,r.protocol,r.verification,r.paper_id,r.reference_id,r.citation,r.source_sha256]),
 [3,5,6,7],{A:27,B:27,D:30,K:120,L:42,M:28,N:35,O:100,P:70});
add('原文数据范围','45篇论文的原始实验数据范围','273条是含子集的原文数据集描述，不等于273个独立数据集。缺口逐项保留。',
 ['论文/算法','任务','原文数据集描述','指标行数','有数值行数','本地绑定状态','全文SHA256','来源URL','复现缺口'],
 imputation.original_dataset_scope.map(r=>[r.paper_id,r.task,r.dataset_description,r.registered_local_metric_rows,r.numeric_local_metric_rows,r.status,r.source_sha256,r.citation,r.reproduction_gap]),[],
 {A:32,B:40,C:90,D:18,E:18,F:80,G:72,H:105,I:160});
workbook.recalculate();
console.log((await workbook.inspect({kind:'table',range:'严格结果!A5:N10',include:'values,formulas',tableMaxRows:6,tableMaxCols:14,maxChars:1800})).ndjson);
console.log((await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1200})).ndjson);
for(const {sheet} of process.argv.includes('--skip-render') ? [] : sheets){
 const preview=await workbook.render({sheetName:sheet.name,range:'A1:J12',scale:1,format:'png'});
 await fs.writeFile(path.join(outputDir,sheet.name+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir,'complete_experiment_metrics.xlsx'));
await fs.copyFile(path.join(outputDir,'complete_experiment_metrics.xlsx'),path.join(source,'complete_experiment_metrics.xlsx'));
await fs.writeFile(path.join(outputDir,'workbook_manifest.json'),JSON.stringify({sheets:sheets.map(s=>({name:s.sheet.name,rows:s.rows.length,columns:s.headers.length})),source:path.join(source,'complete_results.json')},null,2));
console.log(JSON.stringify({output:path.join(outputDir,'complete_experiment_metrics.xlsx'),sheets:sheets.length,rows:sheets.reduce((s,t)=>s+t.rows.length,0)}));
