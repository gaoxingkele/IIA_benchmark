"""Publish a bounded archival protocol analysis,not new benchmark performance."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import summarize_verified_protocol_effects_v1 as runner

ROOT,read,sha,write=runner.ROOT,runner.read,runner.sha,runner.write
config=read(ROOT/runner.CONFIG);runner.verify(config)
folder=ROOT/config['output_directory'];report=read(folder/'report.json');receipt=read(folder/'run_receipt.json')
assert receipt['passed'] and receipt['configuration_sha256']==sha(ROOT/runner.CONFIG)
assert receipt['new_training_updates']==receipt['new_model_inference']==0
for name,digest in receipt['generated_files_sha256'].items():assert sha(folder/name)==digest
render=read(folder/'publication_render_v2_receipt.json')
assert render['passed'] and render['source_report_sha256']==sha(folder/'report.json')
assert not render['numerical_values_or_cases_changed']
for name,digest in render['output_sha256'].items():assert sha(folder/name)==digest
xml=ROOT/'tmp/Verified_AD_protocol_effects_validation_20261015_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==7
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
for name,path in {
    'frozen_analysis_configuration.json':ROOT/runner.CONFIG,
    'evidence_analysis_source.py':ROOT/'scripts/flow_matching/summarize_verified_protocol_effects_v1.py',
    'publication_renderer_v2_source.py':ROOT/'scripts/flow_matching/render_verified_protocol_effects_v2.py',
    'validation_tests.py':ROOT/'tests/test_fm_verified_protocol_effects_v1.py',
    'validation.xml':xml}.items():shutil.copyfile(path,folder/name)
context=[]
for row in report['cases']:
    effect=row['same_threshold_effect']
    if 'no_PA_confusion' not in effect:continue
    c=effect['no_PA_confusion'];positive=c['tp']+c['fn'];predicted=c['tp']+c['fp']
    maximum=2*min(positive,predicted)/(positive+predicted) if positive+predicted else 0.
    assert effect['no_PA_f1']<=maximum+2e-12
    context.append(dict(case_id=row['case_id'],test_points=row['test_points'],ground_truth_positive_points=positive,
        positive_prevalence=positive/row['test_points'],unadjusted_alarm_points=predicted,
        unadjusted_alarm_fraction=predicted/row['test_points'],maximum_possible_unadjusted_F1_given_this_alarm_count=maximum,
        derivation='TP<=min(number_of_alarms,number_of_positive_truth_points); F1=2TP/(alarms+positives)',
        measured_F1=effect['no_PA_f1'],measured_AUROC=row['AUROC'],measured_AP=row['average_precision'],
        interpretation_zh='固定低告警数量本身限制点级F1上限；低F1不能单独证明排序无效。AUROC/AP和PA效应应分开判断。'))
write(folder/'fixed_alarm_budget_context.json',dict(rows=context,derived_only_from_already_verified_confusion_counts=True,
    no_new_score_or_threshold_selection=True,no_statistical_significance_or_cross_model_ranking_claim=True))
source,causal,moment,sb=report['cases']
table=['|个案（仅作同个案协议对照）|未调整 F1|同阈值 PA F1|增幅（百分点）|分数 AUROC|',
       '|---|---:|---:|---:|---:|']
for row in report['cases']:
    e=row['same_threshold_effect'];table.append(f"|{row['label']}|{100*e['no_PA_f1']:.2f}%|{100*e['PA_f1']:.2f}%|{e['F1_increase_percentage_points']:.2f}|{row['AUROC']:.4f}|")
(folder/'README.md').write_text(
    '# 已核验异常检测个案的评估协议效应\n\n'
    '本报告分析四个已封存个案，来自三个方法家族、三个数据集。所有行都是“同一模型、同一分数、同一阈值”的PA对照，'
    '不能用来跨算法排名。它不验证论文原始SOTA目标，也不代表其余论文全部属于同一情况。\n\n'
    +'\n'.join(table)+'\n\n'
    'Pi两个分支均为PSM/seed1107，测试87800点，原流程另丢弃尾部41点。它们已有完整检查点和15流独立复核，'
    '但原始训练保护历史未证实，且来源保留了测试集选模、池化测试分数校准和PA；没有认证期刊历史版本或完整TAB。'
    '两行不是两项独立研究。\n\n'
    'MOMENT行为base/512上下文、SMAP常量特征预处理对照。已有完整保存分数审计，尚无本个案的新检查点重推理证书。'
    '表中使用同一冻结验证阈值；本报告没有把证据强度升级。\n\n'
    'SB行为本地流匹配目标的时序异常适配器，表中同阈值对照使用池化训练/测试分数的1%阈值，属于诊断。'
    '该模型的正式验证阈值、未调整F1为28.93%，分数AUROC为0.8212、AP为0.7236，测试正常点误报率为54.93%。'
    '十点测试标签选优网格可给出未调整F1约73.85%、PA后约85.26%；这是阈值与协议改变，不是训练改进。'
    '高排序指标伴随高正常误报，提示校准向测试分布迁移时可能失配；不能从这些记录唯一归因于训练、容量或预处理。\n\n'
    '未调整F1也受告警预算与异常占比共同限制。附加JSON按已核验混淆计数给出固定告警数量下的F1上限；'
    '不能仅据低F1断言网络完全无效，应同时检查AUROC/AP、误报和事件指标。PA提高召回需要测试事件标签，'
    '不能把这种事后赋分直接解释为部署时逐点报警能力。\n\n'
    '首选图为 [protocol_effects_v2.png](protocol_effects_v2.png) 或可缩放的 [protocol_effects_v2.svg](protocol_effects_v2.svg)。'
    '已逐图检查，图例移至数据区外，曲线峰值不被遮挡。v1初始图和冻结分析文件保留以供追溯；'
    'v2只改排版，数值和证据不变。\n\n'
    '7项检查通过，覆盖同支持集/同正常预测的PA语义、错误计数拒绝、实际封存证据、证据强度保留、'
    '测试网格选优与PA效应分离、源文件篡改拒绝。24个证据文件的SHA256及Git已提交字节均已核对。'
    '本报告没有重新读入原始分数或权重，没有新增训练、推理、随机种子、跨行置信区间或p值。\n\n'
    '公平主对比仍要求统一分组/时间点支持、冻结验证校准、未调整F1与AUROC/AP和误报/事件指标、完整种子组和不确定性，'
    '并明确数据预处理与证据强度。完整TAB训练/评估等价性和论文等价性需要另行证明。'
    '原45篇/681方法记录/273数据任务的最低范围和其他未完成实验继续保留，全部任务尚未完成。\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.svg -text -diff\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=7,
    verified_closed_case_records=4,method_families=3,datasets=3,preferred_figures=['protocol_effects_v2.png','protocol_effects_v2.svg'],
    visual_QA_passed=True,original_evidence_files=24,source_git_head=report['source_git_head'],
    files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    new_benchmark_runs=0,new_training_or_inference=0,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),runner.CONFIG,
    'scripts/flow_matching/summarize_verified_protocol_effects_v1.py',
    'scripts/flow_matching/render_verified_protocol_effects_v2.py',
    'tests/test_fm_verified_protocol_effects_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/Verified_AD_protocol_effects_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=7,closed_cases_analyzed=4,new_benchmark_runs=0)))
