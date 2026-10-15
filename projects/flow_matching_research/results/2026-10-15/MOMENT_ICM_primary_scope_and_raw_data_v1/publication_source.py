"""Seal the newly expanded ICM scope and actual read-only240file audit."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import audit_moment_icm_primary_scope_v1 as runner
from scripts.flow_matching import moment_icm_main_table_targets_v1 as main

ROOT,read,sha=runner.ROOT,runner.read,runner.sha
configpath='configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json'
config=read(ROOT/configpath);detailed=runner.verify(config);aggregate=main.targets(ROOT/config['paper'])
folder=ROOT/config['output_directory'];report=read(folder/'complete_raw_data_audit.json')
assert report['file_status_counts']=={'verified':240} and report['paper_sha256']==sha(ROOT/config['paper'])
assert not report['original_pretraining_and_forecasting_and_SVM_experiments_run'] and report['new_fit_or_inference']==0
assert report['classification_files']==234 and report['reported_scalar_targets']==630
assert sum(x['bytes'] for x in report['files'])==2205868527
for row in report['files']:
    if row.get('table')==6:assert all(int(k)>1 for k in row['observed_dimensions'])
    if row.get('table')==7:assert set(row['observed_dimensions'])=={'1'}
checks=0
for name,path,count in [('appendix_validation.xml','tmp/MOMENT_ICM_primary_scope_validation_20261015_v1.xml',14),
                        ('main_table_validation.xml','tmp/MOMENT_ICM_main_tables_validation_20261015_v1.xml',3)]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    checks+=count;shutil.copyfile(ROOT/path,folder/name)
mainconfig='configs/reproducibility/fm_moment_ICM_main_table_targets.2026-10-15_v1.json'
supplement=dict(schema_version=1,primary_scope_configuration=configpath,primary_scope_sha256=sha(ROOT/configpath),
    paper_sha256=sha(ROOT/config['paper']),reported_main_table_targets=aggregate,
    source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in [
        'scripts/flow_matching/moment_icm_main_table_targets_v1.py','tests/test_fm_moment_icm_main_tables_v1.py']],
    boundary='72aggregate target cells,not72new experiments;combined702reported source-table values include630appendix cells. '
        'Do not map Table4fold labels onto Table2beta switches without evidence. Table3std is across26datasets,not training-seed uncertainty.')
with (ROOT/mainconfig).open('x',encoding='utf-8',newline='\n') as stream:json.dump(supplement,stream,indent=2);stream.write('\n')
combined=aggregate+detailed
for name,rows in [('main72_paper_reported_metrics.csv',aggregate),('all702_paper_reported_metrics.csv',combined)]:
    with (folder/name).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,['table','pdf_page','model','dataset','horizon','metric','reported'])
        writer.writeheader();writer.writerows(rows)
for name,path in {
    'frozen_appendix_scope_configuration.json':ROOT/configpath,
    'frozen_main_table_configuration.json':ROOT/mainconfig,
    'raw_audit_and_appendix_index_source.py':ROOT/'scripts/flow_matching/audit_moment_icm_primary_scope_v1.py',
    'main_table_index_source.py':ROOT/'scripts/flow_matching/moment_icm_main_table_targets_v1.py',
    'appendix_validation_tests.py':ROOT/'tests/test_fm_moment_icm_primary_scope_v1.py',
    'main_table_validation_tests.py':ROOT/'tests/test_fm_moment_icm_main_tables_v1.py'}.items():shutil.copyfile(path,folder/name)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
write(folder/'primary_visual_review_receipt.json',dict(paper=config['paper'],paper_sha256=sha(ROOT/config['paper']),
    visually_reviewed_pdf_pages=[3,4,9,10,11,12],render_scale=2.5,
    rendered_page_sha256={str(i):sha(ROOT/f'tmp/moment_long_context_primary_visual_20261015_v1/page{i}.png') for i in [3,4,9,10,11,12]},
    full_table_labels_and_MAE_MSE_order_and_continuations_reviewed=True,
    numeric_reference_anchor_and_structure_tests=17,independent_double_manual_transcription_claimed=False))
write(folder/'experiment_execution_gaps.json',dict(paper_id='moment_long_context',
    all702_original_reported_targets_indexed=True,all117_classification_and6_forecast_raw_datasets_locally_verified=True,
    original_pretraining_completed=False,original_forecast_and_SVM_experiments_completed=False,
    all_context_expansion_baselines_integrated=False,model_components_only=True,
    known_recipe=config['verified_primary_contract'],remaining_fields=config['unresolved_execution_fields'],
    original_archived_AD72_and6controls_are_separate=True,
    numeric_targets_are_not_actual_local_results=True,all_original_experiments_complete=False))
(folder/'README.md').write_text(
    '# MOMENT长上下文/ICM论文完整表格范围与真实数据核验\n\n'
    '针对此前只有组件代码、尚无原文实验执行入口的《Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters》，'
    '本轮按原PDF及附录重新核对了完整结果范围。原文SHA256为3e67405b691183dab729d6f2e56421280d8bb1268761f190468d8bce9e326413，'
    '已视觉检查第3、4、9、10、11、12页，保留渲染收据。\n\n'
    '对照CSV包含全部7张表的702个原文指标值：主文表1至表3共72个汇总值，附录表4至表7共630个详细值。'
    '附录对应198个预测设置及234个分类结果单元；这是432个带表格/模型标签的评估单元，不是432次独立神经网络训练。'
    '表4保留3种原始预训练模型标签，表5保留8种监督模型标签；表6有26个多变量分类数据集，表7另有91个单变量数据集。'
    '旧注释只提26个分类数据集，不能因此漏掉表7。全部原文数字都只是待比对目标，不是本地重现实测结果。\n\n'
    '实际重读并校验了240个文件：117个分类数据集的TRAIN/TEST共234份TS文件，加Exchange、ETTh1、ETTh2、ETTm1、ETTm2、Weather六个CSV，'
    '合计2,205,868,527字节。全部240份当前SHA256、Git blob SHA1及大小均与已登记的MOMENT Time Series Pile校验记录一致；'
    '额外统计了完整实例行数、TS头、每条实例的通道数和分类标签，91个单变量数据集的所有实例均为1通道，26个多变量数据集均为多通道。'
    '本地classification/UCR目录实际也包含这些UEA多变量数据，不能只看目录名判缺失。本轮没有下载重复副本。\n\n'
    '这里没有解析每个浮点值，也没有插补、重采样、归一化、改分组或写入原始数据；并未证明训练/测试无泄漏或历史论文使用了同一字节版本。'
    '240文件核验属于当前已登记数据快照的完整性和结构观察，不能替代预处理/实验复现。\n\n'
    '原文已明确的预算包括：256点输入、96/192/384预测跨度；Tiny为4层、256维、4头、FFN1024；'
    'ICM额外16个beta；预训练一整个epoch；N-BEATS为10个epoch、batch64。正文说多变量最多抽8通道，'
    '附录D则描述分成8通道子组并对余数组过采样；两段都保留。表4的“8 folds/4 folds”标签也没有被擅自解释为表2的beta冻结/微调开关。'
    '表3的标准差是跨26数据集的准确率分散程度，不能当作训练种子不确定性。\n\n'
    '仍需完成的真实工作包括：原Tiny/ICM的一epoch预训练及权重、全部上下文扩展基线的整网对齐、监督预测训练、'
    '冻结表示加SVM的117数据集评估、beta微调消融，以及掩码/优化器/学习率/非NBEATS预算/种子/表示池化/SVM选参的来源对齐或明确披露的重建选择。'
    '原small/base/large的72项异常检测与6项预处理控制不能替代这些实验；当前独立T5/ICM组件可调用也不等于整篇论文已实现或已复现。\n\n'
    '17项检查通过，覆盖全702个坐标的唯一性与数量、MAE/MSE及跨页对应、主文beta分支保留、实际源文件改动拒绝、'
    '整份TS/CSV流式校验和240文件/630附录值的端到端fixture。fixture仅验证审计与导出，不能称为benchmark成绩。'
    '本轮新增训练/模型推理/SVM为0。已有NODE、CrossAD、BRITS训练继续；原45篇/681方法记录/273数据任务是最低范围，'
    '新增发现的原文实验继续保留，全部任务尚未完成。\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=checks,
    primary_reported_scalar_targets=702,actual_rehashed_files=240,actual_rehashed_bytes=2205868527,
    original_models_newly_trained_or_reinferred=0,files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/moment_ICM_primary_full_scope_data.2026-10-15_v1.json'
write(ara,dict(paper_id='moment_long_context',primary_scope=configpath,main_aggregate_targets=mainconfig,
    evidence=folder.relative_to(ROOT).as_posix(),reported_targets=702,
    raw_classification_datasets=117,raw_forecast_datasets=6,raw_verified_files=240,
    original_pretraining_and_forecasting_and_SVM_run=False,AD_replay_is_separate=True,
    gaps_file=(folder/'experiment_execution_gaps.json').relative_to(ROOT).as_posix(),all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),configpath,mainconfig,ara.relative_to(ROOT).as_posix(),
    'scripts/flow_matching/audit_moment_icm_primary_scope_v1.py','scripts/flow_matching/moment_icm_main_table_targets_v1.py',
    'tests/test_fm_moment_icm_primary_scope_v1.py','tests/test_fm_moment_icm_main_tables_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_primary_scope_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),tests_passed=checks,raw_verified_files=240,reported_targets=702,new_model_runs=0)))
