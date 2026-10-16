from pathlib import Path
import json,hashlib,shutil,subprocess,csv,sys
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
import fitz,pypdf,psutil
from scripts.flow_matching import audit_timesnet_moment_original_metrics_v1 as a
root=a.ROOT
packet=root/'projects/flow_matching_research/results/2026-10-16/TimesNet_MOMENT_original_protocol_v1'
assert not packet.exists();packet.mkdir(parents=True)
def write(p,v):
 assert not p.exists(),p
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
r=a.run();write(packet/'original_tables_and_protocol_audit.json',r)
test=root/'tmp/TimesNet_MOMENT_original_metrics_audit_20261016_v1_tests.xml'
suites=list(ET.parse(test).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==26;assert not any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped'])
shutil.copyfile(test,packet/'validation_26_cases.xml')
write(packet/'environment.json',dict(python=sys.version,PyMuPDF=fitz.VersionBind,pypdf=pypdf.__version__,numerical_packages_modified=False))
refs=[]
for key,rows in [('timesnet',r['timesnet']['table15_rows']),('moment',r['moment']['table7_summary_rows']+r['moment']['table22_visible_cells'])]:
 for row in rows:refs.append(dict(paper=key,model=row['model'],dataset=row.get('dataset',row.get('dataset_printed_label')),metric=row['metric'],statistic=row.get('statistic','reported_scalar'),printed_value=row['printed_value'],unit=row['unit'],pdf_page=row['pdf_page'],table=row['table'],evidence_kind='author_reported_only',local_benchmark_result=False))
with (packet/'paper_reference_cells.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(refs[0]));w.writeheader();w.writerows(refs)
assert len(refs)==818
src=root/'tmp/pdfs/fm_timesnet_moment_original_metric_audit_20261016_v1';renders=[]
for stem in ['timesnet_primary_p08','timesnet_primary_p09','timesnet_primary_p12','timesnet_primary_p14','timesnet_primary_p22','moment_p06','moment_p07','moment_p09','moment_p20','moment_p27','moment_p28','moment_p29','moment_p33']:
 target=packet/'reviewed_pages'/f'{stem}.png';target.parent.mkdir(exist_ok=True);shutil.copyfile(src/f'{stem}.png',target)
 renders.append(dict(path=target.relative_to(packet).as_posix(),sha256=a.sha(target),review='complete_page_visually_inspected_in_session',renderer='PyMuPDF 1.28.0',scale=1.6))
write(packet/'page_review_manifest.json',dict(rendered_pages=renders,raw_PDFs_not_modified=True))
shutil.copyfile(root/'tmp/actual_numeric_wait_20261016_1735_v1.json',packet/'verified_numeric_wait_45_seconds.json')
ara=root/'projects/flow_matching_research/ara'
links=[]
for name in ['timesnet','moment']:
 p=ara/'papers'/name/'evidence/tables/reported_results.2026-10-16_v1.json'
 write(p,dict(paper_id=name,status='original_reported_tables_audited_protocol_identity_incomplete',audit_packet=packet.relative_to(root).as_posix(),results=r[name],local_benchmark_result=False,all_original_experiments_complete=False))
 links.append(p)
 readme=p.parent/'README.md'
 readme.write_text(f'''# {name} 原文结果与协议\n\n2026-10-16 逐项核验的补充参考见 [reported_results.2026-10-16_v1.json](reported_results.2026-10-16_v1.json)。早期 reported_results.json 的空结果保留为历史记录；当前异常检测论文对照应先读取本补充记录及其协议限制。\n\n该记录只含论文报告值和发布代码协议，不能作为本地排行榜结果或复现一致性证明。TimesNet 两个卷积变体、论文10 epoch与发布脚本预算分别记录；MOMENT 原文UCR异常任务、best F1、降采样与未解决的样本数量差异分别记录。完整表页、原PDF校验和、两种独立解析器验证以及26项检查见审计包。\n''',encoding='utf-8',newline='\n')
 links.append(readme)
 p=ara/'papers'/name/'src/configs/original_anomaly_protocol.2026-10-16_v1.json'
 write(p,dict(paper_id=name,supplements_legacy_metadata=True,original_protocol=r[name],audit=links[-2].relative_to(root).as_posix(),training_configuration_changed=False))
 links.append(p)
p=ara/'execution/TimesNet_MOMENT_original_protocol.2026-10-16_v1.json';write(p,dict(result_packet=packet.relative_to(root).as_posix(),reference_cells=818,original_NaN_cells_retained=10,tests_passed=26,new_complete_benchmark_conditions=0,all_original_experiments_complete=False));links.append(p)
(packet/'README.md').write_text('''# TimesNet、MOMENT 原始异常检测结果口径核验\n\n本轮核验818个原表单元格：TimesNet Table15全部17模型×16指标=272，MOMENT Table7的均值/中位数/跨序列标准差42项，以及Table22可见42行×12单元格=504（含10个原始NaN，保留为null而非0）。完整相关页已渲染逐页查看；PyMuPDF逐项转录后用独立pypdf提取器核验全部818格，26项检查通过。原论文报告值没有计入本地实验成绩。\n\nTimesNet Table5/15中PSM 97.47%来自ResNeXt，当前发布代码及本地转录使用Inception，其PSM为95.21%。SWaT分别为ResNeXt91.74%、Inception92.10%；此前提到的92.4%未在这两张原表中确认。Table5最高SWaT分数实际上是LightTS93.33%。两个已下载PDF的Table15转录完全一致，Table5标准化文字一致。保留原表Stationary在SMD的84.72%（Table5）与84.62%（Table15）差异。\n\n论文Table7的异常检测统一预算10 epoch、3层、top-k3、batch128、Adam1e-4，Appendix A说明重复3次；发布脚本SMAP/PSM/SWaT为3 epoch、MSL1 epoch、SMD10 epoch，架构设置也有差异。当前常量拟合SMAP51轨道保留发布预算，不因本次审计而改动。该轨道不是原文Table6样本数或完整NASA54/109实体的证明。ResNeXt原始实验架构与完整消融尚未认证。\n\n发布TimesNet评估器用训练分数与测试分数合并的百分位数定阈值，然后依据测试标签做point adjustment；该阈值并非以测试标签最大化F1。这里只证明所固定版本发布代码的语义，不能声称已经查明历史原文实际运行过程。严格验证集定阈值、不做PA的F1不能直接判作原表是否复现。\n\nMOMENT本原文Table7/22的异常检测任务为UCR，使用adjusted best F1及VUS-ROC。Table12的SMD/NASA属于预训练语料说明，不是SMD下游F1=0.832的证据。原Table7的MOMENT0/LP均值分别为adj.best F1 0.585/0.628、VUS-ROC0.670/0.684；其std是跨序列分布，不能改成训练种子SE/CI。\n\n原文对数据数量存在未解决差异：正文44、Table22表注45、F.3正文46、Table7表注248，本Table22实际可见42行。没有强行合并这些集合或据此重新计算原文均值。F.3规定窗口512、MSE、长于2560的序列降采样10倍；固定发布指标实现使用100个测试分数线性阈值、>=预测、标签驱动PA并用测试标签选择最大F1。这不是严格TAB分数，也不等于本地全阈值精确oracle。脚注17还说明Anomaly Transformer未用原联合异常判据，原文承认可能使其处于不利地位。\n\n对照门禁逐项检查架构、精确数据、分割预处理、训练预算、阈值来源、PA、聚合集合、指标实现和不确定性定义；任一不同或未知都拒绝“数值复现一致/不一致”的自动结论。完全匹配元数据也不会自动认证作者等价。旧ARA笼统数据集列表保留为历史材料，本补充明确其不能认证MOMENT原文异常任务。\n\n本轮没有改动模型、运行配置、原数据或数值进程，没有新增完整训练/benchmark结果。归档的45秒进程核验显示四个精确PID/创建时间/命令一致且CPU推进。完整45论文/681方法记录/273数据任务/702既有指标范围及原论文/TAB/新数据目标继续保留；本包只补充文献协议证据，整体目标未完成。\n''',encoding='utf-8',newline='\n')
shutil.copyfile(__file__,packet/'closure_source.py') if '__file__' in globals() and Path(__file__).is_file() else None
write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=a.sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
paths=[a.CONFIG,'scripts/flow_matching/audit_timesnet_moment_original_metrics_v1.py','tests/test_fm_timesnet_moment_original_metric_audit_v1.py']+[p.relative_to(root).as_posix() for p in links]+[p.relative_to(root).as_posix() for p in packet.rglob('*') if p.is_file()]
def head(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(root/'tmp/TimesNet_MOMENT_original_protocol_manifest_20261016_v1.json',dict(previous_root=head(root),previous_parent=head(root.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),files=len(paths),reference_cells=len(refs),bytes=sum(p.stat().st_size for p in packet.rglob('*') if p.is_file()))))
