"""Publish current read-only entity coverage and explicit execution obligations."""
from collections import defaultdict
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from scripts.flow_matching import audit_original_entity_coverage_inventory_v1 as audit

ROOT=audit.ROOT
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:audit.file_fact(p)['sha256']
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
c=read(ROOT/audit.CONFIG);folder=ROOT/c['report_root'];proof=read(folder/'actual_original_entity_coverage.json')
assert proof['passed'] and proof['configuration_sha256']==sha(ROOT/audit.CONFIG)
assert not proof['labels_merged_or_selected'] and not proof['raw_files_modified']
assert not proof['numerical_coordinate_audit_complete'] and not proof['preprocessing_or_training_or_model_inference_executed']
assert len(proof['raw_files'])==246 and len(proof['registered_CSV_files'])==191 and len(proof['mirror_raw_members'])==164
assert [r['raw_distinct_labelled_entities'] for r in proof['dataset_coverage']]==[54,27,28]
assert [r['registered_CSV_test_entities'] for r in proof['dataset_coverage']]==[51,26,18]
assert len(proof['duplicate_channel_labels'])==1 and proof['duplicate_channel_labels'][0]['channel']=='P-2'
first,second=defaultdict(list),defaultdict(list)
for r in proof['original_label_rows']:
    row=r['metadata'];first[(row['spacecraft'],row['chan_id'])].append(row)
with zipfile.ZipFile(proof['mirror_archive']['path']) as archive:
    for row in csv.DictReader(io.StringIO(archive.read('labeled_anomalies.csv').decode('utf-8-sig'))):
        second[(row['spacecraft'],row['chan_id'])].append(row)
key=lambda x:json.dumps(x,sort_keys=True)
assert set(first)==set(second)
assert all(sorted(first[k],key=key)==sorted(second[k],key=key) for k in first)
assert all(m['matching_TranAD_raw_byte_hash'] is True for m in proof['mirror_raw_members'].values() if m['labelled_channel'])
extras=[k for k,v in proof['mirror_raw_members'].items() if not v['labelled_channel']]
assert sorted(extras)==['test/T-10','train/T-10']
interpretation=dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    metadata_row_order_differs=not proof['mirror_metadata_rows_equal_to_TranAD'],
    metadata_row_multisets_exactly_equal=True,label_groups_with_semantic_differences=0,
    labelled_mirror_train_test_arrays_bitwise_identical_to_TranAD=162,
    mirror_unlabelled_train_test_members=extras,unlabelled_members_used_for_performance=False,
    raw_file_checksums=246,registered_CSV_file_checksums=191,mirror_raw_member_checksums=164,
    raw_file_bytes=sum(r['bytes'] for r in proof['raw_files'].values()),
    registered_CSV_file_bytes=sum(r['bytes'] for r in proof['registered_CSV_files']),
    duplicates_silently_deduplicated=False,labels_selected_by_model_performance=False,
    numerical_coordinate_equivalence_proven=False,original_paper_dataset_equivalence_certified=False,
    all_original_experiments_complete=False)
write(folder/'metadata_interpretation_and_mirror_identity.json',interpretation)
sources=[audit.CONFIG,'scripts/flow_matching/audit_original_entity_coverage_inventory_v1.py']
for path in sources:
    source=ROOT/path;shutil.copyfile(source,folder/('frozen_'+source.name))
shutil.copyfile(__file__,folder/'publication_source.py')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
with (folder/'dataset_entity_coverage.csv').open('x',encoding='utf-8',newline='') as stream:
    rows=[]
    for record in proof['dataset_coverage']:
        rows.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in record.items()})
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(folder/'README.md').write_text('''# 原始实体范围核验：登记子集与原始来源不能视为同一数据范围

本次只读核验 246 个原始文件（660,983,211 字节）、191 个 mTSBench CSV（292,783,905 字节）和 NASA 压缩包的 164 个原始 train/test 成员。逐文件哈希、NPY 头部与完整载荷字节数、文本物理行数、标签元数据边界和实体集合已记录；没有改写原始数据、执行预处理或训练模型。

|数据集|原始不同实体|标签元数据行|当前 CSV 测试实体|原始不同实体测试点数|CSV 测试点数|
|---|---:|---:|---:|---:|---:|
|SMAP|54|55|51|435826|408160|
|MSL|27|27|26|73729|71418|
|SMD|28|不适用|18|708420|256589|

当前 SMAP 测试缺 D-12、D-13、P-7，P-7 只有训练 CSV；MSL 缺 T-9。SMD 缺 machine-1-1、machine-1-5、machine-2-6、machine-2-8、machine-3-1、machine-3-11、machine-3-4、machine-3-7、machine-3-8、machine-3-9。全部额外原始文件本地已经存在；尚未完成相应完整实体数值预处理与模型对照，不能把原队列跑完解释为原文全量实验跑完。

SMAP P-2 在两个本地 NASA 标签快照中都出现两次，分别给出 [5350,6575] 和 [5300,6420]；55 行对应 54 个不同通道。没有选择、合并、丢弃其中任何标签。两份 CSV 文件字节不同，但解析后的行顺序和完整元数据内容完全一致，不是新的标签内容冲突。162 个有标签的原始 train/test NPY 与压缩包成员逐字节相同。压缩包额外有 T-10 的 train/test 两个文件，但无对应标签，不把它当作已具备有监督测试数据。

当前 CSV 与原始数据尚未逐数值坐标、时间映射和标签对照。因此不能仅凭实体相交断言内容相同，也不能把两边点数差直接全部归因于缺少实体。未认证原文历史版本、端到端实验或标准 TAB 训练流程等价。

公开 Telemanom 说明称遥测值已按测试集 min/max 预缩放。后续新拟合仅训练集标准化能够隔离本地预处理拟合，却不能撤销来源文件原有缩放；必须保留这一边界。TranAD 本地公开预处理代码对 NASA 标签使用首条匹配记录和右端不包含切片，这与另一可能的包含端点解释需要显式对照，不能依据测试指标选择。

后续执行义务见同目录 JSON：完成全部原始实体的数值映射审计和独立准备，显式处理 P-2 标注/端点对照，再按原冻结算法与种子运行完整数据比较。它们补充现有 45 篇论文/681 方法记录/273 数据任务及其他新任务；当前数值队列继续运行，原子集结果与所有失败保留。

来源：[Telemanom 数据说明](https://github.com/khundman/telemanom#raw-experiment-data)、[TranAD](https://github.com/imperial-qore/TranAD)、[OmniAnomaly SMD](https://github.com/NetManAIOps/OmniAnomaly)、[mTSBench](https://huggingface.co/datasets/PLAN-Lab/mTSBench)。
''',encoding='utf-8')
obligations=dict(scope='Additional full-entity scientific experiments;existing registered-subset obligations remain',
    source_configuration=audit.CONFIG,evidence=folder.relative_to(ROOT).as_posix(),
    full_data_targets=dict(SMAP=dict(distinct_entities=54,metadata_rows=55),MSL=dict(distinct_entities=27),SMD=dict(distinct_entities=28)),
    tasks=[
        dict(id='raw_to_CSV_every_coordinate_and_label_mapping',status='not_run',requirements=['all shared and missing entities','original row identities','all channels','all role arrays','both original label endpoints and duplicate P-2 rows']),
        dict(id='full_raw_entity_preprocessing',status='not_run',requirements=['all available original entities','separate immutable F-drive outputs','fit-only local repair/scaling','original NASA prescaling disclosed','no source overwrite']),
        dict(id='full_raw_entity_all_registered_algorithm_and_ablation_models',status='not_run',requirements=['original algorithms/capacities/epochs/precision','frozen seeds and hyperparameters','all raw target datasets','full independent model inference and metrics','no test-based setting selection']),
        dict(id='explicit_P2_label_and_endpoint_sensitivity',status='not_run',requirements=['retain both conflicting source metadata rows','declare source-first-row/end-exclusive baseline','other labels/endpoints separate controls','do not count repeated channel as independent bootstrap group']),
        dict(id='full_cohort_and_protocol_comparison',status='not_run',requirements=['complete original seeds','grouped uncertainty','registered subset versus full source comparison','PA/oracle separated from validation-only metrics','author equivalence remains unproven until audited'])],
    all_original_experiments_complete=False)
write(folder/'additional_full_entity_execution_obligations.json',obligations)
write(folder/'archive_validation.json',dict(**interpretation,
    files_sha256={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/original_full_entity_coverage_and_pending_experiments.2026-10-16_v1.json'
write(ara,obligations)
paths=[*sources,ara.relative_to(ROOT).as_posix()]+[p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/original_entity_coverage_inventory_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(**interpretation,exact_files=len(paths))))
