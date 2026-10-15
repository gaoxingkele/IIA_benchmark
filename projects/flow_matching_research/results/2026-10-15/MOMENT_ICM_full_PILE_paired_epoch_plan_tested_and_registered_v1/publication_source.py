"""Archive complete-budget paired planner and reconcile existing ARA readiness."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as p

ROOT,read,sha,write=p.ROOT,p.read,p.sha,p.write
config=read(ROOT/p.CONFIG);sources=p.verify(config);budget=config['count_audit']
assert budget['total_items']==80850189 and budget['optimizer_batch_slots_per_full_epoch']==39479
assert len(sources)==2086 and len(config['excluded'])==76
base=ROOT/config['state_root'];handle=read(base/'controller_launch.json');status=read(base/'status.json')
owner=psutil.Process(handle['pid']);assert abs(owner.create_time()-handle['create_time'])<.001 and owner.cmdline()==handle['command']
assert handle['configuration_sha256']==sha(ROOT/p.CONFIG)
validation=ROOT/'tmp/MOMENT_ICM_disk_and_full_epoch_validation_20261015_v2.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==18
assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
primary=read(ROOT/'configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json')
targets_csv=ROOT/primary['output_directory']/'all702_paper_reported_metrics.csv'
with targets_csv.open('r',encoding='utf-8-sig',newline='') as f:assert sum(1 for _ in csv.DictReader(f))==702
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full_PILE_paired_epoch_plan_tested_and_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
for name,path in {
    'frozen_complete_epoch_plan_configuration.json':ROOT/p.CONFIG,
    'moment_icm_disk_epoch_plan_v1.py':ROOT/'scripts/flow_matching/moment_icm_disk_epoch_plan_v1.py',
    'prepare_moment_icm_full_epoch_plan_v1.py':ROOT/'scripts/flow_matching/prepare_moment_icm_full_epoch_plan_v1.py',
    'test_disk_epoch_plan_v1.py':ROOT/'tests/test_fm_moment_icm_disk_epoch_plan_v1.py',
    'test_full_epoch_plan_v1.py':ROOT/'tests/test_fm_moment_icm_full_epoch_plan_v1.py',
    'actual_controller_launch.json':base/'controller_launch.json',
    'validation_18_checks.xml':validation,
}.items():shutil.copyfile(path,folder/name);assert sha(path)==sha(folder/name)
write(folder/'actual_controller_status_snapshot.json',status)
write(folder/'full_epoch_source_and_step_budget.json',dict(**budget,batch_size=2048,source_epoch_seed=13,epochs=1,drop_last=False,
    batch_size_source=config['batch_size_origin'],channel_policy=config['channel_policy'],
    both_independent_and_ICM_must_share_the_same_references=True,
    old_eager_tuple_only_bytes_lower_bound=72417674*sys.getsizeof((0,0)),
    old_eager_lower_bound_is_analysis_not_a_failed_full_job=True,
    source_data_worker_memory_cap_bytes=3*2**30,
    whole_training_data_readiness_certified=False,neural_fit_inference_SVM=0))
failure_records=[]
for source,failures,explanation in [
    ('tmp/MOMENT_ICM_disk_epoch_plan_validation_20261015_v2.xml',1,
     'Native real-data fixture chose ACSF1,which exists in PILE157 but not downstream117. Fixture now uses registered Adiac with BasicMotions andArticularyWordRecognition. Sampler numerical order never changed.'),
    ('tmp/MOMENT_ICM_full_epoch_scope_validation_20261015_v1.xml',1,
     'New metadata scope checker initially did not reject a changed Monash selection label. Added explicit original count-field selector/case-split/guard checks and complete worker-body test before registration. No full planning worker attempt retried.')]:
    path=ROOT/source;suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('failures',0)) for s in suites)==failures
    failure_records.append(dict(path=source,sha256=sha(path),original_UTF8=path.read_bytes().decode('utf-8'),explanation=explanation,failures=failures))
write(folder/'preserved_pre_registration_validation_failures.json',dict(records=failure_records,
    all_corrections_before_full_scope_registration=True,no_registered_scientific_retry=True))
write(folder/'implementation_and_execution_status.json',dict(candidate_sources=2162,selected_source_references=2086,
    excluded_by_registered_original_gates=76,full_epoch_items=80850189,batch_slots=39479,
    code_complete_for_disk_reference_plan_and_exact_bijection_verifier=True,
    native_environment_real_source_sets_compared_to_old_planner=3,each_reference_batch_equal=True,
    real_batch_values_padding_and_provenance_byte_identical_for_checked_batches=True,
    synthetic_counterexamples_are_not_performance=True,tests_passed=18,
    actual_live_owner=dict(pid=handle['pid'],create_time=handle['create_time'],state=status['state']),
    full_actual80850189_reference_generation_complete=False,model_data_reader_integration_complete=False,
    full322_finite_control_cache_ready=False,original_ICM_pretraining_reproduced=False,new_neural_results=0,
    historical_pipeline_equivalence_certified=False,all_original_experiments_complete=False))

# Reconcile previously generated ARA prose with already verified primary scope.
paper_dir=ROOT/'projects/flow_matching_research/ara/papers/moment_long_context'
claims=paper_dir/'logic/claims.md';paper=paper_dir/'PAPER.md'
shutil.copyfile(claims,folder/'preserved_previous_ARA_claims.md');shutil.copyfile(paper,folder/'preserved_previous_ARA_PAPER.md')
text=claims.read_text(encoding='utf-8');start=text.index('**Statement**:',text.index('## C03'));end=text.index('\n\n**Conditions**:',start)
text=text[:start]+'''**Statement**: 原文七张表格的702个标量目标已转录并校验（正文72、附录630），对应原始数据及缓存准备有独立证据；仍未取得可核实的原MOMENT-Tiny/ICM预训练检查点，完整一轮预训练与全部下游预测/SVM尚未完成。完整输入规划保留2162个候选及2086个原门槛入选来源，322个缺失值来源的有限输入对照仍需实际准备与闭合；不能把原文目标、数据准备或官方MOMENT零样本异常检测复核当作本论文性能复现。''' +text[end:]
text=text.replace('**Conditions**: 数据实体、掩码、预算、阈值、PA、聚合及不确定性定义相同。',
    '**Conditions**: 对齐本论文的数据实体、通道分组、掩码、预训练预算、预测MSE/MAE、SVM准确率及不确定性定义；异常检测阈值/PA属于单独的任务适配。')
text += '''\n执行证据更新（C03状态仍为pending_experiment）：[原文702项目标与原始数据审计](../../../results/2026-10-15/MOMENT_ICM_primary_scope_and_raw_data_v1/README.md)、[四类输入完整一轮规划](../../../results/2026-10-15/MOMENT_ICM_full_PILE_paired_epoch_plan_tested_and_registered_v1/README.md)。Eq.(1)–(2)的所有通道压缩记忆、ELU+1及每头sigmoid门控与现有组件可逐项对应；历史作者插入、初始化和完整训练配方仍不以此认证。正文最大8通道抽样与附录D分组/余量过采样的描述差异保持为不同数据配方。\n'''
claims.write_text(text,encoding='utf-8')
paper.write_text('''# Towards Long-Context Time Series Foundation Models With A Handful Of Additional Parameters

研究任务：`forecasting_and_classification`；方法族：`compressive_channel_memory`。

四层入口：`logic/problem.md`、`logic/claims.md`、`logic/experiments.md`、`evidence/source/source_manifest.json`、`src/code/implementation_mapping.json`、`trace/exploration_tree.yaml`。

原文七张表的702个标量目标（正文72、附录630）已有逐项索引、原始数据检查及页码证据。原文Eq.(1)–(2)用ELU+1聚合所有通道KV、归一化查询，并在每头以sigmoid门控融合局部注意力；4层×4头的16个额外门控参数与已准备组件对应。原文窗口固定256，论文报告的上下文扩展包含变量维度。原作者完整插入/初始化语义及历史训练配方尚未认证。

数据准备已有157分类、9长预测、47 Monash来源索引；异常检测全部1949次原来源准备已尝试，1627个成功缓存独立闭合，其余322个已完整审计出残留非有限值，有限输入对照及有效目标掩码仍需实际闭合与完整训练集成。分类零样本评估的117个原数据集与预训练157候选来源保持不同范围。

新完整一轮数据规划保留2162候选、2086个原门槛入选来源，共80850189个case/group/window引用、39479个批次槽；包括322个待修复来源，没有通过删掉它们声称完整。附录D的8通道分组/余量过采样、分类casewise6/7和Monash原计数门槛均明确登记。batch2048/seed13来自相关来源YAML，论文未公开的精确预训练批次量不据此认证。磁盘规划保持旧RandomState抽样与尾批次，18项验证通过；实际完整索引生成及神经训练以各自成功证明为准。

原MOMENT-Tiny/ICM检查点、完整一轮预训练、所有上下文扩展基线、冻结编码器的预测头/β微调、117数据集SVM及严格预处理对照仍需完成。官方MOMENT small/base/large异常检测检查点的独立复核属于另一任务，不能替代本论文原预测/分类结果。

[原文目标与数据证据](../../../results/2026-10-15/MOMENT_ICM_primary_scope_and_raw_data_v1/README.md)，[最新完整一轮规划](../../../results/2026-10-15/MOMENT_ICM_full_PILE_paired_epoch_plan_tested_and_registered_v1/README.md)。完整研究目标尚未完成。
''',encoding='utf-8')

(folder/'README.md').write_text('''# 完整PILE一整轮配对引用：磁盘规划与精确范围已验证、登记

现有旧规划器对每个样本建立Python元组并保存整轮批次。仅已闭合异常检测来源的72,417,674个训练窗口，每个二元元组56字节的内存下界就达4,055,389,744字节，超过当前3 GiB数据工作器上限，尚未计算列表、整数及批次对象。这是代码/对象大小推导，不是已执行失败的全量任务。

新实现保留同一个RandomState逐宽度permutation调用与最后Python列表batch shuffle，逐块保存int64磁盘索引，一次只保留一个原生NumPy排列，不为全部样本创建Python元组。每个批次按前缀区间还原来源/样本引用，只映射必要连续索引块，读完即释放。所有样本和尾批次完整保留，不改batch、数据比例、模型、精度或预算来降低内存。

完整输入范围：2162个候选来源，原门槛入选2086个，保留76个不入选原因。入选为95分类、9个完整多变量长预测、33个原计数门槛Monash、1627原成功异常来源及322个明确有限输入对照来源。322项尚未全部准备和闭合，但引用数量已根据完整只读审计纳入；没有通过排除它们得到“完整”结果。

附录D数据配方下共80,850,189个case/group/window引用，batch2048的整轮共有39,479个批次槽：单通道78,908,611项/38,530批，末批1219；8通道1,941,578项/949批，末批74。张量时间坐标24,176,956,160、patch token位置3,022,119,520均包括重叠窗口、padding及余量重复通道，不是独立原始观测数量；有效重建目标数仍须结合完整有效目标mask核验。独立与ICM必须共享同一个配对索引。

batch2048、seed13来源于固定相关MOMENT预训练YAML；论文明确一轮和256窗口、附录D通道分组，却未公开精确预训练批次等全部设置。分类采用已披露casewise6/7，Monash保留原计数门槛，长预测使用完整多变量对照；正文“至多8通道抽样”是另一数据配方，不与当前分组方式静默混用。不能称历史完整ICM流水线认证。

18项验证通过：覆盖不同seed/batch/尾批次、所有索引恰好一次、int64及分块映射、重复索引重新哈希后拒绝、划分/完整预算/门槛拒绝，以及原Py3.10/NumPy1.25.2环境中Adiac、BasicMotions、ArticularyWordRecognition每个旧批次引用完全相同，抽查批次的数值/padding/身份字节相同；完整worker体在小引用夹具上生成并验证证明。夹具和局部真实数据检查都不是80,850,189项完整索引生成或神经性能。

两次登记前失败XML原字节保留：真实夹具误选仅属于PILE157而不在下游117的ACSF1，改用已登记Adiac；另一次新元数据检查未拒绝变更Monash选择标签，已补齐明确选择/原门槛检查，新增完整worker体检查。修正均在全量登记前，未放松数值容差或重试原科学任务。

实际保护控制器已启动，等待共享资源槽；沿用16/21/16 GiB门槛及3 GiB进程树上限。模型数据不在此阶段读取或生成，未调用神经模型。完整磁盘索引生成、322项数据修复/独立闭合、完整有效目标读取/训练循环、Tiny/ICM一整轮及全部下游仍需执行。新性能结果为0。至少45篇/681方法记录/273数据任务及新增实验范围不变，研究目标未完成。

ARA论文入口与C03已按真实证据更新：原文702项目标已经转录，仍没有本论文完整性能复现。保留更新前ARA原字节。论文：[2409.13530v1](https://arxiv.org/abs/2409.13530v1)，数据：[Time Series Pile](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)。大索引位于配置指定F盘。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=18,
    candidate_references=2162,selected_references=2086,full_epoch_items=80850189,batch_slots=39479,
    actual_full_reference_generation_complete=False,new_neural_results=0,
    files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_full_PILE_paired_epoch_plan.2026-10-15_v1.json'
write(ara,dict(configuration=p.CONFIG,evidence=folder.relative_to(ROOT).as_posix(),candidate_sources=2162,selected_sources=2086,
    paired_epoch_items=80850189,all_batch_slots=39479,tests_passed=18,
    actual_full_reference_generation_complete=False,full_model_reader_and_training_integration_complete=False,
    historical_ICM_pipeline_certified=False,new_neural_results=0,all_original_experiments_complete=False))
targets=[p.CONFIG,'scripts/flow_matching/moment_icm_disk_epoch_plan_v1.py','scripts/flow_matching/prepare_moment_icm_full_epoch_plan_v1.py',
    'tests/test_fm_moment_icm_disk_epoch_plan_v1.py','tests/test_fm_moment_icm_full_epoch_plan_v1.py',
    folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),claims.relative_to(ROOT).as_posix(),paper.relative_to(ROOT).as_posix()]
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_full_epoch_plan_publication_manifest_20261015_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(tests_passed=18,eligible_sources=2086,items=80850189,batch_slots=39479,owner_state=status['state'],exact_publication_files=len(paths),new_neural_results=0)))
