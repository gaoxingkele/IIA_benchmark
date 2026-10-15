"""Start fresh full-data metadata owner; archive actual TimesNet full closure."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import prepare_original_full_entity_tsad_v1 as data
from scripts.flow_matching import run_timesnet_SMAP_remaining_full_audits_v2 as audit

ROOT,read,write,sha=data.ROOT,data.read,data.write,data.sha
configuration=read(ROOT/data.CONFIG);jobs=data.verify(configuration)
xml=ROOT/'tmp/original_full109_entity_preparation_validation_20261016_v1.xml'
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==21
assert all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
base=ROOT/configuration['state_root'];assert not base.exists()
for p in psutil.process_iter(['cmdline']):assert data.MODULE not in (p.info['cmdline'] or [])
base.mkdir(parents=True,exist_ok=False)
command=[sys.executable,'-B','-X','utf8','-u','-m',data.MODULE]
with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command,configuration_sha256=sha(ROOT/data.CONFIG))
write(base/'launch.json',handle)
for _ in range(70):
    assert child.poll() is None,'Preserve actual source failure and diagnose without restart'
    if (base/'status.json').exists():break
    time.sleep(.1)
state=read(base/'status.json')
assert state['pid']==handle['pid'] and state['create_time']==handle['create_time']
assert psutil.Process(handle['pid']).cmdline()==command
folder=ROOT/'projects/flow_matching_research/results/2026-10-16/original_full109_entity_preparation_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
copies=dict(frozen_configuration=ROOT/data.CONFIG,original_implementation=ROOT/(data.MODULE.replace('.','/')+'.py'),
    original_tests=ROOT/data.TEST,validation_21_checks=xml,actual_controller_launch=base/'launch.json',status_at_registration=base/'status.json')
for name,path in copies.items():
    target=folder/(name+path.suffix);shutil.copyfile(path,target);assert sha(path)==sha(target)
summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),entities=109,datasets=dict(SMAP=54,MSL=27,SMD=28),
    original_registered_CSV_files_to_audit=191,test_label_controls=7,tests_passed=21,actual_controller=handle,
    all_raw_preprocessing_complete=False,neural_models_trained=0,original_resource_policy_unchanged=True,
    original_dataset_and_existing_experiment_queues_changed=False,all_original_experiments_complete=False)
write(folder/'registration_summary.json',summary)
(folder/'README.md').write_text('''# 完整原始 109 实体准备已登记

全量范围为 SMAP 54 个不同通道、MSL 27 个、SMD 28 个，共 109 个实体，后续另与现有 191 个 CSV 逐坐标和标签对照。程序已实现并通过 21 项测试，控制进程已启动；目前等待原 CPU 数据资源通道，尚未宣称全部原始数据处理完成或产生新模型成绩。

新输出全部位于 F 盘独立目录。源文件哈希在各实体处理前后核验，保存完整原始行号；每个归一化输入坐标用独立统计核对路径重新构建，写出的 NPZ 再完整重读。保持 fit-only 修复、变动通道均值/标准差和常数通道精确首值/单位缩放。按原 80% 训练切点、100 点隔离区，新增规则只对不足 100 点验证段的短序列向前移动切点：D-12 的 312 个原训练点得到 112 fit/100 purge/100 validation；T-9 的 439 点得到 239/100/100。其余实体保持原切点，原有队列、数据和预算不变。这是新登记的完整实体对照划分，不冒充论文原始划分。

NASA 主标签固定为来源代码首条匹配记录、右端不包含切片；SMAP 保存首条/末条元数据 × 包含/不包含右端点四种标签，MSL 保存两种端点标签，SMD 保留来源二值点标签，共七个标签版本。标签覆盖文件不改变训练、验证输入和模型分数，不按测试成绩选标签，也不把重复 P-2 计作独立实体。

原始 NASA 公开值的测试集 min/max 预缩放仍明确披露；本地仅训练集拟合不能撤销来源预缩放。CSV 与原始数值的映射尚待实际 worker 核验；程序比较原始 TRAIN/TEST 两种角色及零/一基时间戳，报告无匹配、近似差异和精确匹配，不猜测或隐去不同数据来源。

沿用已登记数据任务的 16 GiB 可用物理内存、21 GiB 提交余量、16 GiB 紧急余量、3 GiB RSS/private 上限及原 CPU 锁，无资源门槛放宽。测试仅验证数据处理契约和实现，不是 benchmark 性能。完整原始实体上的模型训练、全量模型重推理和五种子统计仍待后续完成；所有其他论文、消融与现有队列义务保留。

来源：[Telemanom](https://github.com/khundman/telemanom)、[TranAD](https://github.com/imperial-qore/TranAD)、[OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)、[mTSBench](https://huggingface.co/datasets/PLAN-Lab/mTSBench)。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(**summary,files_sha256={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}))
ara=ROOT/'projects/flow_matching_research/ara/execution/original_full109_entity_preparation_registered.2026-10-16_v1.json'
write(ara,dict(entities=109,label_controls=7,implementation_tests_passed=21,evidence=folder.relative_to(ROOT).as_posix(),
    actual_raw_preparation_complete=False,all_original_experiments_complete=False))

ac=read(ROOT/audit.CONFIG);audit.verify(ac);job=next(j for j in ac['jobs'] if j['seed']==1104)
closed=ROOT/'projects/flow_matching_research/results/2026-10-16/TimesNet_SMAP1104_full_two_stage_closed_v2'
closed.mkdir(parents=True,exist_ok=False);proofs={};guards={}
for mode in ['point','native']:
    stage=audit.stage_config(ac,job,mode);audit.binding(stage)
    source=Path(stage['output_directory'])/'independent_full_saved_score_verification.json'
    p=read(source);attempt=ROOT/stage['state_root'];guard=read(attempt/'receipt.json')
    assert p['passed'] and p['canonical_id']==job['id'] and p['mode']==mode
    assert p['configuration_sha256']==sha(attempt/'frozen_stage.json') and guard['proof_sha256']==sha(source)
    assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
    assert p['full_entities']==51 and p['full_test_points']==408160 and len(p['coverage'])==153
    assert p['all_saved_streams_verified'] and p['all_original_point_core_event_bootstrap_metrics_exact']
    if mode=='native':
        assert p['full_native_checkpoint_reinference_passed'] and p['all_replayed_timestamp_scores_bitwise_equal']
        assert sha(Path(stage['output_directory'])/'replayed_scores.npz')==p['replay_scores_sha256']
    proofs[mode]=p;guards[mode]=guard
    for name,path in [('proof',source),('successful_guard',attempt/'receipt.json'),('frozen_stage',attempt/'frozen_stage.json'),('launch',attempt/'launch.json')]:
        shutil.copyfile(path,closed/(mode+'_'+name+'.json'))
assert proofs['point']['independently_recomputed_primary_metrics']==proofs['native']['independently_recomputed_primary_metrics']
assert proofs['point']['conditional_entity_cluster_f1_ci95']==proofs['native']['conditional_entity_cluster_f1_ci95']
result=read(ROOT/job['output_directory']/'result.json')
assert sha(ROOT/job['output_directory']/'model.pt')==result['checkpoint_sha256']
assert sha(ROOT/job['output_directory']/'scores.npz')==result['scores_sha256']
assert proofs['native']['original_result_sha256']==sha(ROOT/job['output_directory']/'result.json')
shutil.copyfile(ROOT/job['output_directory']/'result.json',closed/'original_completed_result.json')
shutil.copyfile(ROOT/audit.CONFIG,closed/'frozen_remaining4_registration.json')
parent_state=read(ROOT/ac['state_root']/'status.json');owner=psutil.Process(parent_state['pid'])
assert owner.create_time()==parent_state['create_time']
assert all(dict(id=job['id'],mode=m) in parent_state['completed'] for m in ['point','native'])
write(closed/'actual_controller_status.json',parent_state)
p=proofs['native'];metrics=p['independently_recomputed_primary_metrics']
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),id=job['id'],seed=1104,entities=51,test_points=408160,
    parameters=28132633,full_point_and_native_review_passed=True,all153_roles_and_timestamp_scores_bitwise_equal=True,
    primary_validation_only_noPA_metrics=metrics,conditional_entity_bootstrap_CI=p['conditional_entity_cluster_f1_ci95'],
    test_calibrated_oracle_diagnostic=p['original_test_calibrated_TAB_core_diagnostic'],
    configured_epochs=3,persisted_actual_epoch_update_counter_available=False,required_original_seeds=[1103,1104,1105,1106,1107],
    fully_closed_original_seeds=[1103,1104],complete_five_seed_aggregate=False,
    full_TAB_range_metrics_completed=False,canonical_NASA_full_entity_equivalence_certified=False,
    historical_author_equivalence_certified=False,new_training_seed_slots=0,all_original_experiments_complete=False)
write(closed/'full_two_stage_closure_summary.json',report)
(closed/'README.md').write_text(f'''# TimesNet SMAP seed1104：两阶段全量审计成功

原 28,132,633 参数模型成功完成原登记训练，保存分数独立复核与完整检查点重新推理均已成功退出，无资源中止。实际覆盖 51 个本地登记实体、153 个 TRAIN/VAL/TEST 角色流和 408,160 个 TEST 时间戳；所有重推理分数与原分数逐字节一致，原检查点、分数与结果当前哈希再次核对。没有新增训练种子、拟合、降低容量或放宽容差。

固定验证集 1% 阈值、无 PA：F1={metrics['f1']:.12f}，AUROC={metrics['auroc']:.12f}，AP={metrics['average_precision']:.12f}。全部 0.1/1/5% 阈值结果、原事件指标、bootstrap 及测试校准诊断也逐项核对一致。实体簇区间是当前单模型条件区间，不能冒充完整五种子统计。测试标签挑阈值和 PA 诊断单独保存，不作为严格主指标。

原五种子队列现有 seed1103、1104 两个完整关闭，1105..1107 仍待原训练及两阶段审计。这里只对原本地 51 实体子集成立，不认证原 NASA 全量或历史作者流程；实际 epoch/更新轨迹未保存，3 轮仅为配置和预算保留回执。完整 TAB 范围指标及全研究其他实验仍未完成。

来源：[TimesNet](https://arxiv.org/abs/2210.02186)、[TAB](https://github.com/decisionintelligence/TAB)。
''',encoding='utf-8')
(closed/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,closed/'publication_source.py')
write(closed/'archive_validation.json',dict(full_source_proofs_and_current_array_hashes_verified=True,
    files_sha256={p.relative_to(closed).as_posix():sha(p) for p in closed.rglob('*') if p.is_file()},all_original_experiments_complete=False))
other_ara=ROOT/'projects/flow_matching_research/ara/execution/TimesNet_SMAP1104_full_two_stage_closed.2026-10-16_v2.json'
write(other_ara,dict(id=job['id'],evidence=closed.relative_to(ROOT).as_posix(),full_native_replay_passed=True,
    full_saved_score_review_passed=True,complete_five_seed_aggregate=False,all_original_experiments_complete=False))
paths=[data.CONFIG,data.TEST,data.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix(),other_ara.relative_to(ROOT).as_posix()]
for root in [folder,closed]:paths += [p.relative_to(ROOT).as_posix() for p in root.rglob('*') if p.is_file()]
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/full109_and_TimesNet1104_publication_manifest_20261016_v1.json',dict(previous_root=git(ROOT,'rev-parse','HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(full109_controller=handle,data_preparation_state=state,tests_passed=21,
    TimesNet1104_actual_full_native_closed=True,TimesNet1104_metrics=metrics,exact_files=len(paths))))
