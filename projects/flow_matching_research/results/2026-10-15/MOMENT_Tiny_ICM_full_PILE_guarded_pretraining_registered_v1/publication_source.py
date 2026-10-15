"""Archive actual complete-scope queue owners, tests and serialization proofs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_moment_icm_complete_pretraining_v1 as p

ROOT,read,sha,write=p.ROOT,p.read,p.sha,p.write
stamp=datetime.now(timezone.utc).isoformat()
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_Tiny_ICM_full_PILE_guarded_pretraining_registered_v1'
folder.mkdir(parents=True,exist_ok=False)
validation=ROOT/'tmp/MOMENT_ICM_full_guarded_training_integration_validation_20261015_v5.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==82
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
shutil.copyfile(validation,folder/'validation_82_checks.xml')
c=read(ROOT/p.CONFIG);body=p.verify(c);mc=read(ROOT/p.masks.CONFIG);p.masks.verify(mc)
assert c['settings']==read(ROOT/p.POLICY)['settings']
for label,path in [('frozen_two_model_pretraining_configuration',ROOT/p.CONFIG),('frozen_full_mask_preflight_configuration',ROOT/p.masks.CONFIG),
    ('unchanged_original_neural_GPU_resource_policy',ROOT/p.POLICY),
    ('actual_native_artifact_IO_checks',ROOT/'tmp/MOMENT_full_runtime_final_test_temp_20261015_v5/test_native_complete_source_ca0/native_artifact_check.json'),
    ('actual_native_mask_and_gradient_checks',ROOT/'tmp/MOMENT_full_runtime_final_test_temp_20261015_v5/test_native_full_architecture_0/native_check.json')]:
    shutil.copyfile(path,folder/(label+'.json'));assert sha(path)==sha(folder/(label+'.json'))
io=read(folder/'actual_native_artifact_IO_checks.json')
assert io['passed'] and io['complete_saved_model_optimizer_AMP_scheduler_RNG_exact'] and io['checkpoint_IO_consumes_no_training_RNG']
assert not io['native_model_reinference_or_full_scientific_training_executed']
failed=ROOT/'tmp/MOMENT_ICM_complete_pretraining_validation_20261015_v1.xml'
assert sum(int(s.get('failures',0)) for s in ET.parse(failed).getroot().iter('testsuite'))==2
write(folder/'preserved_pre_registration_guard_fixture_failure.json',dict(original_xml_utf8=failed.read_text(encoding='utf-8'),sha256=sha(failed),
    diagnosis='The process-memory fixture lacked vms; real psutil has vms. Added fixture field, without changing the source guard.',
    policy_review='Data preparation3GiB cap is distinct from original neural GPU policy. Before registration, adopted the complete existing SAITS neural policy unchanged. Source GPU worker observed private4,379,553,792bytes, already larger than the data cap.',
    scientific_attempt_started=False,original_resource_or_numerical_sources_modified=False))
states={}
modules=[p,p.masks,p.masks.inputs,p.masks.inputs.epoch,p.masks.inputs.finite,p.masks.inputs.finite.metadata]
for module in modules:
    config=read(ROOT/module.CONFIG);base=ROOT/config['state_root'];state=read(base/'status.json');proc=psutil.Process(state['pid'])
    assert proc.create_time()==state['create_time']
    key=module.__name__;states[key]=dict(status=state,exact_live_handle_checked_utc=stamp,command=proc.cmdline())
    if module in [p,p.masks]:
        assert state['active_pid'] is None and not Path(config['artifact_root']).exists()
        assert not (base/'worker_launch.json').exists()
        shutil.copyfile(base/'launch.json',folder/('actual_training_queue_launch.json' if module is p else 'actual_mask_audit_launch.json'))
write(folder/'actual_full_prerequisite_and_queue_owner_snapshots.json',states)
legacy={}
for name,pid,ct in [('NODE1104',36952,1792006037.8557332),('CrossAD_GECCO1104',41992,1791948094.8603492),('MOMENT_full78',22316,1792048301.2950852),('SAITS_full210',2972,1791918817.7624688)]:
    proc=psutil.Process(pid);assert proc.create_time()==ct
    legacy[name]=dict(pid=pid,create_time=ct,command=proc.cmdline(),cpu_seconds=sum(proc.cpu_times()[:2]),exact_live_checked_utc=stamp)
write(folder/'actual_original_queues_live_snapshot.json',legacy)
write(folder/'scope_and_status.json',dict(captured_utc=stamp,status='TWO_FULL_MODEL_JOBS_REGISTERED_WAITING_ACTUAL_COMPLETE_PREREQUISITES',
    jobs=c['jobs'],mask_audit_sources=2086,mask_audit_training_items=80850189,tests_passed=82,
    full_scientific_pretraining_models_started=0,full_scientific_pretraining_models_completed=0,
    numerical_training_work_has_not_been_counted_from_fixtures=True,
    checkpoints_save_full_model_optimizer_scaler_scheduler_and_RNG=True,
    full_original_source_GPU_guard_and_resource_lane_unchanged=True,
    full_data_mask_preflight_completed=False,complete_input_binding_completed=False,
    independent_checkpoint_model_reinference_pending=True,all_downstream_evaluations_pending=True,
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    all_original_experiments_complete=False))
(folder/'README.md').write_text('''# MOMENT Tiny / ICM：完整语料两模型训练队列已登记

已实现并启动冷元数据控制器。两个独立任务分别训练Tiny独立通道基线和ICM模型；每例完整2086源、80,850,189训练项、batch2048、39,479个完整/尾批、seed13和一整轮。不会以少源、少步、小批次或CPU/精度回退代替。训练复用已验证的原AdamW/MSE/AMP、掩码、裁剪、scheduler调用及本地有效目标/检查点上下文适配。

全部前置数据工作仍是硬条件：322控制准备和独立闭合、完整整轮索引、全2086输入绑定以及新登记的完整掩码兼容性审计。掩码审计检查每个训练案例，按原Monash洗牌选择并精确计入通道组；共用固定掩码的所有stride1窗口用解析计数完整覆盖。由原ceil(0.7*完整观测补丁数)证明任意原随机掩码都留有RevIN输入，不消耗Torch随机数。不丢案例，不发明归一化回退；若发现不兼容，保留失败并阻止训练。

神经网络任务沿用现有SAITS GPU队列完整策略：共享GPU锁、12GiB物理/28GiB提交准入、16GiB主机应急、16GiB GPU准入和4GiB GPU应急。GPU身份、原主机/GPU监控、观测重试及进程树处理保持。数据预检的3GiB进程上限只属于数据任务；登记前审查发现真实现有GPU任务private已超过该值，未错误套给神经网络训练。没有修改任何既有队列的门槛或运行进程。

82项相关检查通过。包含原guard执行语义一致性、GPU身份/应急分支、两模型完整预算拒绝缩减、冷控制器不导入Torch、全掩码解析审计、真实缓存尾批训练、梯度检查点修正及dropout RNG/梯度一致性、实际检查点完整IO。检查点保存模型、所有活跃AdamW状态/步数、AMP、scheduler、当前学习率和Python/NumPy/Torch CPU/CUDA随机状态；重新加载后逐项相等且不消耗训练随机数。训练项或步骤不足时拒绝生成最终完成证明。原周期检查点处于scheduler调用前，最终检查点处于最后scheduler之后，两者明确区分。

原源码触发的外部评估回调保存为明确待执行义务，未自动启动另一GPU，也未标记完成。最终训练状态成功后，独立原生模型重推理和下游论文/新数据集评估仍要实际执行。源epoch0 warmup后保持init_lr的规则保持；相关源码配方、初始化和数据控制不自动等于历史论文精确配方或严格TAB。

此包保存时，两新控制器均确认PID/创建时间/命令存活，但无数值子进程、无科学模型输出目录，实际整轮训练模型启动/完成数都为0。三案例和合成输入的测试不是benchmark或整轮训练。数据闭合与资源准入通过后，已登记入口才会自动启动完整模型。原NODE、CrossAD、SAITS、MOMENT队列仍存活，45篇/681方法消融记录/273数据任务及新增实验全部保留，研究目标未完成。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=stamp,files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()},
    tests_passed=82,full_scientific_training_complete=False,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_Tiny_ICM_full_PILE_pretraining_registered.2026-10-15_v1.json'
write(ara,dict(paper_id='moment_long_context',configuration=p.CONFIG,mask_preflight_configuration=p.masks.CONFIG,
    full_guarded_scientific_entrypoint_available=True,registered_models=2,
    status='REGISTERED_WAITING_ACTUAL_COMPLETE_DATA_AND_MASK_PROOFS',evidence=folder.relative_to(ROOT).as_posix(),
    full_scientific_models_started=0,full_scientific_models_complete=0,tests_passed=82,
    downstream_or_independent_replay_complete=False,all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),p.CONFIG,p.masks.CONFIG,
    'scripts/flow_matching/audit_moment_icm_full_training_masks_v1.py','tests/test_fm_moment_icm_full_training_masks_v1.py',
    'scripts/flow_matching/moment_icm_training_artifacts_v1.py','tests/test_fm_moment_icm_training_artifacts_v1.py',
    'scripts/flow_matching/run_moment_icm_complete_pretraining_v1.py','tests/test_fm_moment_icm_complete_pretraining_v1.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_complete_pretraining_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_publication_files=len(paths),tests_passed=82,full_model_jobs_registered=2,actual_full_models_started=0,
    all_original_experiments_complete=False)))
