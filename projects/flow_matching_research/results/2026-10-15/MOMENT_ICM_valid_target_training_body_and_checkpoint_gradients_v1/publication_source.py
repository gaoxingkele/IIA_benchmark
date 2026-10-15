"""Freeze the tested callable integration; never declare full training done."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import moment_icm_masked_training_v1 as p
from scripts.flow_matching import bind_moment_icm_full_training_inputs_v1 as inputs

ROOT,read,sha,write=inputs.ROOT,inputs.read,inputs.sha,inputs.write
stamp=datetime.now(timezone.utc).isoformat()
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_valid_target_training_body_and_checkpoint_gradients_v1'
folder.mkdir(parents=True,exist_ok=False)
validation=ROOT/'tmp/MOMENT_ICM_masked_training_and_full_input_validation_20261015_v5.xml'
suites=list(ET.parse(validation).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==41
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
shutil.copyfile(validation,folder/'validation_41_checks.xml')
native=Path('C:/Users/10175/AppData/Local/Temp/pytest-of-admin/pytest-162/test_native_full_architecture_0/native_check.json')
proof=read(native)
assert proof['passed'] and proof['repaired_gradient_max_abs_difference']==0.
assert proof['original_reset_bug_gradient_max_abs_difference']>1e-4
assert proof['source_dropout_checkpoint_rng_and_gradients_bitwise_preserved']
assert proof['zero_valid_target_batch_keeps_source_optimizer_slot']
assert proof['actual_full_pretraining_epochs_executed']==0 and not proof['fixture_is_benchmark_performance']
shutil.copyfile(native,folder/'actual_native_torch_component_checks.json')
source=ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
names=['moment/utils/masking.py','moment/utils/optims.py','moment/utils/utils.py','moment/tasks/base.py','moment/tasks/pretrain.py',
    'moment/utils/data.py','moment/models/layers/embed.py','moment/models/layers/revin.py','moment/models/moment.py']
node=p.controlled_train_node(source/'moment/tasks/pretrain.py',sha(source/'moment/tasks/pretrain.py'))
(folder/'actual_isolated_training_method.py').write_text(ast.unparse(node)+'\n',encoding='utf-8')
config='configs/reproducibility/fm_moment_ICM_valid_target_training_body.2026-10-15_v1.json'
model='configs/models/fm_supplemental_moment_long_context_t5__reconstruction.json'
dependencies=[model,inputs.CONFIG,inputs.epoch.CONFIG,
    'configs/reproducibility/fm_moment_ICM_one_full_epoch_contract.2026-10-15_v1.json',
    'configs/experiments/fm_moment_pretrained_execution.v1.json',
    'scripts/flow_matching/moment_icm_masked_training_v1.py','tests/test_fm_moment_icm_masked_training_v1.py',
    'scripts/flow_matching/moment_icm_epoch_contract_v1.py','scripts/flow_matching/moment_icm_disk_epoch_plan_v1.py',
    'scripts/flow_matching/moment_icm_pile_cache_reader_v1.py',
    'src/iia_benchmark/models/fm_supplemental_moment.py','src/iia_benchmark/models/fm_supplemental.py',
    'src/iia_benchmark/models/fm_supplemental_author.py']
dependencies += [(source/n).relative_to(ROOT).as_posix() for n in names]
c=dict(schema_version=1,paper_id='moment_long_context',paper_citation='https://arxiv.org/abs/2409.13530v1',
    data_citation='https://huggingface.co/datasets/AutonLab/Timeseries-PILE',core_model_configuration=model,
    training_entrypoint='scripts.flow_matching.moment_icm_masked_training_v1:SourceTrainingTask',
    wrapper_entrypoint='scripts.flow_matching.moment_icm_masked_training_v1:masked_model',
    loader_entrypoint='scripts.flow_matching.moment_icm_masked_training_v1:CompleteEpochLoader',
    full_inputs_configuration=inputs.CONFIG,full_epoch_configuration=inputs.epoch.CONFIG,
    selected_sources=2086,epoch_items=80850189,batch_size=2048,full_epoch_batches=39479,epochs=1,seed=13,
    channel_policy='appendix_D_partition8_remainder_replacement',gradient_checkpointing=True,mask_ratio=.3,
    source_optimizer='AdamW',source_learning_rate=.0001,source_weight_decay=.05,source_max_norm=5.,
    source_torch_dtype='bfloat16',source_use_amp=True,source_warmup_steps=1000,source_warmup_lr=.00001,
    source_schedule='pinned_source_epoch0_warmup_then_init_lr;not_new_per_batch_cosine',
    required_source_callbacks=dict(local_logger=True,actual_checkpoint_and_final_state=True,
        explicit_downstream_evaluation_obligations=True),
    original_source_root=source.relative_to(ROOT).as_posix(),original_numerical_source_sha256={n:sha(source/n) for n in names},
    python=read(ROOT/'configs/experiments/fm_moment_pretrained_execution.v1.json')['python'],
    verified_native_environment=dict(python='3.10.21',torch='2.2.1+cu121',transformers='4.33.3',numpy='1.25.2',
        einops='0.7.0',pandas='2.2.2',scipy='1.9.3',scikit_learn='1.5.2',PyYAML='6.0.3'),
    source_receipts=[dict(path=name,sha256=sha(ROOT/name)) for name in sorted(set(dependencies))],
    reproduction_status='callable_complete_train_body_tested;guarded_full_scientific_entrypoint_pending',
    benchmark_ready=False,full_data_ready=False,full_guarded_training_job_registered=False,
    full_pretraining_executed=False,paper_performance_or_historical_initialization_certified=False,
    boundary=('Full2086/80,850,189items/39,479batches remain required. Only source stop condition '
        'plus mandatory raw-observation/source-finite target intersection, postbackward grouping release, '
        'and read-only finite-gradient audit differ in private training AST. Original optimizer/MSE/AMP/'
        'clip/scaler/scheduler/checkpoint order retained. Frozen files and live workers untouched. '
        'ICM grouping retained during checkpoint backward; no recurrent forward before backward. '
        'Zero valid-target batch keeps its original optimizer slot, with zero reconstruction loss. '
        'No retained input observations causes explicit failure,not an invented RevIN fallback or rowdrop. '
        'Synthetic/fullarchitecture and3realcase fixtures are plumbing,not scientific oneepoch or benchmark. '
        'Actual full masks/preflight,guarded resource CLI,final completecheckpoint/replay and all downstream '
        'tasks remain. Full source input hashes/322closure/epochplan not yet ready; no training launch inferred.'))
assert not (ROOT/config).exists();write(ROOT/config,c)
shutil.copyfile(ROOT/config,folder/'frozen_training_integration_configuration.json')
failures=[]
for version,diagnosis in [(1,'Native fixture imported unrelated benchmark package initializer, which required absent openpyxl. Namespaced exact model loading now avoids unrelated initializer; no package/environment changed.'),
    (2,'Fixture selected generic fm-author environment, which lacks transformers. It now selects the already configured MOMENT environment; neither environment modified.')]:
    path=ROOT/f'tmp/MOMENT_ICM_masked_training_integration_validation_20261015_v{version}.xml'
    assert any(int(s.get('failures',0)) for s in ET.parse(path).getroot().iter('testsuite'))
    failures.append(dict(sha256=sha(path),original_xml_utf8=path.read_text(encoding='utf-8'),diagnosis=diagnosis,
        failure_before_model_fixture_constructed=True,full_scientific_attempt_launched=False))
write(folder/'preserved_pre_registration_fixture_failures.json',dict(records=failures))
states={}
for queue in ['fm_moment_icm_all2086_training_input_bindings_v1','fm_moment_icm_anomaly_finite_control_cold_metadata_v4',
    'fm_moment_icm_all322_finite_control_closure_v1','fm_moment_icm_full_PILE_epoch_plan_v1','fm_moment_full78_checkpoint_replay_v1']:
    state=read(ROOT/'experiments/runs'/queue/'status.json');proc=psutil.Process(state['pid'])
    assert proc.create_time()==state['create_time']
    states[queue]=dict(status=state,exact_live_pid_and_create_time_checked_utc=stamp,command=proc.cmdline())
write(folder/'actual_original_owners_and_prerequisites_snapshot.json',states)
(folder/'README.md').write_text('''# MOMENT Tiny/ICM：有效目标训练体与梯度检查点修正

完整训练体已接通磁盘索引、分块真实缓存、原随机掩码及有效目标。私有AST仅保留已声明的一整轮停止条件，并新增三处：观测掩码乘以每通道/时间的有效目标、反向完成后释放通道上下文、梯度裁剪后只读核验有限性。优化器、MSE、AMP、scaler、裁剪、日志、检查点和scheduler的原调用顺序保留。原代码及运行中任务未修改。

修正本地纸面实现的一处潜在训练错误：原MomentICMT5在forward结束时把所有注意力层channels复位为1；但梯度检查点在backward才重计算，真实8通道会被错误地拆成8个单通道。适配层在backward之前恢复该批通道数，结束后释放，并拒绝重叠批次。论文尺寸4层/256维/4头/1024FF的CPU数值检查中，修复后与未开启检查点的前向和梯度完全相等；原错误版本最大梯度偏差120.83763122558594。这个数值是合成组件检查，不是模型性能改善。另以原dropout0.1验证RNG及梯度逐位相等。

41项相关检查通过。三例真实分类缓存经过完整原train体形成batch2和最后partial1，并实际核对AdamW步数、保存/严格加载状态及scheduler顺序。额外零有效目标批保留优化器和scheduler步骤，重建损失为0；不丢案例或偷偷减预算。原始观测缺失、源非法和padding目标均排除；若随机保留输入全部为空，显式报错，未添加未经验证的RevIN回退。

发现并保留一个容易误改的配方细节：源scheduler在epoch0先warmup，然后保持init_lr；在一整轮条件下不会进入后续epoch的cosine分支，不能擅自改成按batch做余弦衰减。该相关源码选择仍不等于未公开的历史论文精确超参数。

两次登记前测试夹具问题完整保留：无关benchmark包初始化要求openpyxl，以及通用fm-author环境无transformers。现使用不执行无关初始化的精确模型加载和已有MOMENT环境。未安装、升级或修改任何已有环境。

此包是可调用完整训练体及组件验证，不是完整科学训练任务已经启动。全2086源、80,850,189项、batch2048和39,479批一整轮范围保持。实际322控制独立闭合、全索引证明及全输入绑定仍在等待；全量掩码预检、资源保护入口、最终完整检查点/独立重推理和下游任务仍需完成。fixture的三案例和合成输入不可算成整轮预训练、benchmark或论文指标复现。原45篇/681方法记录/273数据任务及新增实验继续保留。
''',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=stamp,tests_passed=41,
    files_sha256={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()},
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    full_pretraining_executed=False,all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/MOMENT_ICM_masked_training_body.2026-10-15_v1.json'
write(ara,dict(paper_id='moment_long_context',configuration=config,evidence=folder.relative_to(ROOT).as_posix(),
    status='CALLABLE_TRAIN_BODY_AND_CHECKPOINT_GRADIENTS_TESTED',tests_passed=41,
    full_data_ready=False,guarded_full_training_runner_pending=True,full_pretraining_complete=False,
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),config,
    'scripts/flow_matching/moment_icm_masked_training_v1.py','tests/test_fm_moment_icm_masked_training_v1.py']
paths=set()
for args in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*args,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*args:subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_masked_training_body_publication_manifest_20261015_v1.json',
    dict(previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_publication_files=len(paths),tests_passed=41,full_pretraining_executed=False,all_original_experiments_complete=False)))
