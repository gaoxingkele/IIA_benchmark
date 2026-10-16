from datetime import datetime, timezone
from pathlib import Path
import ast
import csv
import json
import shutil
import subprocess
import warnings
import zipfile
import psutil
import torch
from scripts.flow_matching import run_tranad_author_all_entities_v1 as s
from iia_benchmark.models import tranad_author_source_v1 as runtime
from iia_benchmark.models.mtsad_detectors import TranADDetector

ROOT=s.ROOT;q=s.read(ROOT/s.CONFIG);s.verify(q)
base=ROOT/q['state_root'];launch=s.read(base/'launch.json')
proc=psutil.Process(launch['pid']);assert proc.create_time()==launch['create_time'] and proc.cmdline()==launch['command']
state=s.read(base/'status.json');assert state['pid']==proc.pid
torch.set_num_threads(2)
model_config=s.read(ROOT/(s.MODEL_PREFIX+'TranAD.json'))
source=ROOT/q['source_root']
models=[]
for variant in s.VARIANTS:
    model,ns=runtime.author_model(s.read(ROOT/(s.MODEL_PREFIX+variant+'.json')),'SMAP',variant,25)
    models.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
        LayerNorm_modules=sum(isinstance(x,torch.nn.LayerNorm) for x in model.modules()),
        model_class_source_file=model.forward.__code__.co_filename,
        model_forward_first_line=model.forward.__code__.co_firstlineno,
        source_learning_rate=model.lr,parameter_dtype=str(next(model.parameters()).dtype),full_training_batch=model.batch))
with warnings.catch_warnings(record=True) as caught:
    local=TranADDetector(device='cpu')._build(25).double()
local_norms=sum(isinstance(x,torch.nn.LayerNorm) for x in local.modules())
assert local_norms==8 and all(m['LayerNorm_modules']==0 for m in models)
_,namespace=runtime.author_model(model_config,'SMAP','TranAD',3)
namespace['TransformerEncoder']=torch.nn.TransformerEncoder
namespace['TransformerDecoder']=torch.nn.TransformerDecoder
native_failure=None
try:
    with warnings.catch_warnings(record=True):
        old_container_model=namespace['TranAD'](3).double();old_container_model.eval()
        x=torch.zeros(10,2,3,dtype=torch.float64)
        old_container_model(x,x[-1:])
except Exception as error:
    native_failure=dict(error_type=type(error).__name__,message=str(error))
assert native_failure is not None
differences=[
    dict(component='normalization',old_local='8 PyTorch standard LayerNorm modules',new_source='0 LayerNorm; unchanged author custom residual layers',
        evidence_source='src/dlutils.py:212-251',evidence_local='src/iia_benchmark/models/mtsad_detectors.py:1279-1302'),
    dict(component='activation',old_local='Standard Transformer layer default ReLU',new_source='Exact author literal nn.LeakyReLU(True), preserved without reinterpretation',
        evidence_source='src/dlutils.py:222 and243',evidence_local='nn.TransformerEncoderLayer/DecoderLayer constructors'),
    dict(component='position_encoding',old_local='Alternating sine/cosine dimensions',new_source='Both sin and cos added in every dimension with author divisor',
        evidence_source='src/dlutils.py:196-210',evidence_local='src/iia_benchmark/models/mtsad_detectors.py:1261-1274'),
    dict(component='learning_rate',old_local='SMAP/MSL/SMD configured .01',new_source='SMAP .001, MSL .002, SMD .0001 from exact constants.py',
        evidence_source='src/constants.py:lr_d',evidence_local='configs/models/fm_strict_baselines/fm_strict_tranad__smap__reference_float64.json'),
    dict(component='training_order',old_local='Seeded torch.randperm every epoch',new_source='Original sequential DataLoader with no shuffle argument',
        evidence_source='main.py:251-273',evidence_local='src/iia_benchmark/models/mtsad_detectors.py:1344-1350'),
    dict(component='inference_batch',old_local='Local inference batch128',new_source='Original one complete population batch, no chunking',
        evidence_source='main.py:254',evidence_local='src/iia_benchmark/models/mtsad_detectors.py:1374'),
    dict(component='window_and_preprocessing',old_local='Local grouped/window grid and prepared inputs',new_source='Exact preprocess.py; original previous-row windows plus repeated-first-row padding; per-entity training',
        evidence_source='preprocess.py, main.py:convert_to_windows andload_dataset',evidence_local='Original local full109 and constant_fit variants kept separate'),
    dict(component='threshold_reporting',old_local='Validation percentiles and independent full TAB metric controls',new_source='Original per-feature and average-score POT with test-label PA, original Hit/NDCG',
        evidence_source='main.py:331-348, src/pot.py',evidence_local='Existing frozen local scores/metric protocols unchanged')]
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TranAD_author_all109_five_branches_source_execution_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n*.zip -text\n',encoding='utf-8')
for name,path in dict(actual_launch=base/'launch.json',actual_controller_state=base/'status.json',
    full3270_frozen_queue=ROOT/s.CONFIG).items():shutil.copyfile(path,packet/(name+'.json'))
shutil.copyfile(launch['tests_path'],packet/'relevant_validation.xml')
s.write(packet/'source_fidelity_audit.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    original_author_commit=q['author_commit'],source_variants=models,local_LayerNorm_modules=local_norms,
    source_with_unadapted_modern_torch_container_actual_failure=native_failure,
    original_class_forward_and_backprop_AST_unmodified=True,
    only_current_torch_container_call_compatibility_restored=True,
    observed_implementation_and_protocol_differences=differences,
    old_local_low_F1_explained_causally=False,paper_table_variant_mapping_certified=False,
    historical_environment_certified=False,full_author_training_conditions_complete=len(state['completed_ids']),
    synthetic_diagnostics_are_benchmark_performance=False,all_original_experiments_complete=False))
with (packet/'all3270_original_source_condition_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(q['jobs'][0])+['execution_status']);writer.writeheader()
    for job in q['jobs']:
        status='complete_train_fresh_replay_original_metrics' if job['id'] in state['completed_ids'] else 'failed_preserved' if job['id'] in state['failed_preserved'] else 'pending_original_preparation_or_resource_admission'
        writer.writerow(dict(job,execution_status=status))
original_sources=[source/p['path'] for p in model_config['source_receipts']]
with zipfile.ZipFile(packet/'unchanged_original_author_source_and_license.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in original_sources:archive.writestr(path.relative_to(source).as_posix(),path.read_bytes())
with zipfile.ZipFile(packet/'unchanged_original_author_source_and_license.zip') as archive:
    assert archive.testzip() is None
    for path in original_sources:assert archive.read(path.relative_to(source).as_posix())==path.read_bytes()
s.write(packet/'old_source_and_local_tracks_unchanged.json',dict(source_receipts=q['source_receipts'],
    old_model_source_path='src/iia_benchmark/models/mtsad_detectors.py',old_model_source_sha256=s.sha(ROOT/'src/iia_benchmark/models/mtsad_detectors.py'),
    old_FP64_adapter_sha256=s.sha(ROOT/'src/iia_benchmark/models/fm_tranad_reference_float64_v1.py'),
    all_old_float32_float64_results_reclassified=False,original_numerical_workers_stopped=False))
s.write(packet/'iteration_status.json',dict(original_scope=q['original_scope'],full_author_source_conditions_required=3270,
    original_entities=109,original_source_branches=5,primary_seeds=q['primary_seeds'],source_default_seed=1,
    primary_five_seed_cohorts_required=545,source_default_conditions=545,original_CLI_default_entity_conditions=90,
    complete_conditions=len(state['completed_ids']),complete_original_performance=False,
    controller_actual_pid=proc.pid,controller_actual_create_time=proc.create_time(),
    current_state=state['state'],preprocessing_and_real_preflight_complete=s.preparation_report(q) is not None,
    all_original_experiments_complete=False))
(packet/'README.md').write_text('''# TranAD作者源码：五分支、全部109实体的独立原始执行轨道

现有TranAD FP64结果修复了精度，但并非完整作者结构。实际核验旧模型含8个LayerNorm，而原作者自定义层没有LayerNorm；位置编码、激活、训练顺序、学习率、窗口和阈值流程也有明确差异。SMAP学习率旧配置为0.01，作者为0.001；MSL为0.002，SMD为0.0001。不能把旧轨道低F1归因于PA或数据集难度，也不能未经实验宣称这些差异必然解释性能差距。

新增入口从固定提交原文件AST直接执行TranAD、TranAD_Basic、TranAD_Transformer、TranAD_Adversarial、TranAD_SelfConditioning五个原始类、forward和backprop。未重写层、位置编码、激活或损失。当前PyTorch标准容器对作者自定义层的原始接口出现真实兼容错误，记录于source_fidelity_audit.json；兼容入口使用旧容器的顺序层调用和相同state_dict层名。未补入LayerNorm，也未把原nn.LeakyReLU(True)改成另一种激活。

五分支的模型配置、可调用源码入口、原始数据/文献引用和27项相关检查均已准备，包含源函数五轮训练、优化器/调度器状态和完整数组重放的生命周期检查。短合成输入仅验证程序，不是基准成绩。控制器已实际启动；归档时尚无完整作者轨道性能，不把注册和测试通过计为实验完成。

完整矩阵为54SMAP＋27MSL＋28SMD＝109实体，每实体五个分支；保留源文件默认torch种子1，并另跑固定1103—1107五种子，共3270条件。主要五种子组545个，另有545个源默认种子对照。作者CLI默认P-1、C-1和machine-1-1共90条件优先，仍只是完整矩阵的子集，不替代全部实体。仓库分支名与论文精确消融表绑定尚未认证。

预处理直接执行原preprocess.py的SMAP/MSL/SMD分支，保存到F盘新目录，原始数据不覆盖。包括原训练集归一化、原首条metadata和右端不含标签、SMD原每特征解释标签，以及源previous-row窗口/首行填充。完整源预算五轮、batch128顺序不打乱、FP64、数据集原学习率、AdamW与StepLR、全体一次推理均保留，未用chunk、截断或少轮替代。源数据历史同字节版本及论文环境认证仍保留边界。

每个条件依次经过训练守卫、新解释器检查点完整数组逐字节重放守卫、原始POT/PA及Hit/NDCG守卫。原ROC/AUC字段是PA二值预测的AUC，不能写成连续分数AUROC。PA明确使用测试标签，非有限原始指标保留null和状态；没有用测试F1挑阈值。失败和资源中止保留，不自动重跑。

准备阶段沿用既有16GiB物理/28GiB提交门槛、原只读共享锁和3/6GiB进程上限；完整源训练/整批推理以24GiB物理/40GiB提交门槛、16GiB紧急余量、10/12GiB进程上限登记，复用已有完整源CPU训练锁。未降低其他队列门槛或停止数值进程。当前资源不足时保持等待，不能把等待写成已训练。

旧float32/FP64、原873、common105、已完成TAB指标报告和失败证据全部保留，新轨道不覆盖旧配置或强行认证旧结果。全部45论文/681方法记录/273数据任务/702原始指标及原论文、TAB、新数据整体目标未完成。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'publication_source.py')
s.write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=s.sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/TranAD_author_all109_five_branches_source_execution.2026-10-16_v1.json'
assert not ara.exists()
s.write(ara,dict(result_packet=packet.relative_to(ROOT).as_posix(),actual_controller_pid=proc.pid,required_source_conditions=3270,
    complete_conditions=len(state['completed_ids']),historical_author_or_paper_equivalence=False,all_original_experiments_complete=False))
paths=[s.CONFIG,s.TEST,'src/iia_benchmark/models/tranad_author_source_v1.py',s.MODULE.replace('.','/')+'.py']
paths += [s.MODEL_PREFIX+v+'.json' for v in s.VARIANTS]+[ara.relative_to(ROOT).as_posix(),'projects/flow_matching_research/README.rst']
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
head=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD']).decode().strip()
s.write(ROOT/'tmp/TranAD_author_all_entities_manifest_20261016_v1.json',dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(packet=str(packet),source_models=models,modern_container_failure=native_failure,actual_controller=proc.pid,
    actual_state=state['state'],required_conditions=3270,complete_conditions=len(state['completed_ids']),files=len(paths))))
