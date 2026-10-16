from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import evaluate_tranad_original_SPOT_full30_v1 as s

ROOT=s.ROOT;c=s.read(ROOT/s.CONFIG);s.verify(c)
base=ROOT/c['state_root'];launch=s.read(base/'launch.json');state=s.read(base/'status.json')
proc=psutil.Process(launch['pid'])
assert proc.is_running() and proc.create_time()==launch['create_time']
assert state['pid']==proc.pid and state['create_time']==proc.create_time()
assert not state['failed_preserved']
xml=Path('F:/aicoding/IIA_Data/experiments/TranAD_original_SPOT_full30_final_20261016_v1_tests.xml')
suites=list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(x.get('tests',0)) for x in suites)==27
assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in suites)
failed=Path('F:/aicoding/IIA_Data/experiments/TranAD_original_SPOT_full30_pre_registration_20261016_v1_tests.xml')
old=list(ET.parse(failed).getroot().iter('testsuite'))
assert sum(int(x.get('failures',0)) for x in old)==1
ready=[]
for record in c['entries']:
    b=s.ready(record)
    if b:ready.append(dict(case=record['entry']['case'],full_entities=len(b['dataset']['entities']),source_lineage=b['lineage']))
oldtool=ROOT/'scripts/mtsad/score_pot.py';text=oldtool.read_text(encoding='utf-8')
assert 'np.quantile(finite, 1.0 - level)' in text
assert 'initial + sigma * np.log(risk * n / exceedance_count)' in text
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TranAD_original_SPOT_all25_unique_30slots_execution_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for source,dest in [(xml,'relevant_validation.xml'),(failed,'preserved_pre_registration_validation_failure.xml'),
    (base/'launch.json','actual_controller_launch.json'),(base/'status.json','captured_actual_controller_state.json'),
    (ROOT/s.CONFIG,'frozen_configuration.json'),(ROOT/s.TEST,'validation_source.py'),
    (ROOT/(s.MODULE.replace('.','/')+'.py'),'execution_source.py')]:shutil.copyfile(source,packet/dest)
s.write(packet/'actual_ready_source_bindings.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),ready_cases=ready,
    ready_does_not_mean_SPOT_controls_complete=True))
s.write(packet/'historical_approximation_fault_evidence.json',dict(path=oldtool.relative_to(ROOT).as_posix(),sha256=s.sha(oldtool),
    faults=[dict(old='quantile(finite,1-level)',source='sorted(init_data)[int(level*n)]',effect='opposite_tail_initialization'),
            dict(old='initial+sigma*log(risk*n/Nt)',source='initial-sigma*log(risk*n/Nt)',effect='reversed_zero_shape_extreme_quantile')],
    old_source_and_records_preserved=True,new_lane_uses_unmodified_author_Grimshaw_and_POT=True,
    scipy_genpareto_fit_is_not_original_Grimshaw=True,original_source_PA_index0_and_epsilon_retained=True,
    author_source_files=[p for p in c['source_receipts'] if '/tranad_baselines/original/' in p['path']]))
complete=list(state['completed_ids'])
for identity in complete:
    report=Path(c['artifact_root'])/identity/s.REPORT
    guard=base/'attempts'/identity/'guard_receipt.json'
    a,b=s.read(report),s.read(guard)
    assert a['passed'] and b['passed'] and b['exit_code']==0 and not b['resource_guard_aborted'] and b['proof_sha256']==s.sha(report)
    shutil.copyfile(report,packet/(identity+'.json'));shutil.copyfile(guard,packet/(identity+'.guard.json'))
s.write(packet/'iteration_status.json',dict(original_scope=c['original_scope'],required_unique_cases=25,required_comparison_slots=30,
    fullraw_unique_cases=15,matched_unique_cases=10,shared_original_SMD_comparison_slots=5,
    complete_original_SPOT_cases=len(complete),actual_live_controller=dict(pid=proc.pid,create_time=proc.create_time(),command=proc.cmdline()),
    relevant_tests=27,initial_failed_pre_registration_scope_check_preserved=True,
    source_model_training_or_inference_performed=False,all_original_experiments_complete=False))
(packet/'README.md').write_text(f'''# TranAD 原始SPOT：25个独立案例、30个比较位置

已冻结并启动完整队列：原始数据轨15个案例，matched105轨10个SMAP/MSL案例；matched轨的5个SMD位置明确引用原始SMD案例，不伪造新增运行。所有主训练种子1103至1107和65个实际标签视图保留。文件名full30代表比较位置，独立运行数以配置required_cases=25为准。

旧scripts/mtsad/score_pot.py使用1-level初始化且零形状参数分支对数符号相反；与原作者SPOT的排序分位及Grimshaw拟合不同。旧文件、冻结配置和历史结果保留，新入口直接执行冻结作者src/spot.py与src/pot.py，保留原始PA的index0行为、epsilon和二值预测ROC/AUC语义。该ROC/AUC不能冒充连续分数AUROC。

只用每个实体完整保存的fit-training分数校准一次，q=1e-5；所有标签视图共用阈值，不以测试标签调阈值。PA仍使用测试标签，需作为作者协议控制单列。常量训练分数明确记为无法拟合，绝不回退百分位。完整原始实体、源/native/标签复核及资源守卫全部要求。

模型仍是旧本地FP64架构与预处理；这是固定模型上的原作者校准对照，不能替代已启动的109实体、5分支、3270条件原作者模型训练，也不证明完整TAB或历史论文性能复现。27项相关检查通过；初版错误假设matched轨也有15个独立案例被测试拒绝，原失败XML保留，依据冻结注册表修正为15+10及5个共享引用后才注册和启动，没有数值运行被重试或覆盖。

捕获时完整SPOT案例为{len(complete)}/25；运行中的精确控制器见launch/state记录。资源门槛与共享只读锁沿用原策略，没有降低门槛或停止现有数值任务。准备就绪、测试通过、队列启动均不算基准实验完成。全部45篇论文、681方法记录、273数据任务、702指标及所有原论文/TAB/新数据任务继续保留。
''',encoding='utf-8')
shutil.copyfile(__file__,packet/'packaging_source.py')
s.write(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=s.sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara=ROOT/'projects/flow_matching_research/ara/execution/TranAD_original_SPOT_all25_unique_30slots.2026-10-16_v1.json'
assert not ara.exists();s.write(ara,dict(packet=packet.relative_to(ROOT).as_posix(),configuration=s.CONFIG,configuration_sha256=s.sha(ROOT/s.CONFIG),
    required_unique_cases=25,comparison_slots=30,actual_complete_cases=len(complete),all_original_experiments_complete=False))
cross=ROOT/'projects/flow_matching_research/results/2026-10-16/CrossAD_row4_seed1104_full16_v1'
assert s.read(cross/'iteration_status.json')['CrossAD_full_CUDA_conditions']==16
crossara='projects/flow_matching_research/ara/execution/CrossAD_row4_seed1104_full16.2026-10-16_v1.json'
paths=[s.CONFIG,s.TEST,s.MODULE.replace('.','/')+'.py',ara.relative_to(ROOT).as_posix(),crossara,'projects/flow_matching_research/README.rst']
paths+=[p.relative_to(ROOT).as_posix() for folder in [packet,cross] for p in folder.rglob('*') if p.is_file()]
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/TranAD_SPOT_and_CrossAD16_manifest_20261016_v1.json'
assert not manifest.exists()
s.write(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths),
    raw_evidence_whitespace_exceptions=[(packet/'preserved_pre_registration_validation_failure.xml').relative_to(ROOT/'projects/flow_matching_research').as_posix()]))
print(json.dumps(dict(packet=str(packet),files=len(paths),SPOT_complete=len(complete),CrossAD_complete=16,manifest=str(manifest))))
