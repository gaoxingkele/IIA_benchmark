from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_cfm_node1106_epoch280_resume_v1 as cfm
from scripts.flow_matching import evaluate_tranad_original_SPOT_full30_v1 as spot
from scripts.flow_matching import audit_tranad_original_SPOT_completed_cases_v1 as audit

ROOT=cfm.ROOT;paths=[]
def check_xml(path,required):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in suites)==required
    assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in suites)
def close(packet):
    shutil.copyfile(__file__,packet/'packaging_source.py')
    cfm.exclusive(packet/'packet_integrity.json',dict(files=[dict(path=p.relative_to(packet).as_posix(),sha256=cfm.sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
    paths.extend(p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file())

c=cfm.read(ROOT/cfm.CONFIG);cfm.verify(c)
cfm.replay_namespace()['verify'](cfm.read(ROOT/cfm.REPLAY))
base=ROOT/cfm.BASE;launch=cfm.read(base/'launch.json');handles={}
for role,h in launch['handles'].items():
    p=psutil.Process(h['pid']);assert p.is_running() and p.create_time()==h['create_time']
    assert p.cmdline()==h['command']
    handles[role]=dict(h,actual_command=p.cmdline())
source_state=cfm.read(base/'status.json');replay_state=cfm.read(base/'replay/status.json')
assert not source_state.get('resource_guard_aborted') and not replay_state['failed_preserved']
xml=Path('F:/aicoding/IIA_Data/experiments/CFM_NODE1106_epoch280_metadata_20261016_v1_tests.xml');check_xml(xml,17)
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CFM_NODE1106_terminal284_full_epoch280_resume_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
for source,name in [(ROOT/c['previous_terminal'],'preserved_terminal_resource_abort.json'),
    (ROOT/c['previous_progress'],'preserved_partial284_progress.json'),(Path(c['previous_complete_epoch_ledger']),'all28_complete_epoch_save_receipts.json'),
    (ROOT/cfm.PREVIOUS,'previous_frozen_runtime_configuration.json'),(ROOT/cfm.CONFIG,'frozen_resume_configuration.json'),
    (ROOT/cfm.REPLAY,'all285_native_independent_routes.json'),(base/'launch.json','actual_launch.json'),
    (base/'status.json','captured_source_controller_state.json'),(base/'replay/status.json','captured_replay_controller_state.json'),
    (xml,'relevant17_metadata_and_source_checks.xml')]:shutil.copyfile(source,packet/name)
ledger=cfm.read(c['previous_complete_epoch_ledger']);latest=ledger[-1]
assert latest['completed_original_epochs']==280 and cfm.sha(latest['full_state_path'])==latest['full_state_sha256']
cfm.exclusive(packet/'iteration_status.json',dict(original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
    original_canonical_CFM_jobs=285,actual_failed_partial_epochs=284,saved_complete_original_boundary=280,
    discarded_unsaved_epochs=[281,282,283,284],remaining_original_epochs=120,required_original_epochs=400,
    complete_native_state_sha256=latest['full_state_sha256'],new_recovery_source_complete=cfm.source_closed(c),
    new_recovery_native_certificate_complete=False,actual_live_handles=handles,
    inherited_actual_primary_verified_models=len(replay_state['verified_original_ids']),
    inherited_models_are_not_new_recovery_completions=True,old_worker_and_parent_missing_or_different_identity=True,
    all_original_experiments_complete=False))
(packet/'README.md').write_text('''# CFM NODE1106：保留284轮资源中止，从完整280轮状态恢复

旧400轮任务在完成284轮更新后触发原16GiB紧急提交内存守卫；实际最低可用提交内存16897204224字节，退出1，原失败记录保留。不是数值完成，也不把284轮算作完整400轮。

28份完整10轮间隔检查点及保存回读收据全部核对哈希。最新完整状态为280轮，含模型、Adam优化器、参数梯度、CPU/NumPy/Python随机状态、外层随机分支、损失历史、求解器计数、预处理及训练数据顺序。新路径直接复用冻结原生恢复实现：逐位校验恢复后继续281至400轮，舍弃未保存281至284轮，不混合两次迭代的优化步数。

原32GiB准入、16GiB紧急提交内存、CPU单线程训练、batch258、dopri5及原容差和完整500点测试保持；新输出目录只允许一次显式恢复，原失败、所有检查点与原始数据均不覆盖。17项新注册、错误边界拒绝、全285范围和冻结源码检查通过。原生完整网络恢复实现未改，生产恢复仍需实际准入后执行，并以恢复证明及完整400轮结果验收；元数据检查不冒充已恢复训练。

源控制器与全285原生重放/独立复核控制器已启动；只对这个新执行调度重放，其余284位置和已有主轨实际证书保留，继承证书不算本次新完成。新源守卫、实际280轮恢复证明、完整400更新和500点输出齐备后才允许原生重放及独立复核。等待内存或上游不是完成。

这是既有论文派生重建模型的受控恢复，不是原作者等价性认证。45篇、681方法、273数据任务、702原指标及全部原论文/TAB/新数据范围继续保留，整体目标未完成。
''',encoding='utf-8');close(packet)
ara=ROOT/'projects/flow_matching_research/ara/execution/CFM_NODE1106_epoch280_resume.2026-10-16_v1.json'
cfm.exclusive(ara,dict(packet=packet.relative_to(ROOT).as_posix(),required_original_epochs=400,actual_saved_boundary=280,
    source_complete=False,native_complete=False,all_original_experiments_complete=False))
paths.extend([ara.relative_to(ROOT).as_posix(),cfm.CONFIG,cfm.QUEUE,cfm.REPLAY,cfm.TEST,cfm.MODULE.replace('.','/')+'.py'])

c=spot.read(ROOT/spot.CONFIG);spot.verify(c)
src=Path('F:/aicoding/IIA_Data/experiments/TranAD_SPOT_first3_independent_audit_20261016_v1/independent_full_array_audit.json')
full=spot.read(src);assert full['complete_cases']==3 and len(full['cases'])==3
packet=ROOT/'projects/flow_matching_research/results/2026-10-16/TranAD_SPOT_first3_complete_full54_independent_v1';packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n',encoding='utf-8')
shutil.copyfile(src,packet/'independent_full_array_audit.json');summary=[]
for actual in full['cases']:
    record=next(r for r in c['entries'] if r['entry']['case']==actual['case'])
    assert audit.audit_case(c,record)==actual
    identity=actual['case']['id'];source=Path(c['artifact_root'])/identity/spot.REPORT
    guard=ROOT/c['state_root']/'attempts'/identity/'guard_receipt.json'
    shutil.copyfile(source,packet/(identity+'.source.json'));shutil.copyfile(guard,packet/(identity+'.guard.json'))
    summary.extend(dict(case_id=identity,**row) for row in actual['full_denominator_macro_controls'])
with (packet/'all3_cases_24_macro_controls.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
xml=Path('F:/aicoding/IIA_Data/experiments/TranAD_SPOT_independent_audit_20261016_v1_tests.xml');check_xml(xml,4)
shutil.copyfile(xml,packet/'relevant4_independent_metric_checks.xml')
spot.exclusive(packet/'iteration_status.json',dict(required_unique_cases=25,comparison_slots=30,complete_source_SPOT_cases=3,
    original_raw_model_cases=2,matched105_model_cases=1,full_entities_per_case=54,full_test_points_per_case=435826,
    independent_point_metric_cells=sum(x['independent_point_metric_cells'] for x in full['cases']),
    undefined_constant_calibration_entity_instances=2,undefined_full_macro_controls=16,defined_full_macro_controls=8,
    original_model_training_or_inference_performed=False,all_original_experiments_complete=False))
(packet/'README.md').write_text('''# 原作者SPOT：首3个完整54实体案例的独立复核

完成原始SMAP种子1103、1104和matched105 SMAP种子1103，每个54实体、435826测试点；全部4标签视图、原始/原作者PA两种预测均核对，独立标准库重新读取NPY字节、计算阈值预测、事件调整、混淆计数和原epsilon指标，10240个点指标单元一致。4项独立计算检查通过。没有重训模型或重新拟合POT。

原始数据轨两案例的SMAP_D-12训练分数恒定，各7918测试点无法定义原SPOT阈值。保持完整54实体分母，相关16个完整宏观F1留空；53实体有定义子集均值明确另列，不补零、不回退百分位，也不声称是全数据集成绩。matched105案例54实体全部可拟合，8个宏观控制有定义。

主标签source_first_exclusive：matched105种子1103的原始宏观F1为0.091164434880，原作者PA后为0.217451363757；这是一个固定本地FP64模型案例，不是完整五种子结论，也不是作者模型架构或严格TAB认证。原始轨种子1104只在53个有定义实体上的宏观F1为0.073874893518，PA后0.182087109591，不能与完整54实体成绩混作同一分母。

该对照直接采用冻结作者Grimshaw SPOT，排除了旧近似工具两处公式错误，但低分仍存在，故不能把现有低分全部归因于该工具。完整作者架构、预处理、训练/评估协议和数据分布仍需各自实际实验。原作者二值预测ROC/AUC并非连续分数AUROC；PA使用测试标签，必须与验证集校准、无PA的严格评价单列。

队列保留25个独立案例和30个比较位置，其余等待实际上游原生模型产物；当前3个案例不代表整个方向完成。45/681/273/702完整范围与所有既有任务继续保留。
''',encoding='utf-8');close(packet)
ara=ROOT/'projects/flow_matching_research/ara/execution/TranAD_SPOT_first3_full54_independent.2026-10-16_v1.json'
spot.exclusive(ara,dict(packet=packet.relative_to(ROOT).as_posix(),source_cases_complete=3,source_cases_required=25,
    full_entities_per_case=54,independent_point_metric_cells=10240,all_original_experiments_complete=False))
paths.extend([ara.relative_to(ROOT).as_posix(),'scripts/flow_matching/audit_tranad_original_SPOT_completed_cases_v1.py',
    'tests/test_fm_tranad_original_SPOT_independent_audit_v1.py','projects/flow_matching_research/README.rst'])
head=lambda cwd:subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD']).decode().strip()
manifest=ROOT/'tmp/CFM_epoch280_and_SPOT3_manifest_20261016_v1.json'
cfm.exclusive(manifest,dict(previous_root=head(ROOT),previous_parent=head(ROOT.parent/'flow-matching-research'),paths=sorted(paths)))
print(json.dumps(dict(files=len(paths),new_CFM_recovery_source_complete=False,SPOT_source_cases_complete=3,independent_metric_cells=10240,manifest=str(manifest))))
