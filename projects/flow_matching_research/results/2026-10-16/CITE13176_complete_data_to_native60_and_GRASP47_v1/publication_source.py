"""Publish concrete full-data/future-native bindings and a newly closed GRASP model."""
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching import prepare_CITE13176_GEO_raw_backed_QC_v2 as prep
from scripts.flow_matching import run_otcfm_CITE13176_GEO_native60_v1 as native

packet=ROOT/'projects/flow_matching_research/results/2026-10-16/CITE13176_complete_data_to_native60_and_GRASP47_v1'
data=read(ROOT/prep.CONFIG);prep.verify(data)
plan=native.read_plan()
assert len(plan['planned_cases'])==plan['required_models']==60 and plan['required_preflight_combinations']==12
ids=[j['id'] for j in plan['planned_cases']]
for path in ['configs/experiments/fm_otcfm_singlecell_native_full120.v4.json',
    'configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json','configs/experiments/fm_otcfm_EB100_code_referenced_native_full90.v1.json']:
    ids += [j['id'] for j in read(ROOT/path)['jobs']]
assert len(ids)==len(set(ids))==420
old=read(ROOT/prep.previous.CONFIG);old_base=ROOT/old['state_root']
assert not Path(old['output_directory']).exists() and not (old_base/'worker.stdout.log').exists()
assert "KeyError: 'enough'" in (old_base/'controller.stderr.log').read_text(encoding='utf-8')
tests=[(Path('F:/aicoding/IIA_Data/experiments/CITE13176_GEO_deferred_full60_source_native_and_data_v1_tests.xml'),19),
    (Path('F:/aicoding/IIA_Data/experiments/GRASP_Tau8_seed1106_full47_closure_v1_tests.xml'),89)]
for path,count in tests:
    suites=list(ET.parse(path).getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==count
    assert not any(int(x.get(k,0)) for x in suites for k in ('failures','errors','skipped'))
grasp=ROOT/'projects/flow_matching_research/results/2026-10-16/grasp_tau8_seed1106_full47_model_increment_v1'
closure=read(grasp/'actual_full47_model_increment_closure.json')
assert closure['actual_current_unique_complete_models']==47 and len(set(closure['current_complete_original_ids']))==47
assert closure['canonical_id']=='grasp_protocol_v3__tau_8__swan__swan_sf__seed1106'
assert closure['complete_original_training_and_full_native_replay_and_independent_recipe_review']
assert closure['epochs']==1500 and closure['optimizer_updates']==93000 and closure['parameter_count']==69244
assert closure['complete_Tau8_Swan_seeds']==[1103,1104,1105,1106] and not closure['full_ten_seed_Tau8_cohort_complete']
for label,digest in closure['current_source_proof_hashes'].items():assert sha(grasp/(label+'.json'))==digest
packet.mkdir(parents=True,exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.py -text\n*.xml -text\n*.txt -text -whitespace\n',encoding='utf-8',newline='\n')
for label,path in [('preserved_failed_controller_trace',old_base/'controller.stderr.log'),
    ('preserved_failed_controller_launch',old_base/'launch.json'),('preserved_first_configuration',ROOT/prep.previous.CONFIG),
    ('corrected_preprocessing_launch',ROOT/data['state_root']/'launch.json'),
    ('deferred_native60_launch',ROOT/plan['state_root']/'launch.json')]:
    shutil.copyfile(path,packet/(label+('.txt' if path.suffix=='.log' else '.json')))
live={}
for label,state_root in [('full_CITE_preprocessing',data['state_root']),('full_native60_after_data',plan['state_root'])]:
    s=read(ROOT/state_root/'status.json');p=psutil.Process(s['pid']);assert p.create_time()==s['create_time'] and p.is_running()
    live[label]=dict(status=s,exact_process_command=p.cmdline(),children=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline(),cpu_seconds=sum(c.cpu_times()[:2]))
        for c in p.children(recursive=True) if c.name().lower()=='python.exe'])
write(packet/'actual_live_preparation_and_deferred_native_handles.json',live)
write(packet/'iteration_summary.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    full_QC_cells=29394,all_original_raw_RNA_features=58676,original_raw_cells_preserved=35261,
    new_data_pipeline_completed=False,preprocessing_state=live['full_CITE_preprocessing']['status']['state'],
    new_full_native_model_conditions=60,prior360_conditions_preserved=True,total_distinct_singlecell_conditions_defined=420,
    native60_numerical_work_started=False,native60_queue_materialized=(ROOT/native.QUEUE).exists(),
    all12real_data_preflight_conditions_still_required=True,new_model_replay_numerical_work_completed=False,
    metadata_failure_before_any_numeric_work_preserved=True,full_worker_bytecode_unchanged=True,
    actual_GRASP_complete_original_models=47,GRASP_total_required_original_models=9120,
    original_scope=data['original_scope'],all_original_experiments_complete=False))
write(packet/'full_data_preparation_policy_and_failure_boundary.json',dict(
    full_gene_axis='all58676actual rawRNA genes,never the17959intersection or fabricated zeros for per-day filtered-out genes',
    full_published_QC_roster=data['expected_QC_cells'],published_raw_count_audit='every coordinate including implicit zeros at exact cell/gene identities',
    independent_normalization_library=data['independent_normalization_library'],normalization=data['normalization'],
    normalization_candidate_selection_changes_preparation=False,candidate_tolerance=data['candidate_audit_tolerance'],
    pca_dimensions=5,pca_initial_vector_seed=0,author_whiten_once=True,all_timepoints_in_preprocessing_fit=True,
    unchanged_resource_policy=data['settings'],stable_samples=data['stable_samples'],original_failure=data['metadata_recovery'],
    original_author_or_Kaggle_matrix_identity_certified=False))
for index,(path,count) in enumerate(tests):shutil.copyfile(path,packet/f'validation_{count}_checks.xml')
paths=[prep.previous.CONFIG,prep.CONFIG,native.PLAN,
    'scripts/flow_matching/prepare_CITE13176_GEO_raw_backed_QC_v1.py',
    'scripts/flow_matching/prepare_CITE13176_GEO_raw_backed_QC_v2.py',
    'tests/test_fm_CITE13176_GEO_raw_backed_QC_v1.py','tests/test_fm_CITE13176_GEO_raw_backed_QC_v2.py',
    native.MODULE.replace('.','/')+'.py',native.TEST,
    'scripts/flow_matching/audit_otcfm_CITE13176_GEO_full60_replay_v1.py','tests/test_fm_otcfm_CITE13176_GEO_full60_replay_v1.py',
    *[f'configs/models/fm_otcfm_CITE13176_GEO_native__{m}.v1.json' for m in native.contract.METHODS]]
for index,path in enumerate(paths):shutil.copyfile(ROOT/path,packet/(f'source{index:02d}'+Path(path).suffix))
shutil.copyfile(__file__,packet/'publication_source.py')
write(packet/'source_manifest.json',dict(files=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
    preprocessing_source_receipts=data['source_receipts'],deferred_native_source_receipts=plan['source_receipts']))
(packet/'README.md').write_text('''CITE13176完整数据到60项原生实验自动接续，以及GRASP第47项闭环

新GEO输入包含四天共29394个发布者QC细胞，原始10x文件35261个细胞均保留。处理版每一天基因过滤不同；不能仅取17959共同基因，也不能将没有提供的基因补成假的零表达。新独立轨道保留全部58676个真实原始RNA基因，按精确细胞/基因ID核对每个发布者raw.X计数坐标，包括全部隐式零；从实际原始计数重建完整QC群体。三套固定归一化公式仅用于诊断发布者转换，不改变预先声明的全RNA库大小CP1M/log1p输入。采用既有完整中心化ARPACK PCA5、初值种子0和一次原作者白化；原NPZ加载器及每个细胞/时间点/PC都须实际核验。全部时间点参与PCA/白化，与作者路径一致，不认证严格归纳性或历史Kaggle预处理身份。

第一次新控制器在读取资源API时遇到KeyError:enough，尚未启动数值子任务或生成数据产物。原配置、代码和stderr全部保留。独立V2仅将私有控制器AST的一个API键改为实际has_headroom，原完整数学worker字节码不变。真实API门槛检查和AST逆变换检查通过；原32GiB提交余量、16GiB物理可用/紧急线、3GiB RSS/6GiB private、六次稳定采样与共享锁均保持。新控制器已实际启动，但采集时仍等待原资源条件，不能称已完成数值预处理。

后续60项条件已完整定义并由实际存活的接续控制器等待：CFM/OT-CFM/SB-CFM，两套完整停止配置，留出时间点1/2，种子42–46。数据原守卫退出0、完整逐坐标/PCA/加载器证明通过后，才写入含真实输入哈希的训练队列，并自动启动完整检查点复核。训练前需12个真实数据原生预检，全部源worker字节码、9093参数、原配对/噪声/AdamW、1000epoch上限/停止规则、CPU2及所有原门限不变。每例后续须从选择权重和优化器计数复核、原生重推理全部36指标、两组完整细胞代价矩阵和独立W1 OPTIMAL/扰动界检查。登记或等待不算训练/复核完成。现有360项保留，合计定义420个不同单细胞条件；其余原方法/作者队列也保持。

GRASP Tau8/Swan/swan_sf/seed1106完成原1500epoch、93000更新、69244参数及原5源样本×10流评估配方；3935验证点、72000测试点的检查点重推理、全部前向预算和独立完整数组/指标复核通过。验证99%阈值且无点调整：precision0.4024185239、recall0.7384079602、F1=0.5209364361；测试标签优化的回顾性Best_F1=0.5216053043、AP=0.5549611464、ROC=0.7145137607，另列而不替代验证指标。当前新增闭环为第47/9120项，Tau8/Swan仅4/10种子，没有完整十种子CI，也不作作者等价或严格TAB认证。冻结模板中Tau1seed1104的错误说明保留，结构化profile/canonical/seed/权重及独立证明明确本模型为Tau8seed1106。

19项新数据/60项接续/原生复核合同检查及89项GRASP流程/复核检查通过，所有实际数值与失败证据保留。45论文/681方法基线消融/273数据任务/702论文标量及其他新实验的完整目标仍未完成。
''',encoding='utf-8',newline='\n')
paths += [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()]
paths += [p.relative_to(ROOT).as_posix() for p in grasp.rglob('*') if p.is_file()]
write(ROOT/'tmp/CITE_data_to_native60_GRASP47_publication_manifest_20261016_v1.json',dict(
    previous_root=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    previous_parent=subprocess.check_output(['git','-C',str(ROOT.parent/'flow-matching-research'),'rev-parse','HEAD'],text=True).strip(),paths=sorted(set(paths))))
print(json.dumps(dict(exact_files=len(set(paths)),new_native_cases_defined=60,prior_singlecell_cases_preserved=360,actual_GRASP_models_closed=47,
    preprocessing_live_wait=True,new60_actual_training_complete=False),ensure_ascii=False))
