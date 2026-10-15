"""Archive actual full data materialization and independent stored-data closure."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from scripts.flow_matching import prepare_moment_icm_classification_v1 as producer
from scripts.flow_matching.register_moment_icm_classification_preprocessing_v1 import CONFIG

ROOT,read,sha,write=producer.ROOT,producer.read,producer.sha,producer.write
config=read(ROOT/CONFIG);jobs=producer.verify(config)
base=ROOT/config['state_root'];status=read(base/'status.json')
assert status['completed_count']==117 and not status['failed_preserved'] and status['active_pid'] is None
assert status['state']=='all117_preprocessing_attempts_terminal_original_model_experiments_remain'
closure=Path(config['artifact_root'])/'full117_processed_data_closure_v1.json'
proof=read(closure);guard=read(base/'closure_verification_v1/guard_receipt.json')
assert proof['passed'] and proof['datasets']==117 and proof['official_splits']==234
assert proof['configuration_sha256']==sha(ROOT/CONFIG) and guard['configuration_sha256']==sha(ROOT/CONFIG)
assert proof['verifier_sha256']==sha(ROOT/'scripts/flow_matching/verify_moment_icm_full117_preprocessing_v1.py')
assert guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['proof_sha256']==sha(closure)
assert proof['total_source_instances']==182780 and proof['total_transformed_signal_coordinates']==267291904
assert proof['new_model_fit_inference_SVM']==0 and not proof['strict_train_only_preprocessing']
assert len(proof['records'])==117 and {r['dataset'] for r in proof['records']}=={j['dataset'] for j in jobs}
originals={};maximum_private=0;maximum_rss=0;fit_count=0
for job,row in zip(jobs,proof['records']):
    assert job['dataset']==row['dataset']
    source=Path(config['artifact_root'])/job['dataset']/'preprocessing.json'
    receipt=base/'attempts'/job['dataset']/'guard_receipt.json'
    assert sha(source)==row['proof_sha256'] and sha(receipt)==row['guard_sha256']
    record,g=read(source),read(receipt)
    assert g['passed'] and g['exit_code']==0 and not g['resource_guard_aborted'] and g['proof_sha256']==sha(source)
    maximum_private=max(maximum_private,g['peak_tree_private_bytes']);maximum_rss=max(maximum_rss,g['peak_tree_rss_bytes'])
    fit_count+=int(record['original_data_standardizer_fitting_performed'])
    # Preserve exact original bytes inside UTF8 strings;read_bytes avoids CRLF
    # normalization. Decoding and encoding these strings recovers original files.
    originals[job['dataset']]=dict(source_sha256=sha(source),guard_sha256=sha(receipt),
        original_source_utf8=source.read_bytes().decode('utf-8'),original_guard_utf8=receipt.read_bytes().decode('utf-8'))
    assert __import__('hashlib').sha256(originals[job['dataset']]['original_source_utf8'].encode('utf-8')).hexdigest()==sha(source)
    assert __import__('hashlib').sha256(originals[job['dataset']]['original_guard_utf8'].encode('utf-8')).hexdigest()==sha(receipt)
folder=ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_ICM_full117_preprocessed_v1'
folder.mkdir(parents=True,exist_ok=False)
checks=0
for name,path,count in [('preprocessing_validation.xml','tmp/MOMENT_ICM_classification_preprocessing_validation_20261015_v4.xml',10),
                        ('stored_data_closure_validation.xml','tmp/MOMENT_ICM_preprocessing_closure_validation_20261015_v1.xml',6)]:
    suites=list(ET.parse(ROOT/path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count
    assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    checks+=count;shutil.copyfile(ROOT/path,folder/name)
for name,path in {
    'frozen_full117_configuration.json':ROOT/CONFIG,
    'full117_preprocessing_source.py':ROOT/'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
    'registration_source.py':ROOT/'scripts/flow_matching/register_moment_icm_classification_preprocessing_v1.py',
    'independent_stored_data_verifier_source.py':ROOT/'scripts/flow_matching/verify_moment_icm_full117_preprocessing_v1.py',
    'preprocessing_validation_tests.py':ROOT/'tests/test_fm_moment_icm_classification_preprocessing_v1.py',
    'closure_validation_tests.py':ROOT/'tests/test_fm_moment_icm_preprocessing_closure_v1.py',
    'actual_original_controller_launch.json':base/'launch.json',
    'actual_terminal_full117_status.json':base/'status.json',
    'full117_processed_data_closure.json':closure,
    'actual_full_closure_guard.json':base/'closure_verification_v1/guard_receipt.json',
    'actual_full_closure_launch.json':base/'closure_verification_v1/launch.json',
}.items():shutil.copyfile(path,folder/name)
write(folder/'all117_original_source_and_guard_receipts.json',dict(
    original_UTF8_bytes_preserved_in_strings=True,records=originals,
    extraction='Encode each original_source_utf8/original_guard_utf8 string asUTF8 to recover exact original proof/guard bytes,includingCRLF;check the associatedSHA256.'))
unit_history=[]
for name,reason in [('tmp/MOMENT_ICM_classification_preprocessing_validation_20261015_v1.xml',
    'Pre-registration privateAST environment missedos;fixed before any actual data launch.'),
    ('tmp/MOMENT_ICM_classification_preprocessing_validation_20261015_v3.xml',
    'Expanded ragged fixture omitted@equalLengthfalse;author parser correctly rejected malformed fixture;fixture header fixed before actual data launch.')]:
    path=ROOT/name;suites=list(ET.parse(path).getroot().iter('testsuite'))
    unit_history.append(dict(path=name,sha256=sha(path),failures=sum(int(s.get('failures',0)) for s in suites),
        reason=reason,actual_dataset_or_neural_experiment_failed=False))
write(folder/'preserved_pre_registration_fixture_diagnoses.json',dict(records=unit_history,
    final_preprocessing_checks=10,final_closure_checks=6,frozen_numerical_sources_or_budgets_relaxed=False))
write(folder/'complete_materialization_summary.json',dict(
    datasets=117,official_splits=234,source_instances=182780,transformed_signal_coordinates=267291904,
    complete_saved_NPY_artifacts_rehashed=936,related_source_standardizer_fitted_dataset_count=fit_count,
    related_source_normalization_excluded_dataset_count=117-fit_count,
    peak_preprocessing_worker_private_bytes=maximum_private,peak_preprocessing_worker_rss_bytes=maximum_rss,
    closure_peak_private_bytes=guard['peak_tree_private_bytes'],closure_peak_rss_bytes=guard['peak_tree_rss_bytes'],
    data_location_from_configuration=config['artifact_root'],raw_files_overwritten=False,
    original_data_standardizer_actual_fit_flag_is_authoritative=True,
    TEST_fit_preserved_boolean_denotes_source_policy_including_its_normalization_exclusions=True,
    new_neural_training_inference_SVM=0,strict_train_only_preprocessing=False,
    independently_reconstructed_scaler_and_resampling_values=False,exact_historical_ICM_pipeline_certified=False,
    original_paper_pretraining_forecast_and_SVM_results_complete=False))
(folder/'README.md').write_text(
    '# 117个分类数据集的完整来源对照预处理已完成\n\n'
    '原文表6的26个多变量、表7的91个单变量数据集均已实际处理，117/117完成、0失败，共234个官方TRAIN/TEST划分、'
    '182,780个原始实例、267,291,904个变换后信号坐标。936份NPY产物保存在配置指定的F盘目录，'
    '原始TS数据没有覆盖、重排或删样本，也没有因通道数较多而截断通道。\n\n'
    '在精确登记的Python3.10.21、NumPy1.25.2、pandas2.2.2、SciPy1.9.3、scikit-learn1.5.2环境，'
    '通过私有AST副本执行固定MOMENT来源的TS解析、分类数据类、归一化排除清单、13个图像/频谱数据插值例外及插值/填充工具，'
    '作者源文件及正在训练的模型没有修改。输入长度256来自ICM论文；完整官方TRAIN对应来源脚本把train与val表示拼回后的全部训练样本。'
    '这一合并的逐坐标等价性已测试，包括原拆分边界落在同一多通道实例内部的情况。\n\n'
    '输出values保持来源原生dtype及全部实例/通道，另存共享的原始padding mask、训练类编码的标签和完整原始实例行号。'
    '必须注意：这里保留了相关MOMENT来源代码的归一化规则，部分数据集在TEST文件自身拟合StandardScaler；'
    '来源排除清单中的数据集保持不拟合。实际是否拟合以original_data_standardizer_fitting_performed及scaler shape为准，'
    'TEST_fit_preserved标志表示来源规则被保留，包含其排除例外。它是明确登记的来源对照，不能作为严格训练集专属归一化版本。\n\n'
    '117项真实处理各有独立子进程、退出码、峰值内存及保护回执，均通过。新的非神经网络数据处理保护为16GiB可用内存、'
    '21GiB提交余量准入、16GiB紧急线和3GiB进程树RSS/private上限，21=16+3+2GiB余量。'
    '已有NODE/BRITS/CrossAD及其他模型的任何准入门槛、预算、网格、精度或容差均未修改。'
    '这是单独的数据物化任务，所有数据均全量执行，没有为适配资源上限减少样本或通道。\n\n'
    '随后新的受保护复核进程重算936份保存数组的SHA256，重新检查完整shape、dtype、每个信号坐标有限、全部mask二值、'
    '完整实例行序，并从原始TS独立读取全部标签和通道数，以官方TRAIN类集合重建编码，逐行比对TRAIN及TEST标签。'
    '该复核也核对了原始文件哈希及全部来源/保护回执，已通过。它没有独立重写或重算来源的插值及标准化数值，'
    '因此不能据此提高到历史ICM代码、预处理或权重等价认证。全部117份原始证明及保护回执在JSON字符串中按原字节封存，'
    '可按附带SHA256还原检查。\n\n'
    '16项最终检查通过：10项覆盖完整坐标导出、通道/标签/掩码一致性拒绝、变长及缺失值、长序列、插值例外和精确来源解释器；'
    '6项覆盖独立Unicode标签编码、通道/行序/标签/非有限值/已更改字节拒绝及证据边界。'
    '初期隔离导入遗漏及不合法fixture头的失败记录另行保留，均发生于冻结及实际处理之前，不是实际数据集或模型失败。\n\n'
    '本轮完成的是数据准备及其存储/标签复核，新增神经网络训练、模型推理、SVM和模型性能结果均为0。'
    '原始ICM/Tiny的一epoch预训练、全部上下文扩展及预测、117数据集SVM、beta消融、严格归一化对照和未公开设置来源对齐仍需完成。'
    '相关MOMENT代码提交不等于ICM作者整条实验管线；已有72+6异常检测复核也保持分开。'
    '原45篇/681方法记录/273数据任务的最低范围和新增原文任务继续保留，全部研究实验尚未完成。\n',encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
shutil.copyfile(__file__,folder/'publication_source.py')
write(folder/'archive_validation.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),tests_passed=16,
    real_datasets_preprocessed=117,official_splits=234,independently_reconciled_instances=182780,
    transformed_signal_coordinates=267291904,source_and_guard_receipts=234,new_model_results=0,
    files_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},all_original_experiments_complete=False))
ara=ROOT/'projects/flow_matching_research/ara/execution/moment_ICM_full117_source_preprocessing.2026-10-15_v1.json'
write(ara,dict(paper_id='moment_long_context',configuration=CONFIG,evidence=folder.relative_to(ROOT).as_posix(),
    source_control_preprocessing_complete=True,datasets=117,official_splits=234,source_instances=182780,
    full_saved_bytes_labels_channel_and_row_audit_passed=True,strict_train_only_preprocessing=False,
    historical_ICM_pipeline_certified=False,new_model_results=0,
    remaining='Original pretraining,forecasting,context-expansion baselines,SVM,beta ablations and strict controls remain.',
    all_original_experiments_complete=False))
targets=[folder.relative_to(ROOT).as_posix(),ara.relative_to(ROOT).as_posix(),CONFIG,
    'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
    'scripts/flow_matching/register_moment_icm_classification_preprocessing_v1.py',
    'scripts/flow_matching/verify_moment_icm_full117_preprocessing_v1.py',
    'tests/test_fm_moment_icm_classification_preprocessing_v1.py',
    'tests/test_fm_moment_icm_preprocessing_closure_v1.py']
paths=set()
for arguments in [('diff','--name-only'),('ls-files','--others','--exclude-standard')]:
    paths.update(subprocess.check_output(['git',*arguments,'--',*targets],cwd=ROOT,text=True).splitlines())
git=lambda cwd,*a:subprocess.check_output(['git','-C',str(cwd),*a],text=True).strip()
assert paths and not git(ROOT,'diff','--cached','--name-only')
write(ROOT/'tmp/MOMENT_ICM_full117_preprocessing_publication_manifest_20261015_v1.json',dict(
    previous_root=git(ROOT,'rev-parse','HEAD'),previous_parent=git(ROOT.parent/'flow-matching-research','rev-parse','HEAD'),paths=sorted(paths)))
print(json.dumps(dict(exact_files=len(paths),datasets=117,official_splits=234,tests_passed=16,new_model_results=0)))
