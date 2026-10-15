"""Archive the actual complete Base/PSM replay and a matched Small comparison."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as v

ROOT, read, write, sha = v.ROOT, v.read, v.write, v.sha
runtime_path = 'configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
runtime = read(ROOT/runtime_path)
cases = v.verify(runtime)
state = read(ROOT/runtime['state_root']/'status.json')
name = 'fm_moment_pretrained_base__release512__psm__seed42'
small = 'fm_moment_pretrained_small__release512__psm__seed42'
folder = ROOT/'projects/flow_matching_research/results/2026-10-15/MOMENT_base512_PSM42_full_native_closed_v1'
assert not folder.exists()
files, records, summaries = {}, [], []
for identity, parameters in [(small, 35341512), (name, 109641608)]:
    case = cases[identity]
    source, data, binding = v.check_source(runtime, case)
    directory = Path(runtime['artifact_root'])/identity
    native_path, review_path = directory/'native/native_replay.json', directory/'independent/review.json'
    native, review = read(native_path), read(review_path)
    assert identity in state['verified_ids']
    assert native['passed'] and review['passed']
    assert native['source_binding'] == review['source_binding'] == binding
    assert native['job'] == review['job'] == case['job']
    assert native['runtime_sha256'] == sha(ROOT/runtime_path)
    assert review['native_replay_sha256'] == sha(native_path)
    assert native['all_original_groups_reinferred'] and native['all_array_dtype_shape_content_identical']
    assert review['all_original_arrays_independently_checked']
    assert review['all_original_core_metrics_events_and_200_resample_intervals_recomputed']
    assert native['checkpoint_load_audit']['parameter_count'] == parameters
    assert native['checkpoint_load_audit']['all_keys_strictly_loaded']
    assert native['training_updates_performed'] == 0
    assert native['native_coverage'] == source['native_coverage']
    assert [(r['role'], r['native_points']) for r in native['native_coverage']] == [('train', 105884), ('validation', 26497), ('test', 87841)]
    assert native['inline_full_window_MSE_check']['forward_calls'] == 431
    assert sha(native['scores_path']) == native['scores_sha256'] == binding['scores_sha256']
    selected = dict(original_result=ROOT/case['job']['output_directory']/'result.json', native_replay=native_path,
                    independent_review=review_path, model_configuration=ROOT/case['job']['model_config'])
    for phase, proof in [('native', native_path), ('independent', review_path)]:
        base = ROOT/runtime['state_root']/'attempts'/identity/phase
        guard = read(base/'guard_receipt.json')
        assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
        assert guard['proof_sha256'] == sha(proof)
        assert guard['exact_child_handle']['command'] == [str(ROOT/case['queue']['python']), '-B', '-X', 'utf8', '-u', '-m', v.MODULE,
            '--config', runtime_path, '--stage', phase, '--job-id', identity]
        selected[phase+'_guard'] = base/'guard_receipt.json'
        selected[phase+'_launch'] = base/'launch.json'
    size = 'Base' if identity == name else 'Small'
    files[size] = selected
    controls = review['independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls']
    for ratio, original in source['metrics']['strict'].items():
        c = controls[ratio]
        assert c['micro']['f1'] == original['micro']['f1']
        assert c['threshold'] == original['calibration']['threshold']
        ci, events = original['f1_ci95'], original['event_metrics']
        records.append(dict(model='MOMENT-1-'+size.lower(), parameters=parameters, dataset='PSM', context=512, pretrained_seed=42,
            validation_quantile_exceedance_percent=float(ratio), threshold=c['threshold'], no_PA_F1=c['micro']['f1'],
            same_threshold_PA_F1=c['same_validation_threshold_PA_diagnostic']['f1'], AUROC=c['micro']['auroc'],
            AP=c['micro']['average_precision'], precision=c['micro']['precision'], recall=c['micro']['recall'],
            F1_CI95_low=ci['low'], F1_CI95_high=ci['high'], bootstrap_unit=ci['unit'], bootstrap_resamples=ci['resamples'],
            bootstrap_seed=ci['seed'], bootstrap_block_points=ci['block_points'], event_recall=events['event_recall'],
            events=events['events'], detected_events=events['detected_events'],
            false_alarms_per1000_normal=events['false_alarms_per_1000_normal_points']))
    summaries.append(dict(job_id=identity, parameters=parameters, source_binding=binding,
                          role_points=dict(train=105884, validation=26497, test=87841), forward_calls=431,
                          checkpoint_load_audit=native['checkpoint_load_audit'], new_training_seed=False))
assert summaries[0]['source_binding']['input_sha256'] == summaries[1]['source_binding']['input_sha256']
validation = ROOT/'tmp/MOMENT_base512_PSM42_native_closure_validation_20261015_v1.xml'
suites = list(ET.parse(validation).getroot().iter('testsuite'))
checks = sum(int(s.get('tests', 0)) for s in suites)
assert checks >= 15 and all(int(s.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors', 'skipped'])
folder.mkdir(parents=True, exist_ok=False)
for size, selected in files.items():
    target = folder/size.lower()
    target.mkdir()
    for label, path in selected.items():
        shutil.copyfile(path, target/(label+'.json'))
        assert sha(path) == sha(target/(label+'.json'))
shutil.copyfile(ROOT/runtime_path, folder/'frozen_full78_replay_configuration.json')
shutil.copyfile(validation, folder/'validation.xml')
write(folder/'collector_status_snapshot.json', state)
write(folder/'metrics_and_matched_comparison.json', dict(records=records, full_native_models=summaries,
    additional_closed_cases=1, reused_small_case=small, new_training_seeds=0,
    all_validation_quantiles_preserved=True, bootstrap_is_model_seed_uncertainty=False,
    full_TAB_harness_certified=False, historical_paper_equivalence_certified=False,
    independent_lowlevel_VUS_affiliation_integrals=False, independent_full_preprocessing_reconstruction=False,
    original_missing_guard_history_retroactively_certified=False, all_original_experiments_complete=False))
buffer = io.StringIO(newline='')
writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator='\n')
writer.writeheader(); writer.writerows(records)
(folder/'all_validation_quantile_metrics.csv').write_text(buffer.getvalue(), encoding='utf-8', newline='')
primary = [r for r in records if r['validation_quantile_exceedance_percent'] == 1.]
lines = ['# MOMENT Base / PSM / context512：完整原生复核与同数据 Small 比较', '',
    '新增 Base 完整109,641,608参数官方冻结权重的原生重推理与第二独立进程验证；Small为此前已完成的35,341,512参数对照，不重复计数。两者使用同一输入文件哈希、105884训练/26497验证/87841测试点、context512、batch1、seed42、CPU两线程和冻结补齐规则。每例431次全窗口forward，所有原分数、标签等数组dtype、shape与字节相等。两阶段均实际成功退出且未资源中止；独立核对完整数组、核心指标、事件、200次区间及全部pooled阈值诊断。', '',
    '|预训练模型|固定验证1%未做PA F1|同阈值PA F1|AUROC|AP|', '|---|---:|---:|---:|---:|']
for r in primary:
    lines.append(f"|{r['model']}|{r['no_PA_F1']:.8f}|{r['same_threshold_PA_F1']:.8f}|{r['AUROC']:.8f}|{r['AP']:.8f}|")
lines += ['', '两种规模在当前同一冻结PSM设置的AUROC均约0.555，增加预训练参数并未带来明显的排序区分能力改善。Base未做PA F1约0.03918，同分数同阈值PA后约0.89801；该差异来自评估协议。这里没有重训种子重复或模型差异显著性检验，不能仅凭单例将规模效果推广到其他数据集，也不能据此否定论文原设置。', '',
    '0.1%、1%、5%三个事先登记验证分位数全部保留，1%不按测试F1择优。200次seed10042、block1024循环时间块区间是固定模型和阈值下的条件不确定性，不能当训练种子区间。原窗口包含未来上下文，不认证在线因果延迟。历史原执行保护记录缺失、全部预处理值独立重建、VUS/Affiliation底层积分、完整TAB执行器和历史论文精确等价仍未认证。官方冻结零样本评估与Tiny/ICM全PILE训练是不同实验。', '',
    f'{checks}项现有冻结验证通过。本例新增1项完整权重/数据集案例，新增训练种子0。快照队列{state["verified_count"]}/78；其余原实验及至少45篇/681方法记录/273数据任务和新增任务继续保留，全部研究目标未完成。', '',
    '来源：[MOMENT论文](https://arxiv.org/abs/2402.03885)、[官方Base权重](https://huggingface.co/AutonLab/MOMENT-1-base)、[PSM原数据](https://github.com/eBay/RANSynCoders)。冻结配置和实际数据/权重哈希随包保留，大文件留在原磁盘位置。']
(folder/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.csv -text\n', encoding='utf-8')
shutil.copyfile(__file__, folder/'publication_source.py')
write(folder/'archive_validation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), new_complete_cases=1,
    reused_previous_complete_cases=1, tests_passed=checks, files_sha256={p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
    all_original_experiments_complete=False))
ara = ROOT/'projects/flow_matching_research/ara/execution/MOMENT_base512_PSM42_full_native_closed.2026-10-15_v1.json'
write(ara, dict(method='MOMENT-1-base', dataset='PSM', context=512, official_pretrained_seed=42,
    evidence=folder.relative_to(ROOT).as_posix(), full_checkpoint_replay_and_independent_review=True, additional_closed_cases=1,
    original_scope_required_replay_cases=78, new_training_seeds=0, all_original_experiments_complete=False))
targets = [folder.relative_to(ROOT).as_posix(), ara.relative_to(ROOT).as_posix()]
paths = set()
for args in [('diff', '--name-only'), ('ls-files', '--others', '--exclude-standard')]:
    paths.update(subprocess.check_output(['git', *args, '--', *targets], cwd=ROOT, text=True).splitlines())
git = lambda cwd, *args: subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
assert paths and not git(ROOT, 'diff', '--cached', '--name-only')
write(ROOT/'tmp/MOMENT_base512_PSM42_publication_manifest_20261015_v1.json', dict(previous_root=git(ROOT, 'rev-parse', 'HEAD'),
    previous_parent=git(ROOT.parent/'flow-matching-research', 'rev-parse', 'HEAD'), paths=sorted(paths)))
print(json.dumps(dict(new_closed_case=name, collector_closed=state['verified_count'], exact_files=len(paths), primary=primary)))
