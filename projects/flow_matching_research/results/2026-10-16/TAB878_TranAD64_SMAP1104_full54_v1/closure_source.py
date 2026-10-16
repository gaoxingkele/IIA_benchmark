"""Verify every original cell and independently audit fixed-model TAB aggregates."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import psutil
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as s
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as independent
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

c = read(ROOT/s.CONFIG)
entries = s.verify(c)
entry = next(e for e in entries if e['key'] == 'e8a311b6a8a4968fb22a')
assert entry['case']['seed'] == 1104 and entry['case']['family'] == 'tranad64'
bound = s.ready_case(c, entry)
assert bound is not None and len(bound['dataset']['entities']) == 54
attempt, output = s.paths(c, entry, 'summary')
summary_path = output/'complete_case_TAB_metric_controls.json'
summary, summary_guard = read(summary_path), read(attempt/'guard_receipt.json')
assert summary['passed'] and summary['entities'] == 54 and summary['metric_calls'] == 43416
assert summary['case'] == entry['case'] and summary['source_lineage'] == bound['lineage']
assert summary['configuration_sha256'] == sha(ROOT/s.CONFIG)
assert summary_guard['passed'] and summary_guard['exit_code'] == 0 and not summary_guard['resource_guard_aborted']
assert summary_guard['proof_sha256'] == sha(summary_path)
assert summary_guard['exact_child_handle']['command'] == s.worker_command(c, entry, 'summary')
for flag in ['undefined_metrics_preserved', 'original_local_training_fidelity_limits_retained']:
    assert summary[flag]
assert not summary['cross_entity_event_join_performed'] and not summary['source_training_or_scoring_modified']
assert not summary['full_TAB_training_or_author_harness_equivalence']
lineage = bound['lineage']
source_scores = ROOT/bound['job']['output_directory']/'scores.npz'
assert sha(source_scores) == lineage['source_binding']['scores_sha256']
assert sha(lineage['native_scores']) == lineage['native_scores_sha256']
assert source_scores.read_bytes() == Path(lineage['native_scores']).read_bytes()
levels = s.thresholds(c, bound['result'])
policies = entry['case']['required_label_policies']
assert len(policies) == 4 and len(levels) == 13
fragments, archive_inputs, entity_rows, bindings = [], {}, [], []
counts = Counter()
micro = {(p, level, adjusted): Counter() for p in policies for level in levels for adjusted in [False, True]}
label_payloads = {}
for policy in policies:
    label_record = bound['dataset']['labels_only_controls'][policy]
    assert sha(label_record['path']) == label_record['sha256']
    with np.load(label_record['path'], allow_pickle=False) as z:
        label_payloads[policy] = {name: z[name] for name in z.files}

def digest(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), content_sha256=hashlib.sha256(a.tobytes()).hexdigest())

def check_value(cell, value):
    if value is None:
        assert cell['status'] in ['error', 'nonfinite'] and cell['value'] is None
    else:
        assert cell['status'] == 'finite' and math.isclose(cell['value'], value, rel_tol=1e-12, abs_tol=1e-12)

def predicted_confusion(actual, predicted):
    out = Counter()
    for a, b in zip(actual, predicted):
        out['tp' if a and b else 'fn' if a else 'fp' if b else 'tn'] += 1
    return out

def values(confusion):
    tp, fp, fn, tn = (confusion[k] for k in ['tp', 'fp', 'fn', 'tn'])
    return dict(accuracy=(tp+tn)/(tp+fp+fn+tn), precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f_score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def adjust_events(actual, predicted):
    adjusted = list(predicted)
    i = 0
    while i < len(actual):
        if not actual[i]:
            i += 1; continue
        end = i+1
        while end < len(actual) and actual[end]:
            end += 1
        if any(predicted[i:end]):
            adjusted[i:end] = [True]*(end-i)
        i = end
    return adjusted

independent_cells = 0
total_points = 0
with np.load(lineage['native_scores'], allow_pickle=False) as scores:
    for entity in bound['dataset']['entities']:
        prefix = entity['array_prefix']
        path, report = s.checked_fragment(c, entry, prefix)
        guard_path = s.paths(c, entry, prefix)[0]/'guard_receipt.json'
        assert report['entity'] == entity['entity_id'] and report['source_lineage'] == lineage
        assert report['environment'] == c['environment']
        assert report['metric_calls'] == 804 and report['test_points'] == entity['test_points']
        assert not report['entities_concatenated'] and report['original_inputs_and_thresholds_unchanged']
        assert not report['neural_training_or_inference_performed']
        score_array = scores[prefix+'_test']
        assert digest(score_array) == report['source_scores_digest']
        total_points += len(score_array)
        raw_scores = score_array.tolist()
        per_entity = Counter()
        for policy in policies:
            actual_array = label_payloads[policy][prefix+'_labels']
            assert digest(actual_array) == report['label_digests'][policy]
            actual = actual_array.tolist()
            assert len(actual) == len(raw_scores)
            ranking = independent.ranking(raw_scores, actual)
            cells = report['metric_results'][policy]
            check_value(cells['score/auc_roc'], ranking['auroc'])
            check_value(cells['score/auc_pr'], ranking['average_precision'] if sum(actual) else 0.)
            independent_cells += 2
            for level, threshold in levels.items():
                predicted = [v > threshold for v in raw_scores]
                for adjusted in [False, True]:
                    selected = adjust_events(actual, predicted) if adjusted else predicted
                    confusion = predicted_confusion(actual, selected)
                    micro[(policy, level, adjusted)].update(confusion)
                    expected = values(confusion)
                    two_classes = bool(confusion['tp']+confusion['fn']+confusion['fp']) and bool(confusion['tn']+confusion['fn']+confusion['fp'])
                    for metric, value in expected.items():
                        # TAB indexes class1 in precision_recall_fscore_support;
                        # a genuinely single-class vector yields an error there.
                        if metric != 'accuracy' and not two_classes:
                            value = None
                        name = ('adjust_' if adjusted else '')+metric
                        check_value(cells[level+'/'+name], value)
                        independent_cells += 1
            for name, cell in cells.items():
                per_entity[cell['status']] += 1
        assert dict(per_entity) == report['metric_outcome_counts']
        counts.update(per_entity)
        fragments.append(report)
        bindings.append(dict(entity=entity['entity_id'], array_prefix=prefix, points=len(score_array),
            report_path=str(path), report_sha256=sha(path), guard_path=str(guard_path), guard_sha256=sha(guard_path)))
        archive_inputs[prefix+'/entity_TAB_metric_suite.json'] = path
        archive_inputs[prefix+'/guard_receipt.json'] = guard_path
        entity_rows.append(dict(entity=entity['entity_id'], array_prefix=prefix, points=len(score_array), **per_entity))
assert total_points == 435826 and sum(counts.values()) == 43416
assert independent_cells == 54*4*(2+13*8) == 22896
assert summary['fragments'] == [dict(path=x['report_path'], sha256=x['report_sha256']) for x in bindings]

# Same declared seeded bootstrap indices; independent stdlib fsum/quantiles.
indices = np.random.default_rng(entry['case']['seed']+10000).integers(0, 54, size=(200, 54)).tolist()
def quantile(items, q):
    ordered = sorted(items); index = (len(ordered)-1)*q
    low = int(index); high = min(low+1, len(ordered)-1)
    return ordered[low] + (index-low)*(ordered[high]-ordered[low])

macro_rows = []
for policy in policies:
    assert len(summary['macro_metrics'][policy]) == 201
    for metric, aggregate in summary['macro_metrics'][policy].items():
        cells = [f['metric_results'][policy][metric] for f in fragments]
        states = Counter(x['status'] for x in cells)
        vals = [x['value'] for x in cells if x['status'] == 'finite']
        complete = len(vals) == 54
        assert aggregate['total_entities'] == 54 and aggregate['finite_entities'] == len(vals)
        assert aggregate['error_entities'] == states['error'] and aggregate['nonfinite_entities'] == states['nonfinite']
        expected = statistics.fmean(vals) if vals else None
        for field in ['defined_subset_macro_mean', 'complete_entity_macro_mean']:
            value = expected if field == 'defined_subset_macro_mean' or complete else None
            assert (aggregate[field] is None and value is None) or (value is not None and math.isclose(aggregate[field], value, rel_tol=1e-12, abs_tol=1e-12))
        if complete:
            draws = [math.fsum(vals[i] for i in index)/54 for index in indices]
            ci = [quantile(draws, .025), quantile(draws, .975)]
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(aggregate['complete_entity_macro_CI95'], ci))
            assert aggregate['CI_resamples'] == 200
        else:
            assert aggregate['complete_entity_macro_CI95'] is None and aggregate['CI_resamples'] == 0
        assert not aggregate['training_seed_uncertainty']
        assert aggregate['CI_unit'] == 'entity_conditional_on_fixed_model_and_threshold'
        macro_rows.append(dict(label_policy=policy, metric=metric, **aggregate))
assert len(macro_rows) == 804
protocol_rows = []
for policy in policies:
    for level, threshold in levels.items():
        a, b = summary['macro_metrics'][policy][level+'/f_score'], summary['macro_metrics'][policy][level+'/adjust_f_score']
        protocol_rows.append(dict(label_policy=policy, threshold_protocol=level, threshold=threshold,
            full54_macro_F1=a['complete_entity_macro_mean'], full54_macro_PA_F1=b['complete_entity_macro_mean'],
            macro_F1_defined_entities=a['finite_entities'], macro_PA_F1_defined_entities=b['finite_entities'],
            full_micro_F1=values(micro[(policy, level, False)])['f_score'],
            within_entity_PA_micro_F1=values(micro[(policy, level, True)])['f_score'],
            threshold_uses_TEST_scores=level.startswith('diagnostic_pooled_TEST/'), PA_uses_TEST_labels=True,
            threshold_optimized_on_TEST_F1=False))
assert len(protocol_rows) == 52
test_xml = Path('F:/aicoding/IIA_Data/experiments/TAB878_TranAD64_SMAP1104_full54_closure_20261016_v1_tests.xml')
suites = list(ET.parse(test_xml).getroot().iter('testsuite'))
assert all(int(x.get('failures', 0)) == int(x.get('errors', 0)) == int(x.get('skipped', 0)) == 0 for x in suites)
packet = ROOT/'projects/flow_matching_research/results/2026-10-16/TAB878_TranAD64_SMAP1104_full54_v1'
packet.mkdir(exist_ok=False)
(packet/'.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n*.zip -text\n', encoding='utf-8')
shutil.copyfile(summary_path, packet/'complete_case_TAB_metric_controls.json')
shutil.copyfile(attempt/'guard_receipt.json', packet/'actual_complete_summary_guard.json')
shutil.copyfile(test_xml, packet/'relevant_validation.xml')
with zipfile.ZipFile(packet/'all54_original_entity_reports_and_guards.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for name, path in sorted(archive_inputs.items()):
        z.writestr(name, path.read_bytes())
with zipfile.ZipFile(packet/'all54_original_entity_reports_and_guards.zip') as z:
    assert len(z.namelist()) == 108 and z.testzip() is None
    for name, path in archive_inputs.items():
        assert z.read(name) == path.read_bytes()
for filename, rows in [('all804_entity_macro_metrics.csv', macro_rows), ('all52_threshold_and_PA_protocol_controls.csv', protocol_rows), ('all54_entity_outcome_coverage.csv', entity_rows)]:
    with (packet/filename).open('w', encoding='utf-8-sig', newline='') as f:
        headers = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=headers); writer.writeheader(); writer.writerows(rows)
write(packet/'all54_immutable_original_report_guard_bindings.json', dict(records=bindings, source_lineage=lineage))
write(packet/'independent_verification.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    all54_entities_original_guards_and_source_lineage_reverified=True, complete_test_points=total_points,
    complete_original_TAB_calls=43416, complete_original_outcomes=dict(counts),
    separately_recomputed_full_point_raw_PA_confusions_and_tie_ranking_cells=independent_cells,
    all804_macro_means_coverage_and_200_entity_bootstraps_independently_recomputed=True,
    range_affiliation_VUS_definition_reexecution_by_this_closure=False,
    original_range_affiliation_VUS_successful_guarded_calls_preserved=True,
    original_TAB_commit=c['official_TAB_commit'], original_frozen_environment=c['environment'],
    all_metric_source_pins_reverified=True, new_model_training_inference_or_seed_slots=0,
    full_TAB_training_or_author_harness_equivalence=False, all_original_experiments_complete=False))
state_path = ROOT/'experiments/runs/fm_TAB878_fair_entity_continuation_v2/status.json'
state = read(state_path); owner = psutil.Process(state['pid'])
assert owner.create_time() == state['create_time'] and entry['key'] in state['completed_case_keys']
write(packet/'actual_live_continuation.json', dict(pid=owner.pid, create_time=owner.create_time(), command=owner.cmdline(),
    state=state['state'], completed_case_keys=state['completed_case_keys'], failed_cases_preserved=state['failed_cases_preserved']))
write(packet/'iteration_status.json', dict(original_scope=c['original_scope'], TAB_metric_control_complete_cases=1,
    TAB_required_cases=878, complete_label_views=4, required_label_views=2057, full_entities=54, macro_metric_rows=804,
    complete_five_training_seed_cohorts=0, new_training_seed_slots=0, all_original_experiments_complete=False))
lines = ['# TranAD FP64／SMAP／seed1104：全部54实体的原始TAB指标对照', '',
    '完整435826测试点、54原始实体、4种原始标签边界全部保留。每个实体/标签执行原TAB6个分数指标和15个标签指标×13个固定阈值，共43416次实际调用；全部54个实体守卫及汇总守卫通过。完整原始报告与守卫按原字节打包，804行实体macro指标和52行阈值/PA/聚合方式对照可直接查看。', '',
    '| 原始标签边界 | 验证1%实体macro F1 | 同阈值PA实体macro F1 | 验证1% micro F1 | 同阈值PA micro F1 |', '|---|---:|---:|---:|---:|']
for row in protocol_rows:
    if row['threshold_protocol'] == 'validation/1':
        lines.append(f"| {row['label_policy']} | {row['full54_macro_F1']:.9f} | {row['full54_macro_PA_F1']:.9f} | {row['full_micro_F1']:.9f} | {row['within_entity_PA_micro_F1']:.9f} |")
lines += ['',
    '逐点独立重算22896个AUROC/AP、原始/PA混淆计数衍生指标，核对源与原生完整评分档案字节及全部标签摘要；另外使用标准库均值/线性分位数和原预注册bootstrap索引，独立核验全部804项macro均值、有效覆盖和200次实体bootstrap区间。没有重新执行耗时的range/affiliation/VUS定义；这些原始完整调用及异常/警告由实际原守卫和完整原报告保留，不能写成全部指标有第二套数学实现。',
    '只在54实体均为有限值时给出完整macro均值和区间；部分未定义指标保留空白，另列defined-subset均值和明确分母。该区间仅反映一个固定训练模型上的实体条件不确定性，不是五训练种子置信区间。',
    '验证阈值和使用训练+测试分数的原固定比例诊断分开列出。PA用测试真值扩展异常段，不能与无PA混作同一指标；未对测试F1选择最佳阈值或最佳标签边界，未跨实体连接异常段。macro与micro不同，也不能直接互换。',
    '这是原始TAB函数的完整固定评分对照，保留本地原始训练的局限，并非TAB完整训练/数据流程或历史作者实现认证。此轨道1/878完整案例、4/2057标签视图，只有一个训练种子，不增加新的模型训练次数。common105的577案例属于另一轨道，不能合并抵扣。',
    '所有原论文/TAB/新数据轨道及45篇/681方法记录/273数据任务/702原始标量目标保持；整体目标未完成。']
(packet/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
shutil.copyfile(__file__, packet/'closure_source.py')
write(packet/'packet_integrity.json', dict(files=[dict(path=x.relative_to(packet).as_posix(), sha256=sha(x)) for x in sorted(packet.rglob('*')) if x.is_file()]))
ara = ROOT/'projects/flow_matching_research/ara/execution/TAB878_TranAD64_SMAP1104_full54.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(result_packet=packet.relative_to(ROOT).as_posix(), complete_fixed_model_metric_control_cases=1,
    required_cases=878, full_TAB_training_or_author_harness_equivalence=False, all_original_experiments_complete=False))
head = lambda p: subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD']).decode().strip()
write(ROOT/'tmp/TAB878_TranAD64_SMAP1104_full54_manifest_20261016_v1.json', dict(previous_root=head(ROOT), previous_parent=head(ROOT.parent/'flow-matching-research'),
    paths=sorted([ara.relative_to(ROOT).as_posix()]+[x.relative_to(ROOT).as_posix() for x in packet.rglob('*') if x.is_file()])))
print(json.dumps(dict(packet=str(packet), original_TAB_metric_calls=43416, independent_cells=independent_cells,
    macro_rows=804, outcomes=dict(counts), validation_one_percent=[x for x in protocol_rows if x['threshold_protocol']=='validation/1'])))
