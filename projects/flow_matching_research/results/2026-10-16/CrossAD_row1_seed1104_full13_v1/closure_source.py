"""Archive exactly thirteen actual conditions, without launching numerical work."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import collect_crossad_full139_cold_handoff_v1 as cold
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as groups
from scripts.flow_matching import audit_crossad_full139_independent_metrics_v1 as metrics
from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

cc = read(ROOT / cold.CONFIG)
original, q = cold.verify(cc)
mc = read(ROOT / metrics.CONFIG)
_, upstream = metrics.verify(mc)
cs = read(ROOT / cc['state_root'] / 'status.json')
ms = read(ROOT / mc['state_root'] / 'status.json')
assert not cs['errors_preserved'] and not ms['failed_preserved']
prior = ROOT / 'projects/flow_matching_research/results/2026-10-16/CrossAD_row5_full12_v1'
previous = read(prior / 'all12_proof_bindings.json')['records']
identity = 'crossad_ablation__row1__gecco__seed1104'
required = set(previous) | {identity}
assert len(previous) == 12 and len(required) == 13
assert required.issubset(cs['verified_original_CUDA_ids'])
assert required.issubset(ms['verified_ids'])
proofs, metric_reports, strict_rows = {}, {}, []
for cid in sorted(required):
    job = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference'] == cid)
    proof_path = Path(original['artifact_root']) / 'conditions' / (cid + '.json')
    mp = Path(mc['artifact_root']) / 'conditions' / (cid + '.json')
    proof, metric = read(proof_path), read(mp)
    assert groups.condition(original, q, job) == proof
    assert metrics.audit(mc, q, upstream, job) == metric
    guard = read(ROOT / mc['state_root'] / 'guards' / (cid + '.json'))
    assert guard['passed'] and guard['exit_code'] == 0 and not guard['resource_guard_aborted']
    assert guard['report_sha256'] == sha(mp)
    proofs[cid], metric_reports[cid] = proof, metric
    if cid in previous:
        assert sha(proof_path) == previous[cid]['full_array_proof_sha256']
        assert sha(mp) == previous[cid]['independent_metric_report_sha256']
    for ratio, values in metric['independent_strict_metrics'].items():
        strict_rows.append(dict(canonical_id=cid, dataset=job['dataset'], training_or_eval_seed=job['seed'],
            execution_mode=job['mode'], reconstructed_ablation=job['reconstructed_ablation'],
            ablation_row=job.get('ablation_row'), validation_alarm_percent=ratio,
            threshold=values['threshold'], **values['micro']))
with (prior / 'all12_conditions_36_strict_metric_rows.csv').open(encoding='utf-8-sig', newline='') as f:
    old_rows = list(csv.DictReader(f))
textrow = lambda r: {k: '' if v is None else str(v) for k, v in r.items()}
new_rows = {(r['canonical_id'], str(r['validation_alarm_percent'])): textrow(r) for r in strict_rows}
assert len(old_rows) == 36 and len(strict_rows) == 39
assert all(new_rows[(r['canonical_id'], r['validation_alarm_percent'])] == r for r in old_rows)
cohorts = groups.cohorts(q['jobs'], proofs)
assert cohorts['complete_primary_cohorts'] == 1 and not cohorts['paired_seed_ablation_differences']
partial = cohorts['primary_cohorts']['GECCO/paper_reconstruction_row1']
assert partial['verified_seed_ids'] == [1103, 1104] and not partial['complete'] and 'metrics' not in partial
normalize = lambda x: json.loads(json.dumps(x))
old_cohorts = read(prior / 'all22_primary_cohort_completion.json')
assert normalize(cohorts['primary_cohorts']['GECCO/main']) == old_cohorts['primary_cohorts']['GECCO/main']
job = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference'] == identity)
model = read(ROOT / job['model_config'])
assert model['implementation'] == 'paper_reconstructed_author_harness'
assert model['parameters']['components'] == [False, False, False, False]
result_path = ROOT / job['output_directory'] / 'result.json'
result = read(result_path)
budget = training_budget(job, result['details'], read(result_path.parent / 'training_trace.json'))
assert budget['original_maximum_epochs'] == 20 and budget['original_patience'] == 3
paired_rows = []
for seed in [1103, 1104]:
    cid = f'crossad_ablation__row1__gecco__seed{seed}'
    main = f'crossad_author__fresh_train__gecco__seed{seed}'
    a, b = metric_reports[cid], metric_reports[main]
    assert a['complete_points'] == b['complete_points'] == dict(train=55408, validation=13852, test=69261)
    assert proofs[cid]['raw_sha256'] == proofs[main]['raw_sha256']
    for field in ['labels', 'source_rows']:
        assert a['all_score_label_and_timestamp_payload_sha256'][field] == b['all_score_label_and_timestamp_payload_sha256'][field]
    for ratio in ['0.1', '1', '5']:
        av, bv = a['independent_strict_metrics'][ratio]['micro'], b['independent_strict_metrics'][ratio]['micro']
        paired_rows.append(dict(training_seed=seed, validation_alarm_percent=ratio,
            row1_F1=av['f1'], full_F1=bv['f1'], row1_minus_full_F1=av['f1']-bv['f1'],
            row1_AUROC=av['auroc'], full_AUROC=bv['auroc'], row1_AP=av['average_precision'], full_AP=bv['average_precision']))
test_xml = Path('F:/aicoding/IIA_Data/experiments/CrossAD_row1_seed1104_full13_20261016_v1_tests.xml')
suites = list(ET.parse(test_xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) > 0
assert all(int(s.get('failures', 0)) == int(s.get('errors', 0)) == int(s.get('skipped', 0)) == 0 for s in suites)
packet = ROOT / 'projects/flow_matching_research/results/2026-10-16/CrossAD_row1_seed1104_full13_v1'
packet.mkdir(exist_ok=False)
(packet / '.gitattributes').write_text('*.json -text\n*.csv -text\n*.py -text\n*.xml -text -whitespace\n', encoding='utf-8')
paths = dict(source_result=result_path, training_trace=result_path.parent / 'training_trace.json',
    model_configuration=ROOT / job['model_config'],
    native_replay=Path(q['artifact_root']) / identity / 'native_replay/report.json',
    full_checkpoint_array_proof=Path(original['artifact_root']) / 'conditions' / (identity + '.json'),
    independent_metric_report=Path(mc['artifact_root']) / 'conditions' / (identity + '.json'),
    independent_metric_guard=ROOT / mc['state_root'] / 'guards' / (identity + '.json'),
    **{p + '_guard': ROOT / q['state_root'] / 'jobs' / identity / (p + '.guard.json') for p in ['CUDA_probe', 'CPU_reference', 'author', 'replay']})
for label, path in paths.items():
    shutil.copyfile(path, packet / (label + '.json'))
shutil.copyfile(test_xml, packet / 'relevant_validation.xml')
write(packet / 'actual_original_training_budget_verification.json', budget)
write(packet / 'all22_primary_cohort_completion.json', cohorts)
for filename, rows in [('all13_conditions_39_strict_metric_rows.csv', strict_rows), ('same_seed_row1_vs_full_two_seeds.csv', paired_rows)]:
    with (packet / filename).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
write(packet / 'all13_proof_bindings.json', dict(records={cid: dict(
    full_array_proof_path=str(Path(original['artifact_root']) / 'conditions' / (cid + '.json')),
    full_array_proof_sha256=sha(Path(original['artifact_root']) / 'conditions' / (cid + '.json')),
    independent_metric_report_path=str(Path(mc['artifact_root']) / 'conditions' / (cid + '.json')),
    independent_metric_report_sha256=sha(Path(mc['artifact_root']) / 'conditions' / (cid + '.json'))) for cid in sorted(required)},
    prior12_strict_metric_rows_unchanged=True, prior_main_full_five_seed_summary_unchanged=True))
handles = []
for state in [cs, ms]:
    proc = psutil.Process(state['pid'])
    assert proc.is_running() and proc.create_time() == state['create_time']
    handles.append(dict(pid=proc.pid, create_time=proc.create_time(), command=proc.cmdline()))
write(packet / 'actual_live_collectors.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), handles=handles))
write(packet / 'iteration_status.json', dict(original_scope=q['original_scope'], CrossAD_full_CUDA_conditions=13,
    CrossAD_required_conditions=139, complete_reconstructed_ablation_conditions=6,
    row1_verified_seeds=[1103, 1104], row1_required_seeds=groups.SEEDS,
    complete_primary_cohorts=1, required_primary_cohorts=22, original_CPU_completion_credit=0,
    new_numerical_executions_by_this_closure=0, all_original_experiments_complete=False))
a = metric_reports[identity]['independent_strict_metrics']['1']['micro']
b = metric_reports['crossad_author__fresh_train__gecco__seed1104']['independent_strict_metrics']['1']['micro']
(packet / 'README.md').write_text(f'''# CrossAD row1/seed1104：第13个完整GPU条件

本条件完成{budget['actual_completed_epochs']}轮、{budget['full_training_forward_calls']}个完整训练批次。原20轮上限、patience3、batch128及完整432批次/轮规则保持；初始化CPU/GPU梯度比较、作者运行和检查点重放四项守卫均退出0，无资源中止。

全部55408训练点、13852验证点、69261测试点（含13点尾部）逐字节重放一致。独立标准库再次从完整数组复算验证阈值、混淆计数、F1、AUROC和AP。验证集1%阈值、无PA：row1 F1={a['f1']:.12f}，AUROC={a['auroc']:.12f}，AP={a['average_precision']:.12f}；同种子完整模型F1={b['f1']:.12f}，差值={a['f1']-b['f1']:.12f}。

row1四个组件开关均为false，精确作者消融架构未发布，继续标记paper_reconstructed_author_harness。row1现有1103和1104两个训练种子，只有2/5；CSV给出两组同种子点值差，不计算五种子均值、区间或完整消融排名。其余四种重建消融各1/5。

累计13/139：6个新训练主模型、1个发布权重评估、6个重建消融条件。完整五种子主要组仍1/22。重新核验全部13个条件，前12个产物哈希、36行指标及主模型五种子统计均保持。原CPU运行不抵扣GPU条件，单条件源与重放等价也不是原论文/TAB性能认证。

本归档没有启动模型训练或推理，没有停止或改变在运行的数值进程。相关验证XML随附；其通过不计作新增实验完成。45篇论文、681方法记录、273数据任务、702原始指标目标及原论文/TAB/新数据轨道的全部未完成项继续保留，整体目标未完成。
''', encoding='utf-8')
shutil.copyfile(__file__, packet / 'closure_source.py')
write(packet / 'packet_integrity.json', dict(files=[dict(path=p.relative_to(packet).as_posix(), sha256=sha(p)) for p in sorted(packet.rglob('*')) if p.is_file()]))
ara = ROOT / 'projects/flow_matching_research/ara/execution/CrossAD_row1_seed1104_full13.2026-10-16_v1.json'
assert not ara.exists()
write(ara, dict(result_packet=packet.relative_to(ROOT).as_posix(), CrossAD_complete_CUDA_conditions=13,
    required_conditions=139, row1_complete_seeds=2, complete_primary_cohorts=1, all_original_experiments_complete=False))
head = lambda p: subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD']).decode().strip()
write(ROOT / 'tmp/CrossAD_row1_seed1104_full13_manifest_20261016_v1.json', dict(previous_root=head(ROOT),
    previous_parent=head(ROOT.parent / 'flow-matching-research'), paths=sorted([ara.relative_to(ROOT).as_posix()]
    + [p.relative_to(ROOT).as_posix() for p in packet.rglob('*') if p.is_file()])))
print(json.dumps(dict(packet=str(packet), full_conditions=13, row1_complete_seeds=2, one_percent_metrics=a,
    one_percent_F1_difference=a['f1']-b['f1'], all_original_experiments_complete=False)))
