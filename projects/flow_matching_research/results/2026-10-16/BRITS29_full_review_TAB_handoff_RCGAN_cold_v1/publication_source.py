"""Freeze actual complete-model evidence and metadata handoffs; never run inference."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_rcgan_waiting_host_cold_v5 as cold
from scripts.flow_matching import run_saits_completion_replay_queue_v1 as collector
from scripts.flow_matching.verify_saits_completed_full_replay_batch_v1 import check_terminal, verify_registration

root = Path.cwd()
read, sha = cold.read, cold.sha
out = root / 'projects/flow_matching_research/results/2026-10-16/BRITS29_full_review_TAB_handoff_RCGAN_cold_v1'
out.mkdir(parents=True, exist_ok=False)
paths = []
mapping = []


def write(name, value):
    with (out / name).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def snapshot(path, name, publish_original=False):
    source = root / path
    target = out / name
    assert not target.exists()
    shutil.copy2(source, target)
    assert source.read_bytes() == target.read_bytes()
    mapping.append(dict(path=str(path), snapshot=name, sha256=sha(source)))
    if publish_original:
        paths.append(source.relative_to(root).as_posix())


def observe(handle):
    try:
        process = psutil.Process(handle['pid'])
        actual = dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(),
                      cpu_seconds=sum(process.cpu_times()[:2]), private_bytes=process.memory_info().private,
                      descendants=[dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline())
                                   for p in process.children(recursive=True)])
        actual['exact_identity'] = (actual['create_time'] == handle['create_time']
                                    and ('command' not in handle or actual['command'] == handle['command']))
        return actual
    except psutil.NoSuchProcess:
        return dict(pid=handle['pid'], alive=False)


identity = 'saits_paper_brits_electricity_author_missing0.5_seed29'
review_path = 'configs/reproducibility/fm_saits_completion_replay_queue_v1/' + identity + '.review.json'
review = read(root / review_path)
producer = verify_registration(review)
native_path = root / producer['artifact_directory'] / 'checkpoint_replay_audit.json'
proof_path = root / review['output_directory'] / 'independent_full_replay_batch_verification.json'
native, proof = read(native_path), read(proof_path)
native_receipt = read(root / producer['state_root'] / 'receipt.json')
review_receipt = read(root / review['state_root'] / 'receipt.json')
check_terminal(native, native_receipt, producer)
assert review_receipt['exit_code'] == 0 and review_receipt['full_native_batch_independent_review_passed']
assert not review_receipt['resource_guard_aborted']
assert proof['passed'] and proof['configuration_sha256'] == sha(root / review_path)
assert proof['native_replay_report_sha256'] == sha(native_path)
record = proof['records'][0]
assert record['id'] == identity and proof['all_original_grid_coordinates_independently_checked']
result_path = root / 'experiments/runs/flow_matching_campaign' / identity / 'result.json'
result = read(result_path)
assert record['hashes']['result'] == sha(result_path)
assert record['full_test_shape'] == [290, 100, 370] and record['held_out_coordinates'] == 4221734
assert record['full_training_completion']['next_epoch'] == 860
assert record['full_training_completion']['author_patience_remaining'] == 0

# Verify the complete evidence union without treating a status counter as proof.
runtime = read(root / 'configs/runtime/fm_saits_completion_replay_queue.v1.json')
jobs, template = collector.verify(runtime)
all_reviews = set(runtime['known_completed_independent_configurations'])
all_reviews.update(p.relative_to(root).as_posix()
                   for p in (root / runtime['generated_config_root']).glob('*.review.json'))
verified = {}
for path in sorted(all_reviews):
    config = read(root / path)
    rp = root / config['output_directory'] / 'independent_full_replay_batch_verification.json'
    terminal_path = root / config['state_root'] / 'receipt.json'
    if not rp.exists() or not terminal_path.exists():
        continue
    report, terminal = read(rp), read(terminal_path)
    if terminal.get('exit_code') != 0 or terminal.get('resource_guard_aborted'):
        continue
    assert terminal['full_native_batch_independent_review_passed'] and report['passed']
    assert report['all_original_grid_coordinates_independently_checked']
    source = read(root / config['replay_configuration'])
    native_report_path = root / source['artifact_directory'] / 'checkpoint_replay_audit.json'
    nr = read(native_report_path)
    check_terminal(nr, read(root / source['state_root'] / 'receipt.json'), source)
    assert report['configuration_sha256'] == sha(root / path)
    assert report['native_replay_report_sha256'] == sha(native_report_path)
    assert nr['configuration_sha256'] == sha(root / config['replay_configuration'])
    registered = {row['id']: row for row in source['completed_jobs']}
    assert set(registered) == {row['id'] for row in report['records']}
    for row in report['records']:
        key = row['id']
        assert key in jobs
        item = jobs[key]
        native_config = item['native']
        original = root / native_config['output_root'] / 'result.json'
        assert row['hashes']['config'] == sha(root / item['item']['config'])
        assert row['hashes']['result'] == sha(original) == registered[key]['result_sha256']
        assert row['full_training_completion']['full_training_finished']
        entry = dict(id=key, method=native_config['method'], dataset=native_config['dataset'],
                     nominal_missing_ratio=native_config['missing_ratio'], seed=native_config['seed'],
                     metrics=row['independently_recomputed_metrics'],
                     original_result_sha256=row['hashes']['result'],
                     independent_review_configuration=path, proof_path=str(rp.relative_to(root)),
                     proof_sha256=sha(rp), review_guard_sha256=sha(terminal_path))
        if key in verified:
            assert verified[key]['original_result_sha256'] == entry['original_result_sha256']
        else:
            verified[key] = entry
status = read(root / runtime['state_root'] / 'status.json')
assert set(verified) == set(status['verified_original_ids']) and len(verified) == 58
write('actual_58_complete_native_and_independent_model_union.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), required_models=210,
    verified_models=58, remaining_models=152, records=[verified[k] for k in sorted(verified)],
    verification='All review/native success guards, config/report/result hashes and original model membership checked. '
                 'Existing full-array reviews reused; this publication executes no new inference or training.',
    additional_training_seed_slots=0, all_original_experiments_complete=False))

snapshot(review_path, 'review_configuration.json', True)
snapshot(review['replay_configuration'], 'native_configuration.json', True)
snapshot(proof_path, 'independent_full_replay_batch_verification.json', True)
snapshot(root / review['output_directory'] / 'native_checkpoint_replay_audit.json',
         'native_checkpoint_replay_audit.json', True)
snapshot(root / producer['state_root'] / 'receipt.json', 'native_guard_receipt.json')
snapshot(root / review['state_root'] / 'receipt.json', 'independent_guard_receipt.json')
snapshot(result_path, 'original_result.json')
with (out / 'new_original_model_metrics.csv').open('x', encoding='utf-8', newline='') as handle:
    fields = ['id', 'method', 'dataset', 'nominal_missing_ratio', 'realized_missing_fraction', 'seed',
              'test_windows', 'full_grid_cells', 'heldout_coordinates', 'test_months',
              'mae', 'rmse', 'mre', 'mae_month_ci_low', 'mae_month_ci_high',
              'best_epoch', 'next_epoch', 'native_and_independent_verified', 'author_equivalence_certified', 'strict_TAB_result']
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerow(dict(id=identity, method='BRITS', dataset='Electricity', nominal_missing_ratio=.5,
                         realized_missing_fraction=4221734/10730000, seed=29, test_windows=290,
                         full_grid_cells=10730000, heldout_coordinates=4221734, test_months=10,
                         **record['independently_recomputed_metrics'],
                         mae_month_ci_low=result['month_block_bootstrap_95_ci']['mae'][0],
                         mae_month_ci_high=result['month_block_bootstrap_95_ci']['mae'][1],
                         best_epoch=result['validation_best']['epoch'], next_epoch=860,
                         native_and_independent_verified=True, author_equivalence_certified=False, strict_TAB_result=False))

config = read(root / cold.CONFIG)
cold.verify(config)
base = root / config['state_root']
handoff = read(base / 'handoff.json')
before = read(base / 'before_handoff_resources.json')
live = observe(read(base / 'launch.json'))
assert live['exact_identity'] and handoff['successor_pid'] == live['pid']
assert handoff['waiting_controller_only_stopped'] and not handoff['original_native_child_stopped']
assert not psutil.pid_exists(config['previous_owner']['pid'])
assert before['immutable_proof_binding_sha256'] == sha(root / config['immutable_proof_binding'])
cold_state = read(root / config['previous_state_path'])
assert cold_state['pid'] == live['pid'] and cold_state['create_time'] == live['create_time']
assert cold_state['active_pid'] is None and cold_state['state'] == 'waiting_for_memory'
assert not Path(config['recovery_output_directory']).exists()
write('actual_rcgan_cold_metadata_continuation.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), original_private_bytes=before['private_bytes'],
    actual_successor=live, private_bytes_freed=before['private_bytes']-live['private_bytes'],
    original_status=cold_state, full_original_epochs=500, verified_prior_epochs=238, remaining_full_epochs=262,
    required_seed_slots=5, proof_binding_bytes_unchanged=True, scientific_worker_started=False,
    original_native_child_stopped=False, additional_training_seed_slots=0, all_original_experiments_complete=False))
snapshot(base / 'handoff.json', 'rcgan_actual_handoff.json')
snapshot(base / 'handoff_identity.json', 'rcgan_handoff_identity.json')
snapshot(base / 'before_handoff_resources.json', 'rcgan_before_handoff_resources.json')
snapshot(base / 'successor_entry.json', 'rcgan_successor_entry.json')
snapshot(config['immutable_proof_binding'], 'preserved_rcgan_native_proof_binding.json')
xml = Path('F:/aicoding/IIA_Data/experiments/RCGAN_cold_host_published_tests_v5.xml')
suites = list(ET.parse(xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 18
assert sum(int(s.get('failures', 0))+int(s.get('errors', 0)) for s in suites) == 0
snapshot(xml, 'validation18.xml')
for index, path in enumerate([cold.MODULE.replace('.', '/')+'.py', cold.TEST, cold.CONFIG]):
    snapshot(path, 'source%02d%s' % (index, Path(path).suffix), True)

tab_base = root / 'experiments/runs/fm_TAB878_fair_entity_continuation_v2'
tab_handoff = read(tab_base / 'handoff.json')
tab_state = read(tab_base / 'status.json')
tab_live = observe(dict(pid=tab_state['pid'], create_time=tab_state['create_time'], command=tab_state['command']))
assert tab_live['exact_identity'] and tab_handoff['successor_pid'] == tab_live['pid']
assert not tab_handoff['original_numeric_child_stopped'] and tab_handoff['waiting_metadata_only_stopped']
assert not psutil.pid_exists(tab_handoff['previous_owner']['pid'])
assert tab_state['completed_entities']['e8a311b6a8a4968fb22a'] >= 19
write('actual_TAB_safe_handoff_and_preserved_progress.json', dict(
    captured_utc=datetime.now(timezone.utc).isoformat(), successor=tab_live, handoff=tab_handoff,
    state=tab_state, full_original_cases=878, full_original_label_views=2057,
    completed_full_cases=0, full_54_entity_case_complete=False, original_numeric_worker_unchanged=True,
    all_original_experiments_complete=False))
snapshot(tab_base / 'handoff.json', 'TAB_actual_handoff.json')
snapshot(root / 'experiments/runs/fm_TAB878_queued_boundary_bridge_v1/status.json', 'TAB_bridge_terminal_state.json')
write('source_bindings.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), files=mapping,
    numerical_budgets_or_frozen_worker_code_changed=False,
    original_scope=dict(papers=45, method_baseline_ablation_records=681, data_tasks=273, paper_scalar_targets=702),
    all_original_experiments_complete=False))
(out / 'README.md').write_text('''# 新增完整 BRITS 复核与已发生的安全接续

BRITS Electricity、名义 50% 缺失、种子 29 已完成作者早停训练、原生完整检查点重推理及独立逐坐标复核；两项守卫均退出 0，未发生资源中止。验证集选中第 828 轮，完成至 next_epoch=860，作者耐心值耗尽。模型 6,993,328 参数；完整测试 290×100×370，包含 4,221,734 个留出坐标、10 个测试月份。

独立复算 MAE=1.0821324741229577、RMSE=1.5831792391433428、MRE=0.579319952468947，与原结果及完整重推理在预注册 rtol=atol=2e-6 内一致。原始结果的月份区块 MAE 95% 区间为 [1.0585099565693508, 1.10691615268382]；这是同一训练模型的月份敏感性区间，不是五次训练种子的置信区间。

独立证据并集已增至 58/210；此次发布逐一核对成功的原生/独立复核守卫、配置/报告/结果哈希及原始任务成员，现有逐坐标复核结果直接复用。其余 152 项仍待完成。发布过程没有新增推理或训练。名义 50% 的作者有放回掩码实际留出比例约 39.345%；原始预处理种子/逐缺失率调参历史仍不可核实，作者窗口数 1397 与论文 1400 不符。不能据此认证与论文数值完全一致，也不能把插补误差当成异常检测 TAB 成绩。

此前归档为等待中的 TAB 安全接续现已实际完成：仅停掉无数值子进程的旧元数据控制器，原指标工作进程未停止。种子 1104 的完整 54 实体案例已有至少 19 个实体完成；种子 1103 的 6 个成功片段及第 7 个资源中止片段全部保留。878 项/2057 标签视图、原 21 指标/201 次调用、完整 VUS 网格及资源约束不变；仍未完成任何完整 TAB 案例。

RCGAN 的旧空闲控制器实际占用 322,596,864 字节私有内存，已按 PID/创建时间/完整命令/全部子进程二次检查后安全接续；新控制器约 20 MiB，节省约 289 MiB。仅修复元数据证明文件已存在时无法重新进入等待控制器的问题：只允许值及原始字节 SHA256 均完全相同时复用，绝不覆盖既有证明。反向恢复这一处元数据 AST 变更即可得到完整原控制器 AST。5 个原始种子、每种子 500 轮、当前种子已验证 238 轮和剩余 262 轮、原 Py37 数值工作进程、优化器/RNG/评估器及 40 GiB 准入/16 GiB 紧急保护全部保留。没有重算已经验证的训练，也未启动新数值子进程；新控制器仍等待原资源门槛。18 项相关测试通过。

原始 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标以及原生重推理、完整 TAB 和五训练种子统计继续保留；本包是新增真实完成项和安全接续的证据，不是全部实验完成声明。
''', encoding='utf-8', newline='\n')
snapshot(Path(__file__), 'publication_source.py')
git = lambda p: subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()
manifest = dict(previous_root=git(root), previous_parent=git(root.parent/'flow-matching-research'),
                paths=paths+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root / 'tmp/BRITS29_TAB_RCGAN_exact_publication_v1.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(dict(published_files=len(manifest['paths']), verified_models=len(verified),
                     cold_tests=18, rcgan_private_freed_bytes=before['private_bytes']-live['private_bytes'],
                     tab_completed_entities=tab_state['completed_entities'], all_original_experiments_complete=False)))
