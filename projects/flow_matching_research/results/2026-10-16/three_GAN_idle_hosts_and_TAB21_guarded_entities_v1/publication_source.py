"""Archive real handoffs and guarded official TAB entity cells, without inference."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import continue_three_gan_idle_hosts_cold_v1 as s
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as tab

root = s.ROOT
out = root/'projects/flow_matching_research/results/2026-10-16/three_GAN_idle_hosts_and_TAB21_guarded_entities_v1'
out.mkdir(parents=True, exist_ok=False)
mapping, originals = [], []


def write(name, value):
    with (out/name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def snapshot(path, name, publish=False):
    path = root/path
    assert not (out/name).exists()
    shutil.copy2(path, out/name)
    assert path.read_bytes() == (out/name).read_bytes()
    mapping.append(dict(path=str(path), sha256=s.sha(path), snapshot=name))
    if publish:
        originals.append(path.relative_to(root).as_posix())


config = s.read(root/s.CONFIG)
s.verify(config)
hosts = []
for name, role in config['roles'].items():
    base = root/role['state_root']
    handle = s.read(base/'launch.json')
    process = psutil.Process(handle['pid'])
    assert s.common.exact_live(process, handle)
    assert not psutil.pid_exists(role['previous_owner']['pid'])
    before = s.read(base/'before_handoff_resources.json')
    handoff = s.read(base/'handoff.json')
    assert handoff['previous_owner'] == role['previous_owner'] and handoff['successor_pid'] == process.pid
    assert handoff['waiting_controller_only_stopped'] and not handoff['original_native_child_stopped']
    s.already_handed_off(role, base)
    state = s.read(root/role['previous_state_path'])
    assert state['pid'] == process.pid and state['create_time'] == process.create_time()
    assert state['active_pid'] is None and not state['shared_lock_owned']
    descendants = s.common.descendants_of(process)
    assert all(s.common.allowed_console_child(p['name'], p['exe']) for p in descendants)
    assert (base/'controller.stderr.log').stat().st_size == 0
    private = process.memory_info().private
    assert private < 128*2**20
    row = dict(role=name, actual_owner=handle, original_private_bytes=before['private_bytes'],
        actual_private_bytes=private, private_bytes_freed=before['private_bytes']-private,
        state=state, descendants=descendants, prior_metadata_bytes_unchanged=True,
        original_native_child_stopped=False, numerical_child_started=False)
    hosts.append(row)
    for source, suffix in [('before_handoff_resources.json','before.json'), ('handoff.json','handoff.json'),
                           ('handoff_identity.json','identity.json'), ('successor_entry.json','entry.json')]:
        snapshot(base/source, name+'_'+suffix)
write('actual_three_host_continuations.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    roles=hosts, private_bytes_freed=sum(r['private_bytes_freed'] for r in hosts),
    original_TimeGAN_jobs=45, energy_recovery_occupies_existing_canonical_seed=True,
    original_GTGAN_commands=4, native_generations=29, trained_evaluators=580, native_network_checkpoints=116,
    additional_model_training_seed_slots=0, all_original_experiments_complete=False))

for index, source in enumerate([s.MODULE.replace('.','/')+'.py', s.TEST, s.CONFIG]):
    snapshot(source, 'source%02d%s'%(index,Path(source).suffix), True)
tests = 0
for source, name in [(Path('F:/aicoding/IIA_Data/experiments/three_gan_idle_cold_tests_v1.xml'),'new24.xml'),
                     (Path('F:/aicoding/IIA_Data/experiments/three_gan_existing_handoff_resources_v1.xml'),'existing18.xml')]:
    suites = list(ET.parse(source).getroot().iter('testsuite'))
    assert not sum(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites)
    tests += sum(int(x.get('tests',0)) for x in suites)
    snapshot(source, name)
assert tests == 42

# Every credited entity has an actual exact worker exit0 guard and complete cells.
tc = tab.read(root/tab.CONFIG)
entry = next(e for e in tc['entries'] if e['key']=='e8a311b6a8a4968fb22a')
bound = tab.ready_case(tc, entry)
assert bound is not None and len(bound['dataset']['entities']) == 54
state_path = root/'experiments/runs/fm_TAB878_fair_entity_continuation_v2/status.json'
state = s.read(state_path)
controller = psutil.Process(state['pid'])
assert controller.create_time() == state['create_time'] and controller.cmdline() == state['command']
assert state['completed_entities'][entry['key']] >= 21
ledger, outcomes = [], Counter()
for index in range(21):
    prefix = 'e%03d'%index
    path, proof = tab.checked_fragment(tc, entry, prefix)
    assert proof['source_lineage'] == bound['lineage']
    entity = next(e for e in bound['dataset']['entities'] if e['array_prefix']==prefix)
    assert entity['entity_id'] == proof['entity'] and entity['test_points'] == proof['test_points']
    attempt, _ = tab.paths(tc, entry, prefix)
    guard = attempt/'guard_receipt.json'
    outcomes.update(proof['metric_outcome_counts'])
    ledger.append(dict(entity=proof['entity'], array_prefix=prefix, points=proof['test_points'],
        metric_calls=proof['metric_calls'], outcomes=proof['metric_outcome_counts'],
        proof_path=str(path), proof_sha256=s.sha(path), guard_path=str(guard), guard_sha256=s.sha(guard)))
    if index in [19,20]:
        snapshot(path, prefix+'_full_official_TAB_cells.json')
        snapshot(guard, prefix+'_success_guard.json')
        with (out/(prefix+'_full_official_TAB_cells.csv')).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['case_id','seed','entity','label_policy','metric_threshold','status','value','error','warnings','original_stdout_stderr'])
            writer.writeheader()
            for policy, cells in proof['metric_results'].items():
                for metric, cell in cells.items():
                    writer.writerow(dict(case_id=entry['case']['id'],seed=entry['case']['seed'],entity=proof['entity'],
                        label_policy=policy,metric_threshold=metric,status=cell['status'],value=cell['value'],
                        error=cell['error'],warnings=json.dumps(cell['warnings'],ensure_ascii=False),
                        original_stdout_stderr=cell['original_stdout_stderr']))
write('actual_21_complete_TAB_entity_guard_chain.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    case=entry['case'], required_entities=54, guarded_complete_entities=21,
    complete_metric_calls=sum(e['metric_calls'] for e in ledger), metric_outcome_counts=dict(outcomes),
    source_lineage=bound['lineage'], entities=ledger, actual_dispatch_state=state,
    complete_case=False, entity_subset_metrics_are_not_full_dataset_performance=True,
    nonfinite_or_error_cells_preserved_without_imputation=True, all_original_experiments_complete=False))

# Measure numerical workers directly, rather than trusting parent metadata.
native=[]
for pid, created in [(36912,1792097263.1796665),(40268,1792097260.102444),
                     (41992,1791948094.8603492),(1752,1792102057.2682695)]:
    process=psutil.Process(pid)
    assert process.create_time()==created
    native.append(dict(pid=pid,create_time=created,command=process.cmdline(),
                       cpu_seconds=sum(process.cpu_times()[:2]),private_bytes=process.memory_info().private))
progress=root/'experiments/runs/fm_cfm_node1106_complete_epoch_full_retry_v10/jobs/cfm_ts_original__main_text__node_e400__pendulum__seed1106/progress.json'
write('actual_original_numerical_worker_progress.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    workers=native, cfm_progress=s.read(progress),
    BRITS30_recent_training_lines=(root/'experiments/runs/flow_matching_campaign/saits_paper_brits_electricity_author_missing0.5_seed30/training.log').read_text().splitlines()[-4:],
    moment_zero_shot_scoring_in_progress=True, these_are_progress_counters_not_completed_benchmark_results=True,
    all_original_experiments_complete=False))
write('source_bindings_and_validation.json',dict(tests_passed=42,files=mapping,
    only_completed_metadata_handoff_callbacks_rebound=True,entire_original_main_code_objects_preserved=True,
    numerical_worker_commands_or_training_budgets_changed=False,original_scope=config['original_scope'],
    paper_scalar_targets=702,all_original_experiments_complete=False))
(out/'README.md').write_text('''# 保留完整 GAN 实验的空闲控制器接续及新增 TAB 实体证据

三个旧控制器都在等待资源，逐一检查 OS PID、创建时间、完整命令和全部后代，并在暂停后再次核验无数值子进程、无共享锁、待执行输出仍不存在。只停止了这些空闲元数据控制器，未停止任何数值工作。新的进程已接管原控制器锁和原状态目录，心跳持续更新，stderr 为空；实际节省内存见 actual_three_host_continuations.json。

TimeGAN 仍包含同一组 45 个原始任务，以及占用原规范种子的 Energy 恢复任务，不能计作新增 45 个模型。保留三个训练阶段、50,000 次嵌入/50,000 次监督/原始完整联合训练循环，以及 10×2,000 次判别评估、10×5,000 次预测评估。原始失败/部分尝试完整保留；Energy 种子 1104 已完成结果经原验证器复用，其余待执行任务和较强资源准入保留。

GT-GAN 保留四个原始 Stock 缺失率命令、全部 29 次完整生成、580 个训练过的评估器、116 个原生网络检查点、原始完整数据及随机状态。42/16 GiB commit 准入/紧急保护、18/4 GiB 显存保护、GPU UUID、CPU 线程数、原锁及所有命令均不变。

接续保留完整原 main 代码对象，只在新命名空间中重绑定已经完成的元数据交接回调。只有旧交接证明字节 SHA256 未变、并且新一轮实际安全交接已完成时才能复用。数值子进程仍使用原始冻结入口和参数；Windows 资源快照保持原结构布局、公式与错误处理，缓存 ctypes 类型以限制重复等待开销。新 24 项及既有交接/资源快照 18 项测试通过；包括真实 Windows 1,000 次轮询的类型缓存有界检查。这些测试不计作数据集实验成绩。

TranAD float64、原完整 SMAP、种子 1104 的 TAB 复核现已取得至少 21/54 个实体的成功守卫。此次逐一核对前 21 个实体的原配置、804 次指标调用覆盖、来源链和实际子进程退出 0 的守卫；完整保存新完成 e019、e020 的全部指标单元及 CSV，非有限值、异常、警告和原输出原样保留。21 个实体不是完整 SMAP 结果，不能进行完整数据集排名或宣称完整 TAB 已复现；完整 878 项/2057 标签视图、54 实体及后续汇总继续等待。

CFM、BRITS 种子 30、CrossAD 和 MOMENT 的实际数值工作进程也已核对。CFM 已完成的训练轮数和 BRITS 日志只是进度，MOMENT 是预训练模型的零样本完整时间戳评分，不是新增训练种子。原 45 篇论文、681 项方法/基线/消融、273 项数据任务、702 项论文标量目标及所有未完成原实验保持不变；本次不宣称全部完成。
''',encoding='utf-8',newline='\n')
snapshot(Path(__file__),'publication_source.py')
git=lambda path:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
manifest=dict(previous_root=git(root),previous_parent=git(root.parent/'flow-matching-research'),
              paths=originals+[p.relative_to(root).as_posix() for p in out.iterdir() if p.is_file()])
(root/'tmp/three_GAN_cold_TAB21_exact_publication_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(manifest['paths']),tests=tests,
    private_bytes_freed=sum(r['private_bytes_freed'] for r in hosts),TAB_verified_entities=21,
    TAB_calls=sum(e['metric_calls'] for e in ledger),outcomes=dict(outcomes))))
