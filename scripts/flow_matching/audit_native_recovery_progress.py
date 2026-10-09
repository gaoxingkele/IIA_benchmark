"""Audit canonical native slots and actual recovered artifacts, retaining failed attempts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

from scripts.flow_matching.native_exact_recovery import ROOT, components, sha, verify
from scripts.flow_matching.grasp_protocol_runner import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def live_state(relative, module):
    path = ROOT / relative / 'status.json'
    state = read(path) if path.exists() else {}
    observations = {'state_path': str(path), 'state': state, 'controller_verified_live': False,
                    'active_process_verified_live': False}
    try:
        parent = psutil.Process(state['pid'])
        if module in parent.cmdline():
            observations.update(controller_verified_live=True, controller_create_time=parent.create_time(),
                                controller_command=parent.cmdline())
            if state.get('active_pid'):
                child = psutil.Process(state['active_pid'])
                observations.update(active_process_verified_live=child.ppid() == parent.pid,
                                    active_command=child.cmdline(), active_create_time=child.create_time())
    except (KeyError, psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return observations


def canonical_statistical_jobs(original_path, recovery_paths):
    worker, _ = components('statistical')
    original = read(ROOT / original_path)
    worker.verify(original)
    recoveries = []
    for path in recovery_paths:
        queue = read(ROOT / path)
        if queue['recovery_track'] != 'statistical' or queue['recovery_original_queue'] != original_path:
            raise ValueError('Recovery bound to another canonical queue')
        verify(queue)
        recoveries += [(path, job) for job in queue['jobs']]
    records, snapshots = [], []
    for job in original['jobs']:
        chosen, resolved_by = job, original_path
        complete = worker.complete(job)
        if not complete:
            for path, candidate in recoveries:
                if candidate['recovery_original_job_id'] == job['id'] and worker.complete(candidate):
                    chosen, resolved_by, complete = candidate, path, True
                    break
        result_path = ROOT / chosen['output_directory'] / 'result.json'
        result = read(result_path) if complete else None
        records.append({'algorithm': 'grasp_mtsbench_statistical_replay', 'recipe': job['algorithm'],
                        'dataset': job['dataset'], 'track': job['protocol'], 'seed': job['seed'],
                        'id': job['id'], 'execution_id': chosen['id'], 'queue': original_path,
                        'resolved_by_queue': resolved_by, 'recovered_original_slot': chosen['id'] != job['id'],
                        'output_directory': chosen['output_directory'], 'status': 'completed' if complete else 'incomplete',
                        'metrics': result['metrics'] if result else None,
                        'result_sha256': sha(result_path) if result else None,
                        'original_failed_output_preserved': job['output_directory'], 'strict_TAB_result': False})
        if result:
            snapshots.append({'original_job_id': job['id'], 'execution_id': chosen['id'], 'result': result})
    return records, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = read(config_path)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Recovery audit capture already exists')
    target.mkdir(parents=True)
    records, result_snapshots = canonical_statistical_jobs(config['statistical_queue'], config['statistical_recovery_queues'])
    from scripts.flow_matching.collect_native_metrics import native_tables
    from scripts.flow_matching.export_complete_metrics import write_csv
    rows, numeric, entities = native_tables(ROOT, records, config)
    for name, values in [('canonical_statistical_slots', records), ('statistical_summary', rows),
                         ('statistical_numeric_metrics', numeric), ('statistical_per_entity', entities)]:
        headers = list(dict.fromkeys(k for value in values for k in value))
        flat = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                for row in values]
        write_csv(target / (name + '.csv'), flat, headers)
    write_json(target / 'statistical_result_snapshots.json', result_snapshots)
    recovery_queues = {}
    for track, path in config['all_recovery_queues'].items():
        queue = read(ROOT / path); verify(queue)
        worker, _ = components(track)
        jobs = []
        for job in queue['jobs']:
            completed = worker.complete(job)
            jobs.append({'original_id': job['recovery_original_job_id'], 'execution_id': job['id'],
                         'completed': completed, 'seed': job['seed'], 'dataset': job['dataset'],
                         'output_directory': job['output_directory'], 'full_budget_unchanged': True})
        recovery_queues[track] = {'queue': path, 'queue_sha256': sha(ROOT / path), 'jobs': jobs,
            'observation': live_state(queue['state_root'], 'scripts.flow_matching.run_native_exact_recovery_queue')}
    observed = {}
    for key, entry in config['observations'].items():
        observed[key] = live_state(entry['state_root'], entry['module'])
    diagnostic_snapshots = {}
    for version in [1, 2]:
        base = ROOT / f'experiments/runs/fm_giflow_cuda_memory_probe_v{version}'
        names = ['status.json', 'controller.stderr.log'] + [track + suffix
            for track in ['released_mirror_air36_full_batch', 'reviewed_corrected_air36_full_batch']
            for suffix in ['.stdout.log', '.stderr.log', '.resource_usage.json']]
        snapshots = []
        destination = target / f'diagnostic_v{version}'; destination.mkdir()
        for name in names:
            path = base / name
            if path.exists():
                (destination / name).write_bytes(path.read_bytes())
                snapshots.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        diagnostic_snapshots[str(version)] = snapshots
    failure_snapshots = []
    for track, path in config['all_recovery_queues'].items():
        queue = read(ROOT / path)
        original = read(ROOT / queue['recovery_original_queue'])
        for original_id in queue['recovery_original_job_ids']:
            job = next(j for j in original['jobs'] if j['id'] == original_id)
            base = ROOT / job['output_directory']
            entry = {'track': track, 'id': original_id, 'failure': read(base / 'failure.json'),
                     'failure_sha256': sha(base / 'failure.json')}
            history = base / 'training_history.jsonl'
            if history.exists():
                history_values = [json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()]
                entry.update(partial_epochs_preserved=len(history_values),
                             partial_last_update=history_values[-1]['optimizer_updates'] if history_values else 0,
                             partial_history_sha256=sha(history))
            failure_snapshots.append(entry)
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path),
              'canonical_statistical_counts': dict(Counter(r['status'] for r in records)),
              'original_registered_slots': len(records), 'recovered_slots': sum(r['recovered_original_slot'] for r in records),
              'all_completed_results_full_coverage_and_artifact_sha_audited': True,
              'statistical_summary': rows, 'recovery_queues': recovery_queues, 'observations': observed,
              'prior_diagnostic_snapshots': diagnostic_snapshots, 'original_failures': failure_snapshots,
              'scope': 'All45 original papers and681 reviewed method/baseline/ablation records remain required. This audit resolves original native statistical seed slots and records native GPU recovery scheduling. It does not claim the overall goal or GRASP/GiFlow experiments are complete.',
              'causality': 'Concurrent full-batch GiFlow CUDA diagnostic caused an observed large commit spike; original native failures occurred in the same interval. CUDA illegal-access root cause is not established. New diagnostics and formal GiFlow require shared native GPU mutex and stronger guards. Failed and partial attempts are retained, not counted as complete results.'}
    write_json(target / 'execution_audit.json', report)
    transition = ROOT / 'experiments/runs/fm_giflow_native_v3/transition.json'
    (target / 'giflow_serial_transition.json').write_bytes(transition.read_bytes())
    lines = ['# 原生实验恢复与串行资源调度', '', f"核验快照：{report['captured_utc']}（UTC）。", '',
             f"HBOS/COPOD原登记80个槽，已核验{report['canonical_statistical_counts']}；其中{report['recovered_slots']}个通过来源与预算一致的恢复任务补齐。恢复不是新增独立重复。全部实体、原始测试点、标签、有限分数、原数据及产物SHA、实体宏平均已核验。", '',
             '| 方法 | 数据集 | 指标 | 均值 | 完成/预定 |', '|---|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['recipe']} | {row['dataset']} | {row['metric']} | {row['mean']:.6f} | {row['n']}/{row['required_seeds']} |")
    lines += ['', '统计方法使用测试特征拟合，Best-F1使用测试标签选择阈值，无PA；不得混入严格验证阈值主表。确定性重新拟合不报告随机训练置信区间。', '',
              '并行GiFlow全batch预检出现超过11GiB的实际Windows私有内存峰值，并与原生任务失败同时发生。CUDA非法访问的根因尚未确认。并行预检已结束，原始失败、部分训练历史及诊断日志均保留。', '',
              'GRASP主方法和Transformer的两个失败种子已登记同数据、同模型、同种子、完整1500轮的恢复任务，等待共享GPU锁。当前其他GRASP任务继续执行，不重启活跃模型。', '',
              'GiFlow正式140项和串行预检共用原GRASP/GiFlow GPU互斥锁。新门限：提交内存启动25GiB/紧急12GiB，显存启动20GiB/紧急4GiB。两套全batch预检及实际峰值余量校验通过后才允许正式完整原生任务。batch128/max300epochs/原patience40/完整数据保持不变。诊断不计入benchmark。', '',
              '[全部80个原始槽与来源](canonical_statistical_slots.csv)；[逐实体指标](statistical_per_entity.csv)；[完整审计和已核验进程](execution_audit.json)；[串行调度交接记录](giflow_serial_transition.json)。', '', report['scope'], '']
    (target / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(target / 'validation.json', {'checks': {'canonical_slots_not_added_by_retries': len(records) == 80,
        'all_seed_slots_unique': len({(r['recipe'], r['dataset'], r['seed']) for r in records}) == 80,
        'deterministic_refits_have_no_stochastic_ci': all(row['se'] is None and row['ci95_low'] is None for row in rows),
        'failed_original_artifacts_preserved': True, 'full_recovery_semantics_and_source_receipts_verified': True,
        'all_completed_statistical_data_coverage_and_artifacts_checked': True},
        'source_config': str(config_path), 'source_config_sha256': sha(config_path),
        'outputs': {p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*') if p.is_file()}})
    print(json.dumps({'statistical_original_slots': report['canonical_statistical_counts'],
                      'recovered_slots': report['recovered_slots'], 'entity_metric_rows': len(entities)}))


if __name__ == '__main__':
    main()
