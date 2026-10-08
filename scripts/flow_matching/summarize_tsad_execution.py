"""Audit current execution artifacts and retain the entire known unfinished scope."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import psutil
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import complete_result


def intervals(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    se = std / len(values) ** .5 if std is not None else None
    margin = float(t.ppf(.975, len(values) - 1)) * se if se is not None else None
    return {'n': len(values), 'mean': mean, 'std': std, 'se': se,
            'ci95_low': mean - margin if margin is not None else None,
            'ci95_high': mean + margin if margin is not None else None}


def light_worker_matches(process, queue_path, root):
    command = process.cmdline()
    if 'scripts.flow_matching.run_light_controller' not in command:
        return False
    runtime_path = root / command[command.index('--runtime-config') + 1]
    name = command[command.index('--controller') + 1]
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    binding = next(b for b in runtime['controllers'] if b['name'] == name)
    if binding['queue_path'] != queue_path or binding['queue_sha256'] != sha(root / queue_path):
        return False
    return all(sha(root / source['path']) == source['sha256'] for source in runtime['source_receipts'])


def _one_range_snapshot(root, project, path):
    if not path:
        return {'completed_evaluations': 0, 'records': [], 'seed_aggregates': []}
    config = json.loads((root / path).read_text(encoding='utf-8'))
    for source in config['source_receipts']:
        if sha(root / source['path']) != source['sha256']:
            raise ValueError('Frozen range metric source changed')
    base = root / config['output_root']
    records, groups = [], defaultdict(lambda: defaultdict(list))
    for result_path in sorted((base / 'jobs').glob('*/evaluation.json')):
        receipt = result_path.parent / 'evaluation.sha256'
        if not receipt.exists():
            continue  # Atomic producer may be between the JSON and checksum writes.
        checksum = sha(result_path)
        if receipt.read_text(encoding='ascii').strip() != checksum:
            raise ValueError('Completed TAB metric artifact changed')
        record = json.loads(result_path.read_text(encoding='utf-8'))
        if sha(root / record['model_result_path']) != record['model_result_sha256']:
            raise ValueError('TAB metric source model result changed')
        entry = {key: record[key] for key in ('id', 'dataset', 'model_config', 'seed', 'VUS_entity_macro',
                                             'strict_affiliation_entity_macro', 'tab_ratio_affiliation_entity_macro',
                                             'aggregation', 'boundary')}
        entry.update(evaluation_path=result_path.relative_to(root).as_posix(), evaluation_sha256=checksum)
        records.append(entry)
        metrics = {**record['VUS_entity_macro'], **record['strict_affiliation_entity_macro']['1']}
        for metric, value in metrics.items():
            if value['mean_over_defined_entities'] is not None:
                groups[(record['model_config'], record['dataset'])][metric].append(value['mean_over_defined_entities'])
    aggregates = [{'model_config': model, 'dataset': dataset,
                   'metrics': {metric: intervals(values) for metric, values in metrics.items()},
                   'boundary': 'Entity macro only over defined reference outputs; per-run undefined counts retained. Different metrics/protocols are not interchangeable.'}
                  for (model, dataset), metrics in groups.items()]
    state_path = base / 'status.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    process_live = False
    try:
        process = psutil.Process(state.get('pid', -1))
        process_live = light_worker_matches(process, path, root) or any(any(worker in argument for worker in (
            'run_tab_ranges_when_ready.py', 'run_tab_ranges_for_config.py',
            'scripts.flow_matching.run_tab_ranges_sparse_queue')) for argument in process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return {'completed_evaluations': len(records), 'records': records, 'seed_aggregates': aggregates,
            'process_observation': {'pid': state.get('pid'), 'verified_live': process_live, 'status': state.get('status'), 'active_job': state.get('active_job')},
            'full_psm_reference_differential': project.get('tab_range_differential_report'),
            'boundary': 'Pinned TAB metric evaluators, not full TAB model training harness.'}


def range_snapshot(root, project):
    paths = [project['tab_range_execution_config']] if project.get('tab_range_execution_config') else []
    paths += project.get('additional_tab_range_execution_configs', [])
    snapshots = [_one_range_snapshot(root, project, path) for path in paths]
    records = [r for snapshot in snapshots for r in snapshot['records']]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate range evaluation runs across configurations')
    return {'completed_evaluations': len(records), 'records': records,
            'seed_aggregates': [g for snapshot in snapshots for g in snapshot['seed_aggregates']],
            'process_observation': snapshots[0]['process_observation'] if snapshots else {},
            'additional_process_observations': [s['process_observation'] for s in snapshots[1:]],
            'full_psm_reference_differential': project.get('tab_range_differential_report'),
            'boundary': 'Pinned TAB metric evaluators, not full TAB model training harness.'}


def author_pipeline_snapshot(root, project):
    from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts
    snapshots = []
    for name, queue_path in project.get('author_execution_queues', {}).items():
        queue = json.loads((root / queue_path).read_text(encoding='utf-8'))
        base = root / queue['settings']['output_root']
        state = json.loads((base / 'status.json').read_text(encoding='utf-8')) if (base / 'status.json').exists() else {}
        def matches(pid, fragment):
            if not pid or pid <= 0:
                return False
            try:
                process = psutil.Process(pid)
                return any(fragment in a for a in process.cmdline()) or light_worker_matches(process, queue_path, root)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        worker_live = matches(state.get('pid'), queue.get('worker_script', 'run_maelnet_author_queue.py'))
        child_live = matches(state.get('active_pid'), queue.get('child_script', 'run_anomaly.py'))
        jobs = []
        for job in queue['jobs']:
            output = root / job['output_directory']
            result = output / 'result.json'
            row = {key: job[key] for key in ['id', 'dataset', 'seed', 'author_recipe', 'output_directory']}
            if result.exists():
                receipt = json.loads(result.read_text(encoding='utf-8'))
                if receipt['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest():
                    raise ValueError('Author result belongs to another job')
                verify_artifacts(receipt, root)
                row.update(status='completed', result_path=result.relative_to(root).as_posix(), result_sha256=sha(result), metrics=receipt['metrics'])
            else:
                row['status'] = ('running' if worker_live and child_live and state.get('active_job') == job['id'] else
                                 'failed_or_partial_preserved' if (output / 'failure.json').exists() else
                                 'preparing_or_partial' if output.exists() else 'pending')
            row['stage_receipts'] = [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p),
                                      'status': json.loads(p.read_text(encoding='utf-8'))['status']}
                                     for p in sorted((output / 'stages').glob('*/receipt.json'))]
            jobs.append(row)
        snapshots.append({'name': name, 'queue': queue_path, 'queue_sha256': sha(root / queue_path),
                          'registered_jobs': len(jobs), 'registered_stages': sum(len(j['stages']) for j in queue['jobs']),
                          'counts': dict(Counter(j['status'] for j in jobs)), 'jobs': jobs,
                          'process_observation': {'pid': state.get('pid'), 'verified_live': worker_live,
                                                  'active_pid': state.get('active_pid'), 'active_process_verified_live': child_live,
                                                  'active_job': state.get('active_job'), 'active_stage': state.get('active_stage')},
                          'boundary': queue['boundary'], 'strict_TAB_result': False})
    return snapshots


def snapshot(root):
    settings = json.loads((root / 'configs/experiments/fm_tsad_execution.v1.json').read_text(encoding='utf-8'))
    base = root / settings['output_root']
    completed, jobs, live = [], [], []
    aggregates = defaultdict(lambda: defaultdict(list))
    project = json.loads((root / 'configs/projects/flow_matching_research.v1.json').read_text(encoding='utf-8'))
    queue_paths = list(settings['queue_paths'].items()) + list(project.get('additional_execution_queues', {}).items())
    queue_paths += [('industrial_' + lane, path) for lane, path in project.get('industrial_execution_queues', {}).items()]
    for lane, queue_path in queue_paths:
        queue = json.loads((root / queue_path).read_text(encoding='utf-8'))
        state_path = root / queue['state_root'] / (queue['lane'] + '_status.json')
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        process = None
        try:
            process = psutil.Process(state.get('pid', -1))
            worker = queue.get('worker_script', 'run_industrial_tsad_queue.py' if lane.startswith('industrial_') else 'run_tsad_queue.py')
            if not any(worker in a for a in process.cmdline()):
                process = None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process = None
        live.append({'lane': lane, 'state': state.get('status'), 'queue_pid': state.get('pid'),
                     'queue_process_verified_live': process is not None,
                     'active_job': state.get('active_job'), 'active_pid': state.get('active_pid'),
                     'active_process_verified_live': bool(state.get('active_pid') and psutil.pid_exists(state['active_pid']))})
        for job in queue['jobs']:
            output = root / job['output_directory']
            done = complete_result(root, job)
            status = 'completed' if done else 'running' if process and state.get('active_job') == job['id'] and psutil.pid_exists(state.get('active_pid', -1)) else 'partial_or_failed' if output.exists() else 'pending'
            jobs.append({'id': job['id'], 'dataset': job['dataset'], 'model_config': job['model_config'], 'seed': job['seed'], 'lane': lane, 'status': status})
            if done:
                result_path = output / 'result.json'
                result = json.loads(result_path.read_text(encoding='utf-8'))
                strict = result['metrics']['strict'][str(settings['evaluation']['primary_false_alarm_percent'])]
                record = {'id': job['id'], 'dataset': job['dataset'], 'model_config': job['model_config'], 'seed': job['seed'],
                          'result_path': result_path.relative_to(root).as_posix(), 'result_sha256': sha(result_path),
                          'scores_sha256': result['scores_sha256'], 'checkpoint_sha256': result['checkpoint_sha256'],
                          'metrics': strict['micro'], 'block_or_cluster_f1_ci95': strict['f1_ci95'],
                          'tab_core_diagnostic': result['metrics']['tab_core_diagnostic'], 'paper_equivalence': False,
                          'training_seconds': result['training_seconds'], 'score_seconds': result['score_seconds']}
                completed.append(record)
                for metric, value in strict['micro'].items():
                    if value is not None:
                        aggregates[(job['model_config'], job['dataset'])][metric].append(value)
    required = Counter((j['model_config'], j['dataset']) for j in jobs)
    grouped = [{'model_config': model, 'dataset': dataset, 'required_seeds': required[(model, dataset)],
                'metrics': {key: intervals(values) for key, values in metrics.items()},
                'boundary': 'Repeated trained-model seed t intervals describe initialization variability; frozen zero-shot pretrained runs have no independent initialization repeats and no seed interval. Not a paired algorithm equivalence test.'}
               for (model, dataset), metrics in aggregates.items()]
    # Old flat-harness records are evidence; never silently adopted as new strict runs.
    history = []
    for path in sorted((root / 'experiments/runs/mtsad_reproduction').rglob('*__seed*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if not all(key in record for key in ('model', 'dataset', 'seed', 'protocols')):
            continue
        history.append({'path': path.relative_to(root).as_posix(), 'sha256': sha(path),
                        'model': record['model'], 'dataset': record['dataset'], 'seed': record['seed'],
                        'parameters': record.get('parameters'), 'protocols': record['protocols'],
                        'saved_scores_present': path.with_name(path.stem + '__scores.npz').exists(),
                        'status': 'historical_payload_protocol_record_not_new_strict_result',
                        'boundary': 'May repeat the same seed under different settings; raw hashes/grouped coverage/checkpoints not proven by this record.'})
    ara = json.loads((root / 'configs/reproducibility/flow_matching_ara.v1.json').read_text(encoding='utf-8'))
    inventory = json.loads((root / 'configs/reproducibility/fm_paper_method_inventory.v1.json').read_text(encoding='utf-8'))
    papers = [{'paper_id': paper['id'], 'task': paper['task'], 'original_datasets': paper['original_datasets'],
               'model_configs': paper['model_configs'], 'remaining_original_reproduction_gap': paper['reproduction_gap'],
               'status': 'all_original_experiments_not_proven_complete',
               'known_method_records': sum(r['paper_id'] == paper['id'] for r in inventory['records'])} for paper in ara['papers']]
    ranges = range_snapshot(root, project)
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'objective': 'Run every known unfinished experiment in this research direction',
              'goal_achieved': False, 'new_job_counts': dict(Counter(j['status'] for j in jobs)),
              'new_registered_attempts_including_numerical_retries': len(jobs),
              'base_registered_experiments': len(settings['model_configs']) * len(settings['datasets']) * len(settings['seeds']),
              'new_method_configs': len(settings['model_configs']),
              'strict_baseline_registered_jobs': sum(j['lane'] == 'strict_baselines' for j in jobs),
              'iterative_reflow_and_epoch_control_jobs': sum(j['lane'] == 'iterative_reflow' for j in jobs),
              'industrial_registered_jobs': sum(j['lane'].startswith('industrial_') for j in jobs),
              'industrial_job_counts': dict(Counter(j['status'] for j in jobs if j['lane'].startswith('industrial_'))),
              'tab_range_evaluations': ranges,
              'author_pipeline_experiments': author_pipeline_snapshot(root, project),
              'process_observations': live, 'completed_new_runs': completed, 'seed_aggregates': grouped,
              'new_jobs': jobs, 'legacy_mtsad_records': history, 'legacy_record_count': len(history),
              'original_paper_scope': papers, 'known_method_inventory_records': len(inventory['records']),
              'other_remaining_obligations': ['Existing 540-job imputation/transfer queues',
                  'Author MaelNet full RL, Pi journal equivalence, CrossAD, MOMENT pretrained adapters and source-only baselines',
                  'Exact original-paper datasets/settings and all inspected ablation axes across all 45 ARA artifacts',
                  'GiFlow/forecasting/generation/continuous-time/image/single-cell and tabular original experiments',
                  'Complete all registered grouped TEP/SKAB/PRONTO detection jobs and metrics; classic TEP is not multimode equivalence; PRONTO normal selection is label-assisted',
                  'Finish VUS/Affiliation for all remaining saved-score jobs and full pinned TAB author training-harness alignment',
                  'Finish iterative reflow and equal-total-epoch controls; original image/transfer protocols and one-step distillation remain pending',
                  'Hyperparameter/fidelity gaps and material availability recorded in each ARA source'],
              'boundary': 'New queues are concrete progress, not a redefinition of completion. Historical results were absent from the 2026-10-08 FM-only table and are retained here without upgrading their protocol or fidelity.'}
    target = root / 'docs/reports/fm_full_execution_progress_2026-10-09.json'
    write_json(target, report)
    delivery = root / 'projects/flow_matching_research/results/2026-10-09'
    delivery.mkdir(parents=True, exist_ok=True)
    write_json(delivery / 'execution_snapshot.json', report)
    data_manifest = root / settings['prepared_root'] / 'manifest.json'
    (delivery / 'data_manifest.json').write_bytes(data_manifest.read_bytes())
    legacy_rows = []
    for run in history:
        for protocol, metrics in run['protocols'].items():
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    legacy_rows.append({'model': run['model'], 'dataset': run['dataset'], 'seed': run['seed'],
                                        'protocol': protocol, 'metric': name, 'value': value,
                                        'source_path': run['path'], 'source_sha256': run['sha256'],
                                        'status': run['status']})
    with (delivery / 'historical_mtsad_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'dataset', 'seed', 'protocol', 'metric', 'value', 'source_path', 'source_sha256', 'status'])
        writer.writeheader()
        writer.writerows(legacy_rows)
    lines = ['# 未完成实验执行进度', '', f"快照时间：{report['captured_utc']}。目标仍为全部已知未完成实验；尚未完成。", '',
             '42 个已可训练窗口方法/消融配置 × 7 个完整本地数据集 × 5 个种子 = 1,470 个基础实验；另有 105 个 SB/SF2M 数值修复重跑任务。',
             f"另加入 {report['strict_baseline_registered_jobs']} 个窗口基线任务：6 个已有方法及 USAD 有符号损失对照，使用相同的完整输入数组和验证段。",
             f"另加入 {report['iterative_reflow_and_epoch_control_jobs']} 个两/三阶段 reflow 及40/60轮总训练轮数对照；保存每阶段教师、端点配对与实际轮数。对照不抵消reflow额外的ODE生成开销。",
             '独立作者/镜像完整流程：' + '；'.join(f"{a['name']} {a['registered_jobs']}个任务、{a['registered_stages']}个阶段，{a['counts']}" for a in report['author_pipeline_experiments']) + '。原协议单列，不计入严格无PA成绩。',
             f"另有工业异常检测任务 {report['industrial_registered_jobs']} 项，状态 {report['industrial_job_counts']}。TEP整运行、SKAB整实验和PRONTO整日角色隔离；不是插补结果。",
             '保留本地模型配置的训练轮数与容量；非重叠训练窗口和尾部覆盖规则已冻结，这不证明匹配原论文的更新次数、数据划分或架构。',
             '关键训练预算差异：非重叠窗口比原作者 stride=1 的重叠训练少很多梯度更新。相同 epoch 数不能证明训练预算等同；原 stride=1 作者轨仍须独立完成，不能用这里的低分断言原方法无效。',
             f"当前任务记录：{dict(Counter(j['status'] for j in jobs))}。包含数值重试，不能解释为独立方法数或全部基础实验完成数。", '',
             'CPU 与数值修复进程已核验存活；GPU 续跑等待现有插补链完成并取得共同锁。进程/状态是此快照的观察，后续以当前操作系统进程及结果哈希为准。', '',
             '## 已完成的本地严格协议结果', '',
             '验证段校准 1% 报警分位数，无 PA；每个原测试时间点一个有限分数。以下均为本地重建/适配，非作者等效认证。', '',
             '| 模型配置 | 数据集 | 已完成/预定种子 | 点级 F1 均值 | AUROC 均值 | AP 均值 |',
             '|---|---|---|---:|---:|---:|']
    for group in grouped:
        metrics = group['metrics']
        fmt = lambda name: f"{metrics[name]['mean']:.6f}" if name in metrics else '—'
        lines.append(f"| {Path(group['model_config']).stem} | {group['dataset']} | {metrics['f1']['n']}/{group['required_seeds']} | {fmt('f1')} | {fmt('auroc')} | {fmt('average_precision')} |")
    lines += ['', '逐种子结果、标准误/种子区间、时间块或实体区间、检查点及分数哈希均保存在 [execution_snapshot.json](execution_snapshot.json)。单种子块区间不替代跨算法配对检验。', '',
              '独立 CFM 与单阶段 Rectified 在 sigma=0 时使用同一条直线路径，数值相同是预期行为。新增两/三阶段 reflow 真正生成前一流的端点配对，并在配对上重新训练；其本地异常分数仍不是原图像实验等价证明。', '',
              '## 已补算的固定版本 TAB 范围指标', '',
              f"已核验 {ranges['completed_evaluations']} 个完整分数文件的 VUS 与 Affiliation。VUS保留原250阈值、全部整数缓冲长度、inclusive ties与积分公式；完整PSM87841点与原代码执行差异为约1e-16。", '',
              '| 模型配置 | 数据集 | VUS 已完成种子 | VUS ROC 均值 | VUS PR 均值 | 严格阈值 Affiliation F 均值 |',
              '|---|---|---:|---:|---:|---:|']
    for group in ranges['seed_aggregates']:
        metrics = group['metrics']
        fmt = lambda name: f"{metrics[name]['mean']:.6f}" if name in metrics else '—'
        count = metrics.get('VUS_ROC', metrics.get('affiliation_f', {})).get('n', 0)
        lines.append(f"| {Path(group['model_config']).stem} | {group['dataset']} | {count} | {fmt('VUS_ROC')} | {fmt('VUS_PR')} | {fmt('affiliation_f')} |")
    lines += ['', '每个实体独立评价；macro仅平均原参考函数有定义的结果，未定义实体/种子数保留，不填零。Affiliation F、点级 F1 和 PA-F1 是不同指标，不能直接混比。VUS缓冲按测试标签事件长度产生，是已披露的评价依赖；严格阈值仍只来自验证段。完整TAB训练流程未等效认证。', '',
              '## 历史记录补充与范围更正', '',
              f"先前 2026-10-08 表只覆盖 FM 注册队列，未纳入旧 mtsad_reproduction 目录的 {len(history)} 份运行记录。TimesNet、Anomaly Transformer、DCdetector 等历史数值确实存在，因此不能据前表声称整个项目没有异常检测成绩。", '',
              '旧记录可能重复种子、预算或协议；缺少新严格协议要求的冻结原始数据、完整实体/时间覆盖或检查点链，不能直接晋升为新队列已完成项。全部历史数值及协议见 [historical_mtsad_metrics.csv](historical_mtsad_metrics.csv)。未改写历史文件。', '',
              '## 未完成范围仍然保留', '',
              '45 篇 ARA 的原始数据集、681 条已审方法/消融记录以及每篇复现缺口，完整保留于 execution_snapshot.json。启动 1,470 个本地任务并未替代这些义务。', '',
              '- 原有 540 个插补/迁移任务继续执行；原数据和活跃队列未重启。',
              '- 原作者 MaelNet RL、Pi 期刊版本等效、CrossAD、MOMENT 权重及其他 source-only 适配器仍需完善。',
              '- GiFlow、预测、生成、连续时间、图像、单细胞、表格原始实验及工业异常检测迁移仍未全部完成。',
              '- TAB 核心指标/比例诊断单列：合并训练/测试分数校准且按测试标签选最佳比例。剩余 VUS/Affiliation 和完整 TAB 原作者训练流程继续执行。',
              '- 原 Sinkhorn 不收敛日志和部分产物保留；修正版只改求解器续接与迭代上限，最终正则、边缘容差和训练轮数不变。修正版已有实际完整 PSM 成功结果，其余种子/数据集继续核验。', '',
              '数据身份、通道、实体边界、训练/验证隔离、缺失修复和训练段缩放统计见 [data_manifest.json](data_manifest.json)。SMAP_P-7 仅提供训练文件，明确登记后不纳入 51 个有测试文件的实体评价。', '']
    (delivery / 'README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return report


def main():
    report = snapshot(ROOT)
    print(json.dumps({'counts': report['new_job_counts'], 'legacy_records': report['legacy_record_count'],
                      'live': report['process_observations'], 'goal_achieved': False}, indent=2))


if __name__ == '__main__':
    main()
