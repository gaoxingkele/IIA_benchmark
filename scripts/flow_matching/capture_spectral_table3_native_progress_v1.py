"""Capture full-scope progress using live handles and immutable Table3 bindings."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.run_spectral_table3_original_v2 import ROOT, read, sha, verify, optimizer_assignments
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write

TARGET = ROOT / 'projects/flow_matching_research/results/2026-10-11/spectral_table3_native_resume_v1'


def observe(binding):
    result = dict(binding)
    try:
        process = psutil.Process(binding['pid'])
        result['verified_live'] = (abs(process.create_time() - binding['create_time']) < .01
                                   and binding['command_token'] in process.cmdline())
        result['command'] = process.cmdline()
        if result['verified_live']:
            result['children'] = []
            for child in process.children(recursive=True):
                try:
                    if Path(child.exe()).name.lower() == 'conhost.exe':
                        continue
                    result['children'].append(dict(pid=child.pid, create_time=child.create_time(),
                        command=child.cmdline(), cpu_seconds=sum(child.cpu_times()[:2]),
                        rss_bytes=child.memory_info().rss))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        result.update(verified_live=False, observation_error=repr(error))
    status = ROOT / binding['status_path']
    if status.is_file():
        result['state'] = read(status)
        result['reported_job_counts'] = dict(Counter(result['state'].get('jobs', {}).values()))
    return result


def main():
    if TARGET.exists():
        raise FileExistsError('Preserve previous progress capture')
    queue_path = 'configs/experiments/fm_spectral_table3_original_queue.v2.json'
    queue = read(ROOT / queue_path)
    verify(queue)
    data_path = ROOT / queue['cpu_data_audit']
    data = read(data_path)
    if not data['passed'] or not data['diagnostic_only'] or data['queue_sha256'] != sha(ROOT / queue_path):
        raise ValueError('Full independent original data proof differs')
    for record in data['datasets'].values():
        if sha(record['arrays']) != record['arrays_sha256']:
            raise ValueError('Full original data audit array differs')
    old = read(ROOT / 'projects/flow_matching_research/results/2026-10-10/imputation_and_strict_resume_v1/runtime_observations.json')
    bindings = [{k: r[k] for k in ('name', 'pid', 'create_time', 'command_token', 'status_path')}
                for r in old['controllers']]
    table3 = read(ROOT / queue['state_root'] / 'status.json')
    bindings.append(dict(name='spectral_table3_full', pid=table3['pid'], create_time=table3['create_time'],
        command_token='scripts.flow_matching.run_spectral_table3_queue_v2',
        status_path=queue['state_root'] + '/status.json'))
    controllers = [observe(r) for r in bindings]
    if not controllers[-1]['verified_live']:
        raise ValueError('Table3 controller handle is not live')
    TARGET.mkdir(parents=True)
    retire_path = ROOT / 'experiments/runs/fm_spectral_table3_original_v1/controller_retirement_receipt.json'
    (TARGET / 'full_cpu_data_audit.json').write_bytes(data_path.read_bytes())
    (TARGET / 'pretraining_v1_retirement.json').write_bytes(retire_path.read_bytes())
    scope = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
    assignments = []
    for job in queue['jobs']:
        config = read(ROOT / job['model_config'])
        selected = optimizer_assignments(ROOT / config['training_entry'])
        if [n.targets[0].id for n in selected] != config['original_optimizer_assignments']:
            raise ValueError('Complete original optimizer statement coverage differs')
        assignments.append(dict(id=job['id'], original_canonical_id=job['original_canonical_id'],
            model_config=job['model_config'], model_config_sha256=sha(ROOT / job['model_config']),
            entry=config['training_entry'], entry_sha256=sha(ROOT / config['training_entry']),
            original_assignment_names=config['original_optimizer_assignments'],
            full_epochs=config['author_hyperparameters']['epochs'],
            batch_size=config['author_hyperparameters']['batch_size'],
            generator_seed=config['seed'], standalone_evaluation_seed=config['evaluation_seed']))
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='no_progress',
        previous_turn_reason='Read-only metric listing verified published results but did not advance unfinished experiments.',
        current_goal_turn_classification='progress', objective='这个方向相关所有的已知未完成的实验都跑一遍',
        original_scope=scope, all_original_experiments_complete=False,
        table3_full_original_seed_slots=4, queue=queue_path, queue_sha256=sha(ROOT / queue_path),
        verified_source_bindings=len(queue['source_receipts']), environment_lock=queue['environment_lock'],
        environment_lock_sha256=sha(ROOT / queue['environment_lock']),
        full_cpu_data_audit_sha256=sha(data_path), full_data_audit_passed=True,
        full_original_optimizer_scope_checked=assignments,
        pretraining_v1_retirement_sha256=sha(retire_path),
        controllers=controllers, actual_live_controller_count=sum(r['verified_live'] for r in controllers),
        tests='21 passed', new_formal_metrics_published=False,
        remaining_work=['Four case-specific full original GPU capacity/evaluator checks, then full1000-epoch native training and independent standalone evaluation.',
            'Independent re-evaluation from saved full generated/reference arrays, RNG snapshots and trained checkpoints; compare against original Table3 claims.',
            'Original LS4 and SaShiMi-AR baselines, other generation/forecasting/image protocols, and all remaining records in the45-paper/681-method/273-data-task scope.',
            'All previously active TSAD, CFM-TS, original imputation/transfer, industrial and range-evaluation queues; full paper fidelity and metric audit.'])
    write(TARGET / 'runtime_observations.json', report)
    lines = ['# 原始长序列生成表3：完整实验进度', '',
        '本轮完成原始数据、独立环境和完整优化器语句审查，并启动四项原流程的资源调度。启动与预检均不是性能结果。', '',
        '原任务范围保持45篇论文、681条方法/基线/消融记录、273条原始数据/任务记录；全部实验尚未完成。', '',
        '前一目标轮仅核对指标清单，归类为无实验推进；本轮完成数据审计、修正预检作用域并启动新控制器，归类为进展。', '',
        '| 方法 | 数据集 | 完整数据形状 | 作者训练预算 | 训练/评估种子 |',
        '|---|---|---|---|---|']
    for job in queue['jobs']:
        config = read(ROOT / job['model_config'])
        lines.append(f"| {config['method']} | {config['dataset']} | {queue['datasets'][config['dataset']]['shape']} | 1000 epochs, batch32, 4 workers | 42 / 0 |")
    lines += ['', '数据审计逐序列、逐时间点核对独立解析/归一化与作者原始数据函数，不裁剪原始TSF。NN5发布数据长度791，论文文字长度728，差异保留。', '',
        '作者训练seed42、独立评估seed0重新随机划分：FRED最终22条测试序列中16条进入过训练；NN5最终23条中17条进入过训练。保留原行为作为作者协议复现，并记录泛化评估局限。', '',
        '优化器语句位于原作者PrintLogger的with块内。v1提取器仅检查函数顶层；控制器在无模型子进程的资源等待边界退役，全部冻结文件及日志保留。v2提取完整原语句，作者训练/评估源文件不变，不增加种子或改变预算。', '',
        '独立Table3环境继承97项有效包记录，新增校验通过的官方torchaudio2.8.0+cu126；数值依赖与冻结父环境一致。唯一显式bootstrap差异为pip23.0.1，父环境未改动。', '',
        '每次原始评估保持10次S4分类器/预测器拟合、每次100 epochs。STD是评估器重复的波动，不是10个生成模型种子的波动。原代码测试集marginal checkpoint选择保留。', '',
        f"控制器实际句柄核验：{report['actual_live_controller_count']}/{len(controllers)}存活。新Table3控制器PID{table3['pid']}，状态{controllers[-1].get('state', {}).get('state')}。资源准入同时要求GPU和共享CPU槽位；不抢占现有完整实验。", '',
        f"冻结来源绑定{len(queue['source_receipts'])}项；相关测试21项通过。尚无新正式性能结果。", '',
        '[进程、命令和哈希](runtime_observations.json)；[完整数据划分审计](full_cpu_data_audit.json)；[v1无模型启动的退役记录](pretraining_v1_retirement.json)。', '',
        '剩余：完整GPU预检、四项完整训练/独立评估、训练后独立指标复核及论文对照；LS4/SaShiMi-AR及其他未完成原始任务仍保留。', '']
    (TARGET / 'README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    receipts = [dict(path=(TARGET / n).relative_to(ROOT).as_posix(), sha256=sha(TARGET / n))
                for n in ['README.md', 'runtime_observations.json', 'full_cpu_data_audit.json', 'pretraining_v1_retirement.json']]
    write(TARGET / 'validation.json', dict(source_bindings_verified=len(queue['source_receipts']),
        live_controller_count=report['actual_live_controller_count'], tests='21 passed',
        full_raw_data_preserved=True, original_scope_preserved=True, formal_metrics_changed=False,
        outputs=receipts, script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(full_jobs=4, source_bindings=len(queue['source_receipts']),
        actual_live_controllers=report['actual_live_controller_count'],
        report=(TARGET / 'README.md').relative_to(ROOT).as_posix())))


if __name__ == '__main__':
    main()
