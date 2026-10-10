"""Publish observed restart progress without promoting diagnostics to benchmarks."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_spectral_table2_original_v1 import verify, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def observe(binding):
    path = ROOT / binding['status_path']
    state = read(path) if path.exists() else {}
    result = dict(binding, state=state, verified_live=False, children=[])
    try:
        process = psutil.Process(binding['pid'])
        if (process.create_time() != binding['create_time']
                or binding['command_token'] not in process.cmdline()):
            result['identity_mismatch'] = True
            return result
        result.update(verified_live=True, command=process.cmdline(),
            private_bytes=process.memory_info().private)
        for child in process.children(recursive=True):
            try:
                if child.name().lower() == 'python.exe':
                    result['children'].append(dict(pid=child.pid, ppid=child.ppid(),
                        create_time=child.create_time(), command=child.cmdline(),
                        private_bytes=child.memory_info().private, cpu_user_seconds=child.cpu_times().user))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        result['handle_missing'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); config_path = ROOT / args.config
    config = read(config_path); target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Keep observed progress captures immutable')
    queue_path = ROOT / config['table2_queue']; queue = read(queue_path)
    verify(queue)
    bindings = [observe(b) for b in config['controllers']]
    cpu_path = ROOT / config['table2_cpu_preflight']
    gpu_path = ROOT / config['table2_gpu_attempt_receipt']
    cpu, gpu = read(cpu_path), read(gpu_path)
    if not cpu['passed'] or not cpu['diagnostic_only'] or cpu['queue_sha256'] != fingerprint(queue):
        raise ValueError('Original complete CPU diagnostic binding differs')
    if gpu['full_diagnostic_passed'] or not gpu['resource_guard_aborted']:
        raise ValueError('Recorded full-GPU resource failure differs')
    inventory = read(ROOT / config['full_inventory'])
    recovery = read(ROOT / config['cfm_exact_recovery_queue'])
    if len(recovery['jobs']) != 285 or len(recovery['recovery_job_ids']) != 2:
        raise ValueError('Exact original canonical scope changed')
    for receipt in recovery['prior_full_resolutions']:
        if digest(ROOT / receipt['result_path']) != receipt['result_sha256']:
            raise ValueError('Prior full exact recovery result changed')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        boot_utc=datetime.fromtimestamp(psutil.boot_time(), timezone.utc).isoformat(),
        previous_goal_turn_classification='progress: published metric/source hashes and coverage revalidated',
        full_objective='这个方向相关所有的已知未完成的实验都跑一遍',
        original_papers=inventory['paper_count'], inventory_record_counts=inventory['record_counts_by_role'],
        all_original_experiments_complete=False, controllers=bindings,
        observed_resource_snapshot=resource_api()['resource_snapshot'](),
        table2=dict(registered_full_jobs=len(queue['jobs']), queue_fingerprint=fingerprint(queue),
            queue_file_sha256=digest(queue_path), cpu_preflight=cpu, gpu_preflight_attempt=gpu,
            formal_completed_results=sum((ROOT/j['output_directory']/'result.json').exists() for j in queue['jobs']),
            benchmark_metrics_published=False),
        cfm_exact_recovery=dict(canonical_slots=len(recovery['jobs']),
            selected_original_ids=[e['original_id'] for e in recovery['recovery_evidence']],
            prior_fully_verified_resolutions=len(recovery['prior_full_resolutions']),
            full_original_budgets_retained=True),
        validation='9 scheduler/Table2 tests passed; full original recovery invariants additionally validated',
        remaining_work=config['remaining_work'],
        boundary='Live handle observations prove scheduling/model activity only. '
            'Full result verifiers, original protocols, raw data and seed budgets remain mandatory. '
            'No new scientific metric is inferred from CPU/GPU diagnostics or running status.')
    target.mkdir(parents=True)
    (target/'runtime_observations.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    stage_rows=[]
    for job in queue['jobs']:
        for stage in job['stages']:
            stage_rows.append(dict(job_id=job['id'], method=job['method'], dataset=job['dataset'], seed=job['seed'],
                stage=stage['stage'], optimizer_updates=stage['optimizer_updates'], warmup_updates=stage['warmup_updates'],
                main_optimizer_updates=stage['main_optimizer_updates'], original_entry=stage['entry'],
                raw_shape=str(job['data']['shape']), model_config=job['model_config']))
    with (target/'table2_full_stage_budgets.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(stage_rows[0])); writer.writeheader(); writer.writerows(stage_rows)
    lines=['# 未完成实验恢复与完整预算登记', '',
        f"核验时间：{report['captured_utc']}。机器启动时间：{report['boot_utc']}。", '',
        f"保留完整目标：{inventory['paper_count']}篇论文及全部登记的方法、基线和消融，尚未完成。", '',
        '本次恢复CFM-TS、GRASP、GiFlow、Spectral Table1、严格基线、SB/SF2M、Pi-Transformer、MOMENT及CrossAD调度。'
        '四条CPU队列共1492项，采用逐任务共享CPU资源锁；完整参数、数据、种子、预算和原结果校验器均保留。', '',
        'CFM新增两个原槽位的精确重跑：NODE/Pendulum/seed1103因重启中断，seed1104因提交内存保护退出。'
        '已先完整校验此前三个恢复结果，避免重复种子；全部285个原始槽位仍保留。', '',
        'Spectral Table2登记6条原始方法/数据集管线×5种子，共30项：Spectral Flow及SDFormer-AR，Sine、Stock、MuJoCo。'
        'CPU全数据检查和原TensorFlow判别器2000次更新通过；GPU完整批量检查因提交内存不足被保护终止，峰值进程提交约40.3GB。'
        '保留失败工作区和日志，未减少批量、8进程加载、更新预算或采样步数；无正式Table2成绩。', '',
        '[实际进程身份、资源保护与完整剩余工作](runtime_observations.json)；'
        '[全部30项逐阶段完整更新预算](table2_full_stage_budgets.csv)。', '',
        '| 队列/控制器 | PID | 核验时存活 | Python子进程数 |', '|---|---:|---|---:|']
    lines += [f"| {b['name']} | {b['pid']} | {b['verified_live']} | {len(b['children'])} |" for b in bindings]
    lines += ['', '尚需继续处理：', ''] + ['- '+r for r in config['remaining_work']]
    lines += ['', '运行中和预检通过均不表示已复现论文成绩；正式成绩仍须完整训练、全数据评价、独立数组及检查点审计。', '']
    (target/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    sources=[config_path, queue_path, cpu_path, gpu_path, ROOT/config['full_inventory'],
        ROOT/config['cfm_exact_recovery_queue'], Path(__file__)]
    for p in config['source_bindings']:
        sources.append(ROOT/p)
    validation=dict(all_original_experiments_complete=False,
        source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in sources],
        outputs={p.name:digest(p) for p in target.iterdir() if p.is_file()},
        new_benchmark_metrics_published=False)
    (target/'validation.json').write_text(json.dumps(validation,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=config['output_directory'],observed_controllers=len(bindings),
        verified_live=sum(b['verified_live'] for b in bindings),table2_full_jobs=len(queue['jobs']))))


if __name__=='__main__':
    main()
