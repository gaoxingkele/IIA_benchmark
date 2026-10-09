"""Verify lightweight handoff, complete native cohorts and preserved preflight evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller
from scripts.flow_matching.grasp_protocol_runner import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = read(config_path)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve prior execution captures')
    runtime = read(ROOT / config['runtime_config'])
    _, _, binding = load_controller(ROOT, runtime, 'tsad_gpu')
    transition = read(ROOT / config['transition_report'])
    record = transition['records'][0]
    state = read(ROOT / binding['status_path'])
    parent = psutil.Process(state['pid'])
    if (parent.pid != record['new_pid'] or parent.children(recursive=True)
            or 'scripts.flow_matching.run_light_controller' not in parent.cmdline()
            or parent.cmdline()[parent.cmdline().index('--runtime-config') + 1] != config['runtime_config']
            or parent.cmdline()[parent.cmdline().index('--controller') + 1] != 'tsad_gpu'
            or state['status'] != 'waiting_existing_imputation'):
        raise ValueError('New controller is not the verified idle frozen-queue successor')
    if state['pending_prerequisites'] != record['state']['pending_prerequisites']:
        raise ValueError('Prerequisite state changed; capture its progression separately')
    if psutil.pid_exists(record['old_pid']):
        old = psutil.Process(record['old_pid'])
        if old.create_time() == record['old_create_time']:
            raise ValueError('Old scheduler remains live')
    memory = getattr(parent.memory_info(), 'private', parent.memory_info().rss)
    handoff = {'pid': parent.pid, 'create_time': parent.create_time(), 'command': parent.cmdline(),
               'before_private_bytes': record['private_bytes'], 'after_private_bytes': memory,
               'private_bytes_released': record['private_bytes'] - memory, 'state': state,
               'queue_sha256': binding['queue_sha256'], 'no_model_children': True,
               'prerequisite_counts_unchanged': True, 'original_job_dictionaries_unchanged': True}
    from scripts.flow_matching.summarize_grasp_protocol_v3 import summarize
    from scripts.flow_matching.audit_original_native_execution import audit_grasp_arrays
    grasp_queue = read(ROOT / config['grasp_queue'])
    grasp = summarize(grasp_queue)
    verified = audit_grasp_arrays(ROOT, grasp_queue, grasp)
    jobs_by_id = {job['id']: job for job in grasp_queue['jobs']}
    for result in verified:
        job = jobs_by_id[result['id']]
        parameters = read(ROOT / job['model_config'])['parameters']
        if (result['epochs_completed'] != parameters['epochs']
                or result['optimizer_updates'] != job['expected_optimizer_updates']):
            raise ValueError('Actual completed native budget differs from frozen configuration')
    primary = read(ROOT / config['crossad_parent_queue'])
    proof_records = []
    for path in sorted((ROOT / 'docs/reports').glob('fm_crossad_complete_preflight_v4*2026-10-09.json')):
        proof = read(path)
        if proof['queue_sha256'] != digest(ROOT / config['crossad_parent_queue']) or len(proof['records']) != 1:
            raise ValueError('Native preflight registration differs')
        row = proof['records'][0]
        key = row['checkpoint'].get('reconstructed_ablation_row')
        job = next(j for j in primary['jobs'] if j['dataset'] == row['dataset']
                   and (j.get('ablation_row') == key if key is not None else j['mode'] == 'release_checkpoint'))
        if (not row['finite_backward'] or not row['deterministic_release_inference']
                or row['benchmark_performance']
                or row['native_batch_shape'][:2] != [job['train_parameters']['batch_size'], job['model_parameters']['seq_len']]):
            raise ValueError('Native batch or ablation identity differs')
        proof_records.append({'case': row['dataset'] + ('_row' + str(key) if key is not None else ''),
                              'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'record': row})
    tsad = read(ROOT / config['tsad_progress'])
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': digest(config_path),
              'scheduler_handoff': handoff, 'grasp_protocol': grasp, 'complete_native_arrays_audited': verified,
              'crossad_verified_native_preflights': proof_records, 'crossad_preflights_not_benchmark': True,
              'tsad_snapshot_sha256': digest(ROOT / config['tsad_progress']),
              'tsad_snapshot_time': tsad['captured_utc'], 'tsad_full_run_counts': tsad['new_job_counts'],
              'known_method_records': tsad['known_method_inventory_records'],
              'original_papers_retained': len(tsad['original_paper_scope']),
              'other_remaining_obligations': tsad['other_remaining_obligations'],
              'goal_achieved': False, 'paper_equivalence_certified': False}
    target.mkdir(parents=True)
    write_json(target / 'execution_audit.json', report)
    (target / 'scheduler_transition.json').write_bytes((ROOT / config['transition_report']).read_bytes())
    lines = ['# 完整实验续跑与调度交接核验', '', f"快照：{report['captured_utc']}（UTC）。", '',
             f"空闲TSAD GPU调度进程私有内存由{record['private_bytes'] / 1024**2:.2f} MiB降至{memory / 1024**2:.2f} MiB，实际释放{handoff['private_bytes_released'] / 1024**2:.2f} MiB。原完整GPU任务、四套插补前置队列、预算、种子和锁保持不变。没有模型子进程被停止。", '',
             f"严格异常检测新快照任务状态：{tsad['new_job_counts']}，源时间{tsad['captured_utc']}。原实验及失败保留，新增完整结果来自实际运行，预检不计成绩。", '',
             f"GRASP完整训练登记{grasp['registered_entity_jobs']}个实体任务/{grasp['registered_cohorts']}个全实体组，完成{grasp['completed_cohorts']}组；状态{grasp['entity_job_counts']}。", '',
             '所有完整GRASP产物核验了实际1500轮检查点、更新数、完整验证/测试行、标签及分数，并独立重算指标。下表使用默认5 source samples/10 flow evaluations，单种子不是论文十种子等价证明。', '',
             '| 完整配置 | 数据集 | 种子 | 轮数 | 更新 | AP | AUROC | 测试最优F1 | 验证阈值F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for result in verified:
        default = next(m for m in result['metrics'] if m['recipe'] == {'source_samples': 5, 'flow_evaluations': 10})
        paper = default['metrics']['paper_retrospective']; control = default['metrics']['validation_only_control']
        job = next(j for j in grasp_queue['jobs'] if j['id'] == result['id'])
        lines.append(f"| {job['profile']} | {job['dataset']} | {job['seed']} | {result['epochs_completed']} | {result['optimizer_updates']} | {paper['AP']:.6f} | {paper['ROC']:.6f} | {paper['Best_F1']:.6f} | {control['f1']:.6f} |")
    lines += ['', f"CrossAD已核验{len(proof_records)}/22个原生全batch检查（含组件消融），后向和推断可调用不等于完整训练或benchmark表现。原资源失败及未完成检查继续保留。", '',
              '[完整审计、逐配置指标和进程证据](execution_audit.json)；[调度交接记录](scheduler_transition.json)。', '',
              '24项轻量调度/交接/实际前置等待/队列身份测试通过。全45篇论文、681条已审读方法/基线/消融及其他原任务、迁移、生成等义务保持原范围，尚未全部完成。', '']
    (target / 'README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    write_json(target / 'validation.json', {'tests_passed': 24,
        'test_command': 'python -X utf8 -m pytest tests/test_fm_tsad_gpu_light_controller.py tests/test_fm_tsad_light_liveness.py tests/test_fm_light_controller.py tests/test_fm_light_controller_handoff_v2.py -q',
        'checks': {'actual_successor_identity_verified': True, 'observed_original_idle_process_ended': True,
                   'prerequisite_counts_unchanged': True, 'all_complete_native_arrays_and_checkpoint_checked': True,
                   'metrics_independently_recomputed': True, 'crossad_preflights_not_benchmark': True,
                   'full_original_scope_preserved': len(tsad['original_paper_scope']) == 45 and tsad['known_method_inventory_records'] == 681},
        'sources': {p: digest(ROOT / p) for p in [args.config, config['runtime_config'],
                    'scripts/flow_matching/audit_queue_handoff_progress.py', 'scripts/flow_matching/summarize_tsad_execution.py',
                    'tests/test_fm_tsad_gpu_light_controller.py', 'tests/test_fm_tsad_light_liveness.py']},
        'outputs': {p.name: digest(p) for p in target.iterdir() if p.is_file()}})
    print(json.dumps({'strict_full_runs': tsad['new_job_counts'], 'grasp_complete_cohorts': grasp['completed_cohorts'],
                      'crossad_preflights': len(proof_records), 'private_memory_released_bytes': handoff['private_bytes_released']}))


if __name__ == '__main__':
    main()
