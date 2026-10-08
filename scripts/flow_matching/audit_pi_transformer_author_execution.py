"""Verify pinned Pi queue identity, live handles and original score/PA receipts."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts
from scripts.flow_matching.run_pi_transformer_job import author_pa, binary_metrics


def audit():
    import numpy as np
    settings = json.loads((ROOT / 'configs/experiments/fm_pi_transformer_author_execution.v1.json').read_text(encoding='utf-8'))
    registration = json.loads((ROOT / 'docs/reports/fm_pi_transformer_author_registration_2026-10-09.json').read_text(encoding='utf-8'))
    queue_path = ROOT / settings['queue_path']
    if sha(queue_path) != registration['queue_sha256']:
        raise ValueError('Pi registered queue changed')
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    frozen = {}
    for job in queue['jobs']:
        for source in job['frozen_files']:
            if source['path'] in frozen and frozen[source['path']] != source['sha256']:
                raise ValueError('Pi frozen source conflicts')
            frozen[source['path']] = source['sha256']
    for name, expected in frozen.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen Pi source/data changed: ' + name)
    state_path = ROOT / settings['output_root'] / 'status.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    processes = []
    for role, pid in [('worker', state.get('pid')), ('child', state.get('active_pid'))]:
        try:
            p = psutil.Process(pid) if pid else None
            if p is None:
                continue
            required = queue['worker_script'] if role == 'worker' else queue['child_script']
            if not any(required in a for a in p.cmdline()):
                continue
            for process in [p] + (p.children(recursive=True) if role == 'child' else []):
                processes.append({'role': role, 'pid': process.pid, 'cmdline': process.cmdline(),
                                  'cpu_seconds': sum(process.cpu_times()[:2]), 'rss_bytes': process.memory_info().rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    worker_live = any(p['role'] == 'worker' for p in processes)
    child_live = any(p['role'] == 'child' for p in processes)
    statuses, completed = [], []
    for job in queue['jobs']:
        base = ROOT / job['output_directory']
        final = base / 'result.json'
        if final.exists():
            result = json.loads(final.read_text(encoding='utf-8'))
            if result['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest():
                raise ValueError('Pi result job differs')
            verify_artifacts(result)
            with np.load(base / 'checkpoints/author_scores.npz', allow_pickle=False) as archive:
                values = {k: archive[k] for k in archive.files}
            window = job['author_config']['data']['win_size']
            expected = job['data_audit']['expected_shapes']['test'][0] // window * window
            if len(values['labels']) != expected or result['scored_test_points'] != expected:
                raise ValueError('Pi original inference coverage differs')
            if len(values['fused_train']) != result['training_windows'] * window:
                raise ValueError('Pi original full overlapping train score coverage differs')
            expected_threshold = np.percentile(np.concatenate([values['fused_train'], values['test_energy']]), 100-job['author_config']['data']['anomaly_ratio'])
            if expected_threshold != values['threshold'] or not np.array_equal(values['raw_predictions'], values['test_energy'] > expected_threshold):
                raise ValueError('Pi source calibration differs')
            if not np.array_equal(author_pa(values['labels'], values['raw_predictions']), values['pa_predictions']):
                raise ValueError('Pi source PA cannot be reproduced')
            independent = binary_metrics(values['labels'], values['pa_predictions'])
            independent['pointwise_no_PA_same_author_threshold'] = binary_metrics(values['labels'], values['raw_predictions'], values['test_energy'])
            if independent != result['metrics']:
                raise ValueError('Pi reported metrics cannot be reproduced')
            if result['strict_TAB_result'] or not result['test_used_for_early_stopping']:
                raise ValueError('Pi fidelity/protocol misclassified')
            status = 'completed'
            completed.append({'id': job['id'], 'dataset': job['dataset'], 'seed': job['seed'],
                              'author_recipe': job['author_recipe'], 'metrics': independent,
                              'result_path': final.relative_to(ROOT).as_posix(), 'result_sha256': sha(final)})
        elif worker_live and child_live and state.get('active_job') == job['id']:
            status = 'running'
        elif (base / 'failure.json').exists():
            status = 'failed_preserved'
        elif base.exists():
            status = 'partial_or_preparing'
        else:
            status = 'pending'
        statuses.append(status)
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'queue': settings['queue_path'],
              'queue_sha256': sha(queue_path), 'registered_jobs': len(queue['jobs']),
              'counts': dict(Counter(statuses)), 'frozen_files_verified': len(frozen),
              'state': state, 'verified_live_processes': processes, 'completed_records': completed,
              'strict_TAB_result': False, 'goal_achieved': False, 'boundary': settings['boundary'],
              'protocol_audit': 'docs/reports/fm_pi_transformer_protocol_audit_2026-10-09.json'}
    target = ROOT / 'projects/flow_matching_research/results/2026-10-09/pi_transformer_author'
    write_json(ROOT / 'docs/reports/fm_pi_transformer_author_execution_2026-10-09.json', report)
    write_json(target / 'execution_audit.json', report)
    for name, source in [('registration.json', 'docs/reports/fm_pi_transformer_author_registration_2026-10-09.json'),
                         ('preflight.json', settings['preflight_report']),
                         ('protocol_audit.json', report['protocol_audit'])]:
        (target / name).write_bytes((ROOT / source).read_bytes())
    lines = ['# Pi-Transformer历史镜像完整流程', '', f"快照：{report['captured_utc']}。任务：{len(queue['jobs'])}，状态：{report['counts']}。", '',
             '5个原始数据集 × 2条分支 × 每个数据集19个训练配方 × 6个种子 = 1140个完整流程、2280个阶段。',
             '原镜像全量训练与因果掩码/时间轴修正分开；PSM完整132382个stride=1训练窗口、每批256、模型512维，未削减轮数或容量。', '',
             '训练轴含层数2/3/4、宽度128/256/512/1024、注意力头2/4/8/16、批大小64/128/256/512、轮数3/5/10、KL权重和光滑权重倍率。每个完成检查点还计算温度5档×分数流3种控制。', '',
             'DFA输入以磁盘中转保留作者全部重叠窗口遍历和原抽样顺序，抽样数量/RNG/数值等价已测试。原始数据不变，中转文件放在F盘的数据目录。', '',
             '**评估边界：**测试集早停、训练/测试分数合并定阈值、原PA与尾部丢弃均被保留并标记。两个分支均不证明期刊等价，也不能充当严格TAB泛化结果。',
             '源码API顺序执行与独立CLI测试重置随机状态的对照，以及期刊两次正KL更新、按头相位、原文完整结构/先验消融，仍在待完成范围。', '',
             'preflight.json只证明原生数据和完整模型容量可反传，不是性能成绩。完成结果必须保存模型、分数、标签、训练轮次、原协议/无PA指标，并通过独立指标重算。', '',
             '[执行与真实进程审计](execution_audit.json)、[冻结登记](registration.json)、[原生预检](preflight.json)、[期刊/镜像差异](protocol_audit.json)。', '']
    (target / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'frozen_files_verified': len(frozen), 'live_pids': [p['pid'] for p in processes]}))
    return report


if __name__ == '__main__':
    audit()
