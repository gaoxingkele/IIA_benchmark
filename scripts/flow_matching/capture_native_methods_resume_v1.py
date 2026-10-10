"""Capture live native recovery handles and bind existing full-result audit evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.capture_experiment_restart_progress_v1 import observe
from scripts.flow_matching.run_light_controller import ROOT, digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_spectral_table2_original_v1 import fingerprint, verify


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args(); config_path=ROOT/args.config;config=read(config_path)
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Observed captures must remain immutable')
    sources=[config_path,Path(__file__)]
    runtime_bindings=[]
    for relative in config['native_runtimes']:
        path=ROOT/relative;runtime=read(path);sources.append(path)
        for entry in runtime['source_receipts']:
            source=ROOT/entry['path']
            if digest(source)!=entry['sha256']:raise ValueError('Frozen native source changed: '+entry['path'])
            sources.append(source)
        runtime_bindings.append(dict(path=relative,sha256=digest(path),
            verified_source_receipts=len(runtime['source_receipts'])))
    inventory=read(ROOT/config['full_inventory']);sources.append(ROOT/config['full_inventory'])
    table2_path=ROOT/config['table2_queue'];table2=read(table2_path);verify(table2);sources.append(table2_path)
    controllers=[observe(binding) for binding in config['controllers']]
    saits_runtime=read(ROOT/config['saits_runtime']);saits_queue=read(ROOT/saits_runtime['queue'])
    saits_jobs={read(ROOT/j['config'])['id']:j for j in saits_queue['jobs']}
    verified_results=[]
    for receipt_path in sorted((ROOT/saits_runtime['state_root']/'full_result_audits').glob('*.json')):
        receipt=read(receipt_path)
        if receipt['exit_code']!=0:raise ValueError('Existing original SAITS full audit failed')
        evidence=json.loads(receipt['stdout']);job=saits_jobs[evidence['id']]
        cfg=read(ROOT/job['config']);output=ROOT/cfg['output_root'];result=read(output/'result.json')
        if not evidence['full_training_finished'] or not evidence['metrics_independently_recomputed'] or evidence['benchmark_smoke']:
            raise ValueError('Incomplete audit cannot be promoted')
        for key,name in [('result_sha256','result.json'),('resume_sha256','resume.pt'),
                         ('best_sha256','best.pt'),('statistics_sha256',
                         'group_statistics.npz' if cfg.get('bootstrap_group')=='month' else 'patient_statistics.npz')]:
            if digest(output/name)!=evidence[key]:raise ValueError('Audited SAITS full artifact changed')
        if digest(ROOT/job['config'])!=job['config_sha256'] or result['config_sha256']!=job['config_sha256']:
            raise ValueError('Audited SAITS original configuration changed')
        verified_results.append(dict(evidence,config=job['config'],dataset=cfg['dataset'],
            seed=cfg['seed'],method=cfg['method'],result_path=(output/'result.json').relative_to(ROOT).as_posix(),
            original_queue=saits_runtime['queue'],full_audit_receipt_sha256=digest(receipt_path)))
        sources += [receipt_path,ROOT/job['config'],output/'result.json']
    methods=read(ROOT/config['table2_method_runtime'])
    method_proofs={}
    for method,relative in methods['method_proofs'].items():
        path=ROOT/relative
        method_proofs[method]=dict(path=relative,exists=path.exists(),proof=read(path) if path.exists() else None)
        if path.exists():sources.append(path)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        boot_utc=datetime.fromtimestamp(psutil.boot_time(),timezone.utc).isoformat(),
        previous_goal_turn_classification='no progress toward unfinished experiments: previous reply rechecked and listed existing result captures; this iteration adds full native dispatch and verifies actual live handles',
        current_goal_turn_classification='progress: original MaelNet and SAITS adapters validated; all30 Table2 method-gated full jobs registered and dispatcher started; complete SAITS checkpoints bound to independent audits',
        objective='这个方向相关所有的已知未完成的实验都跑一遍',
        original_papers=inventory['paper_count'],inventory_record_counts=inventory['record_counts_by_role'],
        all_original_experiments_complete=False,controllers=controllers,runtime_bindings=runtime_bindings,
        resource_snapshot=resource_api()['resource_snapshot'](),
        saits=dict(original_jobs=len(saits_queue['jobs']),existing_independently_audited_results=verified_results,
            completed_existing_results=len(verified_results),remaining_jobs=len(saits_queue['jobs'])-len(verified_results),
            native_epoch_resume_retains_model_optimizer_patience_and_all_RNG=True),
        table2=dict(full_jobs=len(table2['jobs']),methods=dict(Counter(j['method'] for j in table2['jobs'])),
            queue_fingerprint=fingerprint(table2),method_proofs=method_proofs,
            simultaneous_cpu_gpu_reservation=True,full_original_budgets_unchanged=True,
            new_formal_metrics_published=False),
        tests=config['tests'],remaining_work=config['remaining_work'],
        boundary='Live handles prove activity or waiting only. Full-result verifiers and independent array/checkpoint audits remain required. No smoke or preflight score enters benchmark results. The full45-paper original scope remains open.')
    target.mkdir(parents=True)
    (target/'runtime_observations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 原始方法与完整预算实验继续执行','',
        '核验时间：'+report['captured_utc']+'。完整45篇论文目标保持不变，尚未完成。','',
        'MaelNet保留150项原作者多阶段任务，逐阶段共享CPU资源锁；旧失败输出和阶段记录保留。'
        'SAITS/BRITS保留210项原始时间序列任务，已有10个完整结果通过独立误差、全测试单元及检查点审计。'
        'Electricity继续使用作者原生整轮检查点恢复，模型、优化器、patience和全部随机状态一起恢复。','',
        'Table2保留Spectral Flow和SDFormer-AR共30项原始任务。各方法仅在所有原数据集的完整批量'
        '反向传播/优化更新及8进程全数据加载检查通过后启动；完整训练同时预约GPU及CPU资源。'
        '未缩减批量、更新预算、采样步数、数据量或评价重复数。Spectral完整GPU预检缺口仍保留。','',
        '[进程身份、原始范围、源代码校验和绑定及未完成项](runtime_observations.json)。','',
        '| 控制器 | PID | 实际存活 | Python子进程数 |','|---|---:|---|---:|']
    lines += [f"| {b['name']} | {b['pid']} | {b['verified_live']} | {len(b['children'])} |" for b in controllers]
    lines += ['','本次相关验证：'+config['tests'],'','仍需继续：','']+['- '+x for x in config['remaining_work']]
    lines += ['','本记录不新增或替换任何科学性能指标；运行中不能算作完成。','']
    (target/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    sources=list(dict.fromkeys(sources))
    validation=dict(all_original_experiments_complete=False,
        source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in sources],
        outputs={p.name:digest(p) for p in target.iterdir()},new_benchmark_metrics_published=False)
    (target/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(observed_controllers=len(controllers),verified_live=sum(b['verified_live'] for b in controllers),
        existing_full_saits_results=len(verified_results),all_table2_jobs=len(table2['jobs']))))


if __name__=='__main__':main()
