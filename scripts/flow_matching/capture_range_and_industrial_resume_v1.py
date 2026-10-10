"""Capture exact live recovery handles and full original evaluator/industrial scope."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.capture_experiment_restart_progress_v1 import observe
from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();path=ROOT/args.config;config=read(path);target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Published observed captures remain immutable')
    observations=[observe(b) for b in config['controllers']]
    for observation in observations:
        state=observation['state'];observation['state_counts']=dict(
            completed_jobs=state.get('completed_jobs'),failed_jobs=state.get('failed_jobs'),
            pending_model_results=state.get('pending_model_results'))
        observation['state']={k:v for k,v in state.items() if k not in {'completed','failures','records','jobs','superseded_original_attempts'}}
    sources=[path,Path(__file__)];scope=[]
    for relative in config['range_configs']:
        source=ROOT/relative;settings=read(source);sources.append(source)
        for entry in settings['source_receipts']:
            p=ROOT/entry['path']
            if digest(p)!=entry['sha256']:raise ValueError('Frozen full evaluator source changed')
            sources.append(p)
        jobs=[]
        for queue_path in settings['queue_paths']:
            sources.append(ROOT/queue_path);jobs.extend(read(ROOT/queue_path)['jobs'])
        scope.append(dict(config=relative,registered_attempts=len(jobs),
            queue_paths=settings['queue_paths'],full_metric_semantics=settings['metric_semantics'],
            original_thresholds_and_all_entity_timestamp_coverage_retained=True))
    for relative in config['new_runtimes']:
        source=ROOT/relative;runtime=read(source);sources.append(source)
        for entry in runtime['source_receipts']:
            p=ROOT/entry['path']
            if digest(p)!=entry['sha256']:raise ValueError('Frozen resumed controller/audit source changed')
            sources.append(p)
    audit_config=read(ROOT/config['grasp_audit_config']);proof_path=ROOT/audit_config['output_directory']/'independent_array_audit.json'
    proof=read(proof_path) if proof_path.exists() else None
    if proof is not None:sources.append(proof_path)
    industrial=read(ROOT/config['industrial_queue']);sources.append(ROOT/config['industrial_queue'])
    industrial_state=read(ROOT/config['industrial_state']);industrial_counts={}
    for record in industrial_state['records']:
        industrial_counts[record['status']]=industrial_counts.get(record['status'],0)+1
    recovery=read(ROOT/config['industrial_exact_recovery_queue']);sources.append(ROOT/config['industrial_exact_recovery_queue'])
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='progress: full native MaelNet/SAITS and all30 method-gated Table2 jobs registered, validated and pushed in6f08cfc/44825e1',
        current_goal_turn_classification='progress: all four complete range queues restored; industrial550-job audit found463 complete and87 blocked, nine stable empty-output original slots registered and recovery started; three full GRASP ablations registered for independent full-array audit',
        objective='这个方向相关所有的已知未完成的实验都跑一遍',
        all_original_experiments_complete=False,full_original_papers=45,
        controllers=observations,range_scope=scope,industrial_original_jobs=len(industrial['jobs']),
        industrial_original_status_counts=industrial_counts,industrial_exact_original_slots=len(recovery['jobs']),
        registered_grasp_full_audits=audit_config['full_results'],grasp_independent_audit=proof,
        resource_snapshot=resource_api()['resource_snapshot'](),tests=config['tests'],remaining_work=config['remaining_work'],
        boundary='Registered range attempts include original/repaired attempts; not unique training counts or benchmark successes. File/process observations do not certify performance. Every original task, method, ablation and seed remains in scope; no budget reduction or smoke substitution.')
    target.mkdir(parents=True)
    (target/'runtime_observations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 完整范围评价与工业实验恢复','',report['captured_utc'],'',
        '四条范围评价队列已恢复：主异常检测1750项登记尝试、Reflow140项、工业1400项、MOMENT72项。'
        '全部保存分数、每个实体/时间点、250个VUS阈值及每个整数缓冲、原Affiliation积分和验证阈值保持不变。'
        '这些3362项包含原始/修复尝试，不代表3362项独立训练或已完成成绩。','',
        '550项原始工业CPU任务核查结束：463项完整，87项受旧目录阻挡。75项属于旧不稳定数值版本，'
        '3项已有完整登记恢复，9项稳定版本只有空目录。为这9项登记并启动同原数据、参数、预算和种子的'
        '精确重跑；旧目录保留，75项旧数值问题不计为已完成。','',
        'GRASP/SWAN/seed1103的τ=0、0.5、1已登记只读独立审计：完整1500轮、93000更新历史、'
        '全部评分配方、完整测试/验证行和标签及检查点身份。审计进程等待或运行均不等于已通过。','',
        '[实际进程身份、完整范围、审计证据和剩余工作](runtime_observations.json)。','',
        '| 控制器 | PID | 核验时存活 | Python子进程数 |','|---|---:|---|---:|']
    lines += [f"| {b['name']} | {b['pid']} | {b['verified_live']} | {len(b['children'])} |" for b in observations]
    lines += ['','相关验证：'+config['tests'],'','尚未完成：','']+['- '+x for x in config['remaining_work']]
    lines += ['','全45篇论文目标仍在执行；本记录不新增或替代科学性能成绩。','']
    (target/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    sources=list(dict.fromkeys(sources))
    validation=dict(source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in sources],
        outputs={p.name:digest(p) for p in target.iterdir()},all_original_experiments_complete=False,
        new_benchmark_metrics_published=False)
    (target/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(controllers=len(observations),verified_live=sum(b['verified_live'] for b in observations),
        full_range_registered_attempts=sum(s['registered_attempts'] for s in scope),industrial_full_jobs=len(industrial['jobs']),
        grasp_audit_passed=proof['full_result_audit_passed'] if proof else None)))


if __name__=='__main__':main()
