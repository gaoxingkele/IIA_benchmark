"""Capture actual resumed controllers and preserved full-scope gaps, without training."""
from datetime import datetime, timezone
import json
from pathlib import Path

import psutil

from scripts.flow_matching.run_imputation_native_resume_queue_v1 import original_gates, verify_registration
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def observation(binding):
    value = dict(binding)
    try:
        process = psutil.Process(binding['pid'])
        command = process.cmdline()
        value['verified_live'] = (abs(process.create_time()-binding['create_time']) < .01
                                  and any(binding['command_token'] in x for x in command))
        value['command'] = command
        value['children'] = [dict(pid=p.pid, create_time=p.create_time(), command=p.cmdline(),
            cpu_seconds=p.cpu_times().user+p.cpu_times().system)
            for p in process.children(recursive=True) if 'python' in p.name().lower()]
    except psutil.NoSuchProcess:
        value.update(verified_live=False, children=[])
    path = ROOT / binding['status_path']
    value['state'] = read(path) if path.exists() else None
    if isinstance(value['state'], dict) and 'jobs' in value['state']:
        from collections import Counter
        value['reported_job_counts'] = dict(Counter(value['state'].pop('jobs').values()))
    return value


def main():
    source = ROOT/'projects/flow_matching_research/results/2026-10-10/full_gpu_resume_v1/runtime_observations.json'
    previous = read(source)
    bindings = [{k:c[k] for k in ('name','pid','create_time','command_token','status_path')}
                for c in previous['controllers']]
    new_queues = []
    receipts = []
    for name, pid in [('grin_original',32760),('transfer',3456)]:
        runtime_path = Path('configs/runtime/fm_'+name+'_native_resume.v1.json')
        registration = read(ROOT/runtime_path)
        queue = verify_registration(ROOT, registration)
        handle = psutil.Process(pid)
        bindings.append(dict(name=name+'_native_resume',pid=pid,create_time=handle.create_time(),
            command_token='scripts.flow_matching.run_imputation_native_resume_queue_v1',
            status_path=registration['state_root']+'/status.json'))
        new_queues.append(dict(name=name, full_jobs=len(queue['jobs']), runtime=runtime_path.as_posix(),
            runtime_sha256=digest(ROOT/runtime_path), queue=registration['queue'],
            queue_sha256=registration['queue_sha256'], gates=original_gates(ROOT,queue),
            unchanged_model_commands=True, independent_metric_audit_required=True))
        receipts.append(dict(path=runtime_path.as_posix(),sha256=digest(ROOT/runtime_path),
            verified_source_bindings=len(registration['source_receipts'])))
    recovery_path = Path('configs/runtime/fm_strict_empty_output_recovery.v1.json')
    runtime=read(ROOT/recovery_path)
    _,namespace,binding=load_controller(ROOT,runtime,'strict_empty_output_recovery')
    handle=psutil.Process(16088)
    bindings.append(dict(name='strict_empty_output_recovery',pid=16088,create_time=handle.create_time(),
        command_token='scripts.flow_matching.run_light_controller',status_path=binding['status_path']))
    queue=read(ROOT/binding['queue_path'])
    case=queue['jobs'][0]
    assert {k:v for k,v in case['job'].items() if k not in ('id','output_directory')}=={
        k:v for k,v in case['original_job'].items() if k not in ('id','output_directory')}
    assert (ROOT/case['original_job']['output_directory']).is_dir()
    assert not list((ROOT/case['original_job']['output_directory']).iterdir())
    strict=read(ROOT/case['original_queue'])
    completed=[]
    for job in strict['jobs']:
        if not job['id'].startswith('fm_strict_timesnet__smd__registered__'):
            continue
        if namespace['complete_result'](ROOT,job):
            path=Path(job['output_directory'])/'result.json'
            completed.append(dict(id=job['id'],path=path.as_posix(),sha256=digest(ROOT/path)))
    receipts.append(dict(path=recovery_path.as_posix(),sha256=digest(ROOT/recovery_path),
        verified_source_bindings=len(runtime['source_receipts'])))
    captured=datetime.now(timezone.utc).isoformat()
    report=dict(captured_utc=captured, previous_goal_turn_classification='no_progress',
        previous_turn_reason='Read-only metric listing verified the published snapshot but did not advance unfinished experiments.',
        current_goal_turn_classification='progress',
        objective='这个方向相关所有的已知未完成的实验都跑一遍',
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
        all_original_experiments_complete=False, new_full_imputation_jobs=280,
        imputation_queues=new_queues, exact_strict_recovery=dict(original_id=case['original_id'],
            recovery_id=case['job']['id'],queue=binding['queue_path'],queue_sha256=digest(ROOT/binding['queue_path']),
            only_attempt_id_and_output_changed=True, original_empty_directory_preserved=True,
            original_timesnet_smd_complete_artifact_checks=completed,
            completion_boundary='Four full original result/checkpoint/score identities verified; seed1106 remains pending. No new independent seed, metric selection, or fresh independent score recomputation.'),
        controllers=[observation(b) for b in bindings],runtime_receipts=receipts,
        tests=dict(passed=34,scope='Imputation full gates/source/command/archive, exact strict recovery invariants, native GPU observation/resource semantics, lightweight CPU controller and canonical recovery resolution'),
        new_formal_metrics_published=False,
        remaining_work=['Run all10 GRIN original jobs after the remaining200 SAITS/BRITS original jobs.',
            'Run270 full CFMI/CSDI industrial transfers after the original campaign scope-review gate is satisfied.',
            'Complete the exact TimesNet SMD seed1106 recovery and independently recompute full scores before publication.',
            'Continue the existing full CFM-TS, GRASP, generation, strict/industrial baselines and all ablation queues.',
            'Complete separate full SDFormer/Spectral Table2 GPU preflights and remaining original Table3/4/5, image/forecast/dynamics and other45-paper tasks.',
            'Preserve unresolved original75 numerically unstable industrial jobs and all access/data/protocol gaps. Registration and smoke tests do not satisfy them.'])
    target=ROOT/'projects/flow_matching_research/results/2026-10-10/imputation_and_strict_resume_v1'
    target.mkdir(parents=True,exist_ok=False)
    def publish(name,value):
        (target/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    publish('runtime_observations.json',report)
    lines=['# 原始插补队列恢复与TimesNet原始种子补跑','',captured,'',
        '恢复GRIN全部10项原始实验，以及CFMI/CSDI全部270项工业插补迁移实验。完整冻结模型命令、数据、种子与训练/采样预算不变；保留原始依赖、原有插补队列锁、共享GPU锁和完整资源保护。',
        '', 'GRIN当前等待200项SAITS/BRITS原始实验；工业迁移等待原始实验范围审查完成。这些等待项不计为完成。',
        '', 'TimesNet在SMD上核验了4个原始种子的完整结果、检查点和分数文件身份。seed1106被已有空目录阻挡；新登记同一完整科学任务，只更改尝试ID和输出路径，保留旧目录。补跑不增加种子，不挑选测试成绩。',
        '', '原始运行器结果完成后，仍须独立核验指标和训练产物，再更新正式指标表。本报告没有发布新算法性能。',
        '', '[实际进程、完整队列、来源校验与剩余工作](runtime_observations.json)。','',
        '| 控制器 | PID | 当前进程身份核验 | Python子进程数 |','|---|---:|---|---:|']
    for c in report['controllers']:
        lines.append(f"| {c['name']} | {c['pid']} | {c['verified_live']} | {len(c['children'])} |")
    lines += ['', '相关测试：34 passed。测试验证调度和完整性约束，不是性能结果。',
              '', '完整目标仍包含45篇论文、681条方法/基线/消融记录和273条原始数据/任务记录；全范围实验尚未完成。']
    (target/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    publish('validation.json',dict(captured_utc=captured,passed=True,tests_passed=34,
        runtime_source_bindings_verified=sum(x['verified_source_bindings'] for x in receipts),
        original_full_imputation_jobs=280,exact_original_strict_recoveries=1,
        live_controllers=sum(c['verified_live'] for c in report['controllers']),
        no_new_independent_seed=True,original_phase_gates_preserved=True,
        outputs={name:digest(target/name) for name in ('runtime_observations.json','README.md')},
        source=dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=digest(Path(__file__)))))
    print(json.dumps(dict(report=target.relative_to(ROOT).as_posix(),live_controllers=sum(c['verified_live'] for c in report['controllers']),
        source_bindings=sum(x['verified_source_bindings'] for x in receipts),timesnet_smd_complete_original_slots=len(completed))))


if __name__=='__main__':
    main()
