"""Record complete native scheduler handoff and unchanged GPU comparator scope."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching.capture_experiment_restart_progress_v1 import observe
from scripts.flow_matching.run_light_controller import ROOT,digest
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.grasp_protocol_runner import complete,verify


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();path=ROOT/args.config;config=read(path);target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Observed full GPU captures remain immutable')
    sources=[path,Path(__file__)];runtime_proofs=[]
    for relative in config['new_runtimes']:
        p=ROOT/relative;runtime=read(p);sources.append(p)
        for entry in runtime['source_receipts']:
            source=ROOT/entry['path']
            if digest(source)!=entry['sha256']:raise ValueError('Frozen full GPU source differs')
            sources.append(source)
        runtime_proofs.append(dict(path=relative,sha256=digest(p),verified_source_receipts=len(runtime['source_receipts'])))
    transition_path=ROOT/config['handoff_receipt'];transition=read(transition_path);sources.append(transition_path)
    queue=read(ROOT/config['grasp_queue']);verify(queue);sources.append(ROOT/config['grasp_queue'])
    job=next(j for j in queue['jobs'] if j['id']==transition['last_full_job'])
    if (transition['model_processes_stopped'] or not transition['handoff_complete']
            or not transition['full_jobs_and_budgets_unchanged'] or not complete(job)):
        raise ValueError('Full childless native boundary is not proved')
    original_model_result=ROOT/job['output_directory']/'result.json';sources.append(original_model_result)
    controllers=[observe(b) for b in config['controllers']]
    for record in controllers:
        state=record['state']
        record['state']={k:v for k,v in state.items() if k not in {'completed','failures','records','jobs','superseded_original_attempts'}}
        if 'original_status_path' in record:
            record['original_phase_gate_state']=read(ROOT/record['original_status_path'])
    gpu_queues=[]
    for relative in config['gpu_runtime_configs']:
        runtime=read(ROOT/relative);original=read(ROOT/runtime['original_runtime'])
        binding=next(b for b in original['controllers'] if b['name']==runtime['controller'])
        source=ROOT/binding['queue_path'];queue_data=read(source);sources += [source,ROOT/runtime['original_runtime']]
        if digest(source)!=binding['queue_sha256']:raise ValueError('Full registered GPU model scope differs')
        gpu_queues.append(dict(controller=runtime['controller'],queue=binding['queue_path'],queue_sha256=digest(source),
            full_registered_jobs=len(queue_data['jobs']),original_phase_gates_retained=True,
            full_source_model_commands_and_result_verifier_retained=True,
            native_gpu_slot_and_emergency_guard_cover_whole_model=True,
            author_equivalence_certified=False))
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='progress: restored complete range evaluators and registered nine exact industrial recovery slots; pushedca0fc08/18cd8fe',
        current_goal_turn_classification='progress: verified full tau8 model boundary and changed scheduler only; restored all2040 registered GPU model jobs with original phase gates and native GPU resource protection',
        objective='这个方向相关所有的已知未完成的实验都跑一遍',full_original_papers=45,
        all_original_experiments_complete=False,controllers=controllers,runtime_proofs=runtime_proofs,
        native_handoff=transition,grasp_full_queue_jobs=len(queue['jobs']),grasp_full_cohorts=queue['cohort_count'],
        gpu_queues=gpu_queues,resource_snapshot=resource_api()['resource_snapshot'](),tests=config['tests'],
        remaining_work=config['remaining_work'],
        boundary='All scientific budgets/data/seeds and existing phase gates retained. Scheduler handoff is verified, not a new training result. Live pending controllers do not prove completed experiments or author equivalence; no new benchmark metric is inferred.')
    target.mkdir(parents=True)
    (target/'runtime_observations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 完整GPU任务恢复与任务结束后的调度切换','',report['captured_utc'],'',
        'GRASP在τ=8/SWAN/seed1103完整任务退出、训练历史和全部产物校验通过后，仅切换调度器。'
        '没有停止模型进程。完整9120项实体任务、480组比较及全部原始预算保持不变，任务结束后让出GPU35秒，覆盖其他队列的资源轮询间隔。','',
        '主异常检测GPU1190项与工业GPU850项控制器已恢复。保留原有前置实验依赖、队列锁、全部模型命令、'
        '数据、种子、预算和完整结果校验器；每个未来模型执行都使用共享原生GPU锁和现有完整资源保护。'
        '此处是既有登记的本地实验预算，不代表已认证作者等价或已完成2040项训练。','',
        '[实际进程身份、切换证明、完整GPU范围和未完成项](runtime_observations.json)。','',
        '| 控制器 | PID | 核验时存活 | Python子进程数 |','|---|---:|---|---:|']
    lines += [f"| {b['name']} | {b['pid']} | {b['verified_live']} | {len(b['children'])} |" for b in controllers]
    lines += ['','相关验证：'+config['tests'],'','仍须继续：','']+['- '+x for x in config['remaining_work']]
    lines += ['','运行中、等待依赖或调度预检成功均不计为完成论文复现。','']
    (target/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    sources=list(dict.fromkeys(sources))
    validation=dict(source_receipts=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in sources],
        outputs={p.name:digest(p) for p in target.iterdir()},all_original_experiments_complete=False,
        new_benchmark_metrics_published=False)
    (target/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_live=sum(b['verified_live'] for b in controllers),
        full_gpu_registered_jobs=sum(q['full_registered_jobs'] for q in gpu_queues),
        original_model_processes_stopped=transition['model_processes_stopped'])))


if __name__=='__main__':main()
