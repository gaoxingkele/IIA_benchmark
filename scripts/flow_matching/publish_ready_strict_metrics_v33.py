"""Include verified original-seed recovery artifacts without changing any budget."""
import argparse
import json
import os
import subprocess

import psutil

from scripts.flow_matching import publish_ready_strict_metrics_v32 as previous

original = previous.original
ROOT = original.ROOT


def verify_recovery_binding(binding, cohorts):
    recovery = original.read(ROOT / binding['recovery_queue'])
    matches = [j for j in recovery['jobs'] if j['original_id'] == binding['original_id']]
    if len(matches) != 1:
        raise ValueError('Exactly one frozen original recovery binding required')
    registered = matches[0]
    queue_path = ROOT / registered['original_queue']
    if original.stream_sha(queue_path) != registered['original_queue_sha256']:
        raise ValueError('Original canonical queue hash changed')
    canonical = [j for j in original.read(queue_path)['jobs'] if j['id'] == binding['original_id']]
    if canonical != [registered['original_job']]:
        raise ValueError('Exact original canonical job required')
    mirror = original.read(ROOT / binding['mirror_queue'])
    recovered = [j for j in mirror['jobs'] if j['id'] == binding['recovery_id']]
    if len(recovered) != 1 or recovered[0] != registered['job']:
        raise ValueError('Exact frozen recovery mirror job required')
    before, after = canonical[0], recovered[0]
    differing = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if differing != {'id', 'output_directory'}:
        raise ValueError('Recovery may only change execution identity and output directory')
    matches = [e for c in cohorts for e in c['runs'] if e['id'] == binding['recovery_id']]
    if len(matches) != 1 or matches[0]['seed'] != before['seed']:
        raise ValueError('Recovery must occupy its unique original seed slot')
    if any(e['id'] == binding['original_id'] for c in cohorts for e in c['runs']):
        raise ValueError('Original and recovery attempts cannot both count as a seed')
    result = original.read(ROOT / matches[0]['result'])
    if result['job'] != after or result['status'] != 'completed':
        raise ValueError('Complete original-budget recovery result required')
    return dict(original_id=before['id'], recovery_id=after['id'], seed=before['seed'],
        only_execution_identity_and_output_changed=True, new_training_seed_slots=0)


def worker(config, path):
    bindings = [verify_recovery_binding(b, config['cohorts']) for b in config['recovery_bindings']]
    if len({b['original_id'] for b in bindings}) != len(bindings):
        raise ValueError('Unique recovery bindings required')
    previous.worker(config, path)
    target = ROOT / config['output_directory']
    for name in ('all_metrics.md', 'README.rst'):
        file = target / name
        text = file.read_text(encoding='utf-8').replace('指标 v32', '指标 v33')
        if name == 'README.rst':
            text += '\n另纳入两份原种子恢复运行：只改变执行ID和输出目录，模型、数据、种子、训练预算及冻结文件逐字段一致。全部353份重核验；原351份未移除。第五个种子的点级成绩保留5/5，范围成绩仍按实际已有评价数。两个先前失败尝试均保留。\n'
        file.write_text(text, encoding='utf-8')
    original.exclusive_write(ROOT / config['proof_directory'] / 'original_seed_recovery_bindings.json', bindings)
    receipt = original.read(target / 'publication_validation.json')
    receipt['original_seed_recovery_bindings_verified'] = bindings
    receipt['previous_failed_audits_preserved'] = config['previous_failed_audits']
    receipt['outputs'] = {p.name: original.stream_sha(p) for p in target.iterdir()
        if p.is_file() and p.name != 'publication_validation.json'}
    (target / 'publication_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = original.read(ROOT / args.config)
    original.verify_registration(config)
    if args.worker:
        return worker(config, args.config)
    api = original.resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full ready-cohort audit already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        state='starting', configuration_sha256=original.stream_sha(ROOT / args.config))
    def update(values):
        state.update(values)
        original.legacy.write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), CUDA_VISIBLE_DEVICES='-1',
                PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (base/'stdout.log').open('xb') as out, (base/'stderr.log').open('xb') as err:
                child = subprocess.Popen([config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.publish_ready_strict_metrics_v33', '--config', args.config, '--worker'],
                    cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = psutil.Process(child.pid)
                update(dict(state='auditing_all_full_saved_scores', active_pid=child.pid,
                    active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                def guard_update(values):
                    values = dict(values)
                    if values.get('state') == 'replaying_full_checkpoints':
                        values['state'] = 'auditing_all_full_saved_scores'
                    update(values)
                code, usage = original.capped_wait(child, config, api, guard_update)
            receipt = dict(exit_code=code, full_saved_score_audit_passed=code == 0, **usage)
            original.exclusive_write(base/'receipt.json', receipt)
            update(dict(state='completed' if code == 0 else 'failed_preserved', active_pid=None, **receipt))
            if code:
                raise RuntimeError('Full ready-cohort audit failed; complete attempt preserved')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
