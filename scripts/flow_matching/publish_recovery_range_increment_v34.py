"""Publish complete original-seed recovery range results, preserving point rows."""
import argparse
import csv
import json
import os
import subprocess

import psutil

from scripts.flow_matching import publish_ready_strict_metrics_v33 as recovery

original = recovery.original
ROOT = original.ROOT


def range_only_delta(before, after):
    if len(before) != len(after) or [original.key(r) for r in before] != [original.key(r) for r in after]:
        raise ValueError('Every original scientific slot and order must remain')
    changed = []
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if (new['protocol'] != original.PROTOCOL or new['task'] != 'anomaly_detection'
                or new['metric'] not in original.RANGE_METRICS):
            raise ValueError('Only original strict range rows may change in this increment')
        if not int(old['n']) < int(new['n']) <= int(old['required_repeats']):
            raise ValueError('Keep the original range repeat budget')
        changed.append(index)
    return changed


def worker(config, path):
    if any(c['point_metrics'] for c in config['cohorts']):
        raise ValueError('This increment cannot register a point metric change')
    bindings = [recovery.verify_recovery_binding(b, config['cohorts']) for b in config['recovery_bindings']]
    original.audit_points = recovery.previous.full_points
    original.worker(config, path)
    def rows(folder):
        with (folder/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    base = ROOT/config['base_publication']
    target = ROOT/config['output_directory']
    _, before = rows(base)
    headers, after = rows(target)
    changed = range_only_delta(before, after)
    text = (target/'all_metrics.md').read_text(encoding='utf-8')
    labels = [line for line in text.splitlines() if line.startswith('| ')][0][2:-2].split(' | ')
    if len(headers) != 31 or len(labels) != 31:
        raise ValueError('Every original CSV field required')
    intro = text.split('\n| ', 1)[0].replace('指标 v31', '指标 v34')
    (target/'all_metrics.md').write_text(intro+'\n\n'+original.legacy.table(labels,
        [[r.get(h) or '—' for h in headers] for r in after])+'\n', encoding='utf-8')
    proof = original.read(ROOT/config['proof_directory']/'independent_complete_saved_score_audit.json')
    if changed != proof['changed_positions']:
        raise ValueError('Every changed range metric must have full audit evidence')
    original.exclusive_write(ROOT/config['proof_directory']/'original_seed_recovery_bindings.json', bindings)
    receipt = original.read(target/'publication_validation.json')
    receipt['all_point_metrics_and_original_capture_times_exact'] = True
    receipt['original_seed_recovery_bindings_verified'] = bindings
    receipt['prior_full_saved_score_audit_evidence'] = config['prior_full_saved_score_audit_evidence']
    receipt['undefined_range_metrics_remain_unavailable'] = True
    (target/'README.rst').write_text('完整算法—数据集指标 v34\n==========================\n\n'
        '保留全部7283项指标、31字段和所有点级成绩。本轮仅把已完成原种子恢复运行的全范围评价纳入既有严格槽位，'
        '逐份核验完整保存分数、验证阈值、原始数据和来源绑定。范围指标采用原完整实体宏平均；无异常事件等导致未定义的指标仍缺失，不补零。\n\n'
        f"本轮核验{len(proof['records'])}份完整保存分数及原生全范围评价，更新{len(changed)}项范围指标。"
        f"实测{receipt['measured_metric_rows']}项，缺结果{receipt['missing_metric_rows']}项。"
        '上轮353份完整核验和独立复查证据保留并校验。恢复执行只改变ID和输出目录，其余原模型/数据/预算/种子/冻结文件完全一致，不增加训练种子。\n\n'
        '本次发布未新增模型训练、模型重推断或VUS算法重运行；实际完整范围评价已由原评价器完成。全部45论文/681方法基线消融/273原始数据任务和采用基线仍须完成；'
        'SMAP历史预处理成绩仍为历史记录，修正控制尚未全部完成。作者等价和完整TAB训练等价未认证。\n', encoding='utf-8')
    receipt['outputs'] = {p.name: original.stream_sha(p) for p in target.iterdir()
        if p.is_file() and p.name != 'publication_validation.json'}
    (target/'publication_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = original.read(ROOT/args.config)
    original.verify_registration(config)
    if args.worker:
        return worker(config, args.config)
    api = original.resource_api()
    base = ROOT/config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('Full recovery-range audit already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        state='starting', configuration_sha256=original.stream_sha(ROOT/args.config))
    def update(values):
        state.update(values)
        original.legacy.write(base/'status.json', state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], update):
            env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'), CUDA_VISIBLE_DEVICES='-1',
                PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (base/'stdout.log').open('xb') as out, (base/'stderr.log').open('xb') as err:
                child = subprocess.Popen([config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.publish_recovery_range_increment_v34', '--config', args.config, '--worker'],
                    cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = psutil.Process(child.pid)
                update(dict(state='auditing_all_registered_recovery_range_scores', active_pid=child.pid,
                    active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                def guard_update(values):
                    values = dict(values)
                    if values.get('state') == 'replaying_full_checkpoints':
                        values['state'] = 'auditing_all_registered_recovery_range_scores'
                    update(values)
                code, usage = original.capped_wait(child, config, api, guard_update)
            receipt = dict(exit_code=code, full_saved_score_audit_passed=code == 0, **usage)
            original.exclusive_write(base/'receipt.json', receipt)
            update(dict(state='completed' if code == 0 else 'failed_preserved', active_pid=None, **receipt))
            if code:
                raise RuntimeError('Complete recovery-range audit failed; attempt preserved')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
