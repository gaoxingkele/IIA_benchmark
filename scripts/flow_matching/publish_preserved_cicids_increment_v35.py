"""Verify the complete final original CICIDS seed and all point/range metrics."""
import argparse
import csv
import json
import os
import subprocess

import psutil

from scripts.flow_matching import publish_ready_strict_metrics_v33 as recovery

original = recovery.original
ROOT = original.ROOT
ALGORITHM = 'fm_rectified_60epoch_budget_control'


def complete_original_seed_delta(before, after):
    if len(before) != len(after) or [original.key(r) for r in before] != [original.key(r) for r in after]:
        raise ValueError('Keep all original scientific slots and order')
    changed = []
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if (new['algorithm'] != ALGORITHM or new['dataset'] != 'CICIDS'
                or new['protocol'] != original.PROTOCOL or new['task'] != 'anomaly_detection'
                or new['metric'] not in original.POINT_METRICS+original.RANGE_METRICS
                or (old['n'], new['n'], old['required_repeats'], new['required_repeats']) != ('4','5','5','5')):
            raise ValueError('Only the original complete fifth CICIDS seed may update')
        changed.append(index)
    if {after[i]['metric'] for i in changed} != set(original.POINT_METRICS+original.RANGE_METRICS) or len(changed) != 8:
        raise ValueError('Every one of the eight original point/range metrics must complete')
    return changed


def worker(config, path):
    if (len(config['cohorts']) != 1 or len(config['cohorts'][0]['runs']) != 5
            or {e['seed'] for e in config['cohorts'][0]['runs']} != {1103,1104,1105,1106,1107}
            or len(config['recovery_bindings']) != 1):
        raise ValueError('All five original unique seeds and their one preserved binding required')
    binding = recovery.verify_recovery_binding(config['recovery_bindings'][0], config['cohorts'])
    original.audit_points = recovery.previous.full_points
    original.worker(config, path)
    def rows(folder):
        with (folder/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    _, before = rows(ROOT/config['base_publication'])
    target = ROOT/config['output_directory']
    headers, after = rows(target)
    changed = complete_original_seed_delta(before, after)
    proof = original.read(ROOT/config['proof_directory']/'independent_complete_saved_score_audit.json')
    if changed != proof['changed_positions']:
        raise ValueError('Every completed metric must have full original source evidence')
    text = (target/'all_metrics.md').read_text(encoding='utf-8')
    labels = [line for line in text.splitlines() if line.startswith('| ')][0][2:-2].split(' | ')
    if len(labels) != len(headers) or len(headers) != 31:
        raise ValueError('Every original CSV field required')
    intro = text.split('\n| ', 1)[0].replace('指标 v31','指标 v35')
    (target/'all_metrics.md').write_text(intro+'\n\n'+original.legacy.table(labels,
        [[r.get(h) or '—' for h in headers] for r in after])+'\n', encoding='utf-8')
    original.exclusive_write(ROOT/config['proof_directory']/'original_preserved_seed_binding.json', binding)
    receipt = original.read(target/'publication_validation.json')
    receipt['complete_original_CICIDS_training_seed_slots'] = 5
    receipt['all_eight_point_and_range_metrics_complete_original_five_seeds'] = True
    receipt['preserved_v4_source_matches_original_budget_verified'] = binding
    receipt['previous_full353_and35_audit_evidence'] = config['previous_full_audit_evidence']
    (target/'README.rst').write_text('完整算法—数据集指标 v35\n==========================\n\n'
        'CICIDS的60轮Rectified Flow本地配置原种子1103已在preserved-v4目录完整训练，旧评价队列误等候旧失败目录。'
        '新队列保留全部六个原恢复任务、旧失败记录和五份已有全范围评价，只修正已完成结果的引用。完整原生CICIDS范围评价成功后，核验全部五个原种子的完整训练/验证/测试保存分数、数据标签、99百分位验证校准、所有原生范围来源和实体宏平均。\n\n'
        '八项点级/范围指标由4/5补齐为5/5；其余7275行及采集时间、所有无关任务导出不变。各原训练种子1103至1107只计一次；恢复任务仅ID/输出目录与原任务不同，保持原模型/数据/设备/训练预算/冻结文件。'
        '本次发布不新增模型训练、模型重推断或范围算法重运行。此前353份及35份核验与独立复查证据保留。\n\n'
        f"全部7283项登记指标；实测{receipt['measured_metric_rows']}项，缺结果{receipt['missing_metric_rows']}项。"
        '这些是本地时间序列转录/预算控制的完整实验，不是原Rectified Flow图像论文复现。全部45论文/681方法基线消融/273原始数据任务和采用基线仍保持原范围、尚未完成，作者和完整TAB训练等价未认证。旧SMAP预处理成绩仍为历史记录。\n', encoding='utf-8')
    receipt['outputs'] = {p.name:original.stream_sha(p) for p in target.iterdir()
        if p.is_file() and p.name != 'publication_validation.json'}
    (target/'publication_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    config=original.read(ROOT/args.config)
    original.verify_registration(config)
    if args.worker:
        return worker(config,args.config)
    api=original.resource_api()
    base=ROOT/config['state_root']
    base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('Full preserved CICIDS audit already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        state='starting',configuration_sha256=original.stream_sha(ROOT/args.config))
    def update(values):
        state.update(values)
        original.legacy.write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',
                PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen([config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.publish_preserved_cicids_increment_v35','--config',args.config,'--worker'],
                    cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid)
                update(dict(state='auditing_all_original_CICIDS_seed_arrays',active_pid=child.pid,
                    active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                def guard_update(values):
                    values=dict(values)
                    if values.get('state')=='replaying_full_checkpoints':
                        values['state']='auditing_all_original_CICIDS_seed_arrays'
                    update(values)
                code,usage=original.capped_wait(child,config,api,guard_update)
            receipt=dict(exit_code=code,full_saved_score_audit_passed=code==0,**usage)
            original.exclusive_write(base/'receipt.json',receipt)
            update(dict(state='completed' if code==0 else 'failed_preserved',active_pid=None,**receipt))
            if code:
                raise RuntimeError('Full original CICIDS audit failed; complete attempt preserved')
    finally:
        owner.close()


if __name__=='__main__':
    main()
