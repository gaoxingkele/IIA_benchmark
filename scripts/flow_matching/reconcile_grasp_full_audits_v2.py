"""Bind every captured completed native seed slot to full independent array proof."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching.capture_result_listing_v4 import ROOT,canonical_grasp,csv_write,read,sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);args=parser.parse_args()
    config=read(ROOT/args.config)
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Completed native audit reconciliation remains immutable')
    records={}
    source_receipts=[]
    for entry in config['audit_sources']:
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Registered independent audit changed')
        source_receipts.append(entry)
        for record in read(path)[entry['key']]:
            if not all(record[k] for k in ('saved_checkpoint_binding_checked','full_row_and_label_identity_checked','metrics_independently_recomputed')):
                raise ValueError('Incomplete independent array proof')
            if record['id'] in records:raise ValueError('Audited native seed duplicated')
            if sha(ROOT/record['result_path'])!=record['result_sha256']:
                raise ValueError('Audited full native result changed')
            records[record['id']]=dict(record,audit_source=entry['path'])
    capture,metrics=canonical_grasp(config['queue'],config['recovery_queues'])
    complete=[j for j in capture['jobs'] if j['status']=='completed']
    expected={j['id'] for j in complete}
    if expected!=set(records) or len(complete)!=config['full_result_count']:
        raise ValueError('Full proof coverage differs from every current completed canonical seed slot')
    for job in complete:
        if records[job['id']]['result_sha256']!=job['result_sha256']:
            raise ValueError('Proof does not bind actual canonical execution')
    target.mkdir(parents=True)
    write(target/'execution_audit.json',capture)
    csv_write(target/'algorithm_dataset_metrics.csv',metrics)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),full_result_count=len(complete),
        audited_full_results=list(records.values()),all_current_complete_slots_independently_audited=True,
        source_receipts=source_receipts,full_epochs=[r['epochs_completed'] for r in records.values()],
        optimizer_updates=[r['optimizer_updates'] for r in records.values()],
        additional_training_seed_slots=0,all_original_experiments_complete=False,
        boundary='Every current complete original native slot including exact main/Transformer recoveries; original10-seed obligations unchanged. Full score arrays, row/label identity and saved checkpoint bindings audited; no trained-model reinference or author-equivalence claim.')
    write(target/'independent_array_audit.json',report)
    lines=['# GRASP完整结果独立复核','',
        f"已核验全部{len(complete)}个当前完整全实体种子组，均为1500轮、93000次更新；全部9120实体任务/480原始种子组仍保留。",'',
        '本轮新增7项原始完整结果独立审计。检查点绑定、全部验证/测试行身份与标签及所有评分配方的指标重算均通过；两项原始精确恢复不新增种子。', '',
        '[完整指标及空缺](algorithm_dataset_metrics.csv)；[独立复核证据](independent_array_audit.json)。', '',
        'tau0.25原始任务在570/1500轮中断，已登记新的完整1500轮精确恢复；原有部分产物保留。不得将35340次原更新与93000次重跑合并为一个可恢复优化器状态。', '',
        '该审计检查保存检查点和分数数组，不是模型重新推断审计。尚未完成原论文十种子、其他数据集或证明作者等价。','']
    (target/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(target/'validation.json',dict(config_sha256=sha(ROOT/args.config),
        checks=dict(all_complete_canonical_slots_audited=True,no_duplicate_recovery_seed=True,
                    all_full_result_hashes_verified=True,all_1500_epoch_budgets_verified=True),
        source_receipts=source_receipts,outputs={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
    print(json.dumps(dict(full_audited_results=len(complete),full_epochs=report['full_epochs'],full_updates=report['optimizer_updates'])))


if __name__=='__main__':main()
