"""Publish complete GiFlow replay status without changing any performance table."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import shutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    for receipt in config['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen completed-model publication evidence changed')
    proof = read(ROOT / config['full_model_union'])
    if (not proof['all_completed_canonical_models_replayed_and_independently_verified']
            or proof['original_canonical_slots'] != 140 or proof['completed_canonical_models'] != 3
            or proof['full_scoring_batches'] != 42 or proof['full_native_forward_calls'] != 840
            or proof['released_training_seeds_completed'] != 2 or proof['released_training_seeds_required'] != 5
            or proof['corrected_training_seeds_completed'] != 1 or proof['corrected_training_seeds_required'] != 5
            or proof['additional_training_seed_slots'] != 0 or proof['author_equivalence_certified']
            or proof['strict_TAB_result']):
        raise ValueError('Exact original completed three-model coverage without extra seeds required')
    base = ROOT / config['base_publication']
    previous = read(base / 'publication_validation.json')
    for name, digest in previous['outputs'].items():
        if sha(base / name) != digest:
            raise ValueError('Previous immutable metric publication changed')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != previous['aggregate_metric_rows'] or sum(bool(row['mean']) for row in rows) != previous['measured_metric_rows']:
        raise ValueError('Original registered metric and measured-row scope must remain unchanged')
    for track, expected in [('released_mirror', 2), ('reviewed_corrected', 1)]:
        selected = [row for row in rows if row['algorithm'] == 'GiFlow/main' and row['dataset'] == 'air36'
                    and row['protocol'] == 'giflow_native/' + track]
        if len(selected) != 4 or any(int(row['n']) != expected or int(row['required_repeats']) != 5 for row in selected):
            raise ValueError('Two released and one corrected seed must retain separate five-seed budgets')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Keep all numerical snapshots immutable')
    target.mkdir(parents=True)
    copied = []
    for name in previous['outputs']:
        if name not in ['README.rst', 'README.md']:
            shutil.copyfile(base / name, target / name)
            if sha(base / name) != sha(target / name):
                raise ValueError('Every numerical table and original row timestamp must be byte-identical')
            copied.append(name)
    captured = datetime.now(timezone.utc).isoformat()
    (target / 'README.rst').write_text('完整算法—数据集指标 v24\n==========================\n\n'
        '%d项指标；%d项有实测值；%d项缺结果。本次只补齐GiFlow完整模型核验状态，全部数值表及行采集时间逐字节保留v23。\n\n'
        % (len(rows), previous['measured_metric_rows'], previous['missing_metric_rows'])
        + 'GiFlow已有的三份规范模型现在全部纳入同一独立证据链：released393、released394和reviewed_corrected393。'
        '逐一核验原始训练早停历史、实际优化更新、评估权重、源文件、掩码、图、缩放器和划分，重新校验全部42批原始与重推断数组。'
        '每份1747个测试窗口、14批、20步Euler，共840次已有原生前向；seed394有264837个目标，其余两份各268613个目标。'
        '所有完整分数及原生无权重批均值指标均在原冻结容差内通过，非逐点位相同预测。\n\n'
        '只核验已完成模型，没有新增训练或种子：released仍为2/5种子，reviewed_corrected仍为1/5种子；'
        'GiFlow原始140项任务中完成3项。released测试集进入验证、每训练批重置EMA及原生重叠窗口指标限制继续保留。'
        '平方根批均值MSE不是全局目标加权RMSE。\n\n'
        'CSV中的旧核验描述为原采集快照；当前核验状态以本README、publication_validation.json及完整模型证据为准。'
        '全部45篇论文、681项方法基线消融、273项原始数据任务及采用基线仍未完成，作者等价与严格TAB认证未取得。\n\n'
        '证据：../giflow_all_completed_models_verified_v1/independent_all3_full_replay_union.json。'
        '完整表：results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。\n', encoding='utf-8', newline='\n')
    receipt = copy.deepcopy(previous)
    receipt.update(published_utc=captured, base_publication_utc=previous['published_utc'],
        newly_measured_metric_rows=0, replaced_metric_rows=0, changed_numeric_metric_rows=0,
        performance_rows_changed=False, all_numeric_tables_byte_identical_to_base=True,
        newly_completed_canonical_training_slots=0, additional_training_seed_slots=0,
        giflow_full_native_models_replayed=3, giflow_independent_model_reinference_complete=True,
        giflow_seed394_full_model_replay_complete=True, giflow_all_completed_full_models_independently_verified=True,
        giflow_all_completed_full_model_evidence=config['full_model_union'],
        all_original_experiments_complete=False)
    receipt['source_receipts'] += config['source_receipts'] + [dict(path=args.config, sha256=sha(ROOT / args.config))]
    receipt['outputs'] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    write(target / 'publication_validation.json', receipt)
    print(dict(rows=len(rows), measured=previous['measured_metric_rows'], full_completed_giflow_models_verified=3,
               numeric_exports_byte_identical=len(copied), additional_training_seeds=0, all_original_experiments_complete=False))


if __name__ == '__main__':
    main()
