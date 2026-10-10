"""Publish completed GiFlow corrected replay while preserving every metric byte."""
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
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full verification/publication evidence changed')
    proof = read(ROOT / config['independent_full_replay_verification'])
    if (proof['full_model_replay_and_independent_saved_arrays_verified'] is not True
            or proof['coverage']['full_test_windows'] != 1747
            or proof['coverage']['full_masked_test_targets'] != 268613
            or proof['coverage']['observed_forward_calls'] != 280
            or proof['additional_training_seed_slots'] != 0
            or proof['training_seeds_completed'] != 1 or proof['training_seeds_required'] != 5
            or proof['author_equivalence_certified']):
        raise ValueError('Actual complete original checkpoint replay/coverage required')
    base = ROOT / config['base_publication']
    previous = read(base / 'publication_validation.json')
    for name, digest in previous['outputs'].items():
        if sha(base / name) != digest:
            raise ValueError('Previous immutable metric publication changed')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 7283 or sum(bool(r['mean']) for r in rows) != 1790:
        raise ValueError('Preserve all registered and measured metric rows')
    selected = [r for r in rows if r['algorithm'] == 'GiFlow/main'
                and r['protocol'] == 'giflow_native/reviewed_corrected' and r['dataset'] == 'air36']
    if len(selected) != 4 or any(int(r['n']) != 1 or int(r['required_repeats']) != 5
                               or float(r['mean']) != proof['original_metrics'][r['metric']] for r in selected):
        raise ValueError('Original four native metrics and five-seed denominator required')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve every immutable publication')
    target.mkdir(parents=True)
    for name in previous['outputs']:
        if name not in ['README.rst', 'README.md']:
            shutil.copyfile(base / name, target / name)
            if sha(target / name) != sha(base / name):
                raise ValueError('Metric table must remain byte-identical')
    now = datetime.now(timezone.utc).isoformat()
    (target / 'README.rst').write_text(
        '完整算法—数据集指标 v20\n==========================\n\n'
        f'发布UTC：{now}。继承v19的7283项指标：1790项有实测值，5493项缺完整结果，141个算法／配置名称。'
        '全部数值表和采集时间逐字节保留。本轮新增完整模型核验，不增加成绩、训练种子或置信区间。\n\n'
        '全部指标见results.html、all_algorithm_dataset_metrics.csv、all_metrics.md。'
        '数值行中的旧核验状态描述保留为原采集快照；当前GiFlow核验以本README和publication_validation.json的证据链接为准。\n\n'
        'GiFlow reviewed_corrected/Air36/seed393：原生验证选择检查点恢复后，全部1747测试窗口、'
        '268613缺失目标、14批次、20步Euler、280次前向重推断完成。独立核验所有保存数组与原数据、'
        '掩码、图、缩放器和划分；在原冻结CPU/GPU容差内通过。原生MAE10.951781375、MSE413.131974901、'
        'MAPE23.823091711%、sqrt(mean_batch_MSE)20.325648204维持原值。\n\n'
        '证据：../giflow_corrected_full_replay_verified_v1/independent_full_replay_verification.json；'
        '完整模型原始证据：../giflow_corrected_checkpoint_replay_v1/reviewed_corrected_checkpoint_replay_audit.json。'
        '训练仍为1/5种子，原生重叠窗口和批次均值指标口径保留；不声称作者等价或完整TAB认证。\n\n'
        '45篇论文、681项方法/基线/消融、273项原始任务及采用基线的全部范围仍未完成。'
        'TimeGAN、GT-GAN和其他未完成任务继续保留，不以预处理、检查点加载或重推断替代训练成绩。\n',
        encoding='utf-8', newline='\n')
    validation = copy.deepcopy(previous)
    validation.update(published_utc=now, base_publication_utc=previous['published_utc'],
                      newly_measured_metric_rows=0, additional_training_seed_slots=0,
                      performance_rows_changed=False, all_numeric_tables_byte_identical_to_base=True,
                      giflow_corrected_full_checkpoint_replay_complete=True,
                      giflow_full_native_models_replayed=2,
                      giflow_corrected_saved_replay_arrays_independently_verified=True,
                      giflow_corrected_full_replay_evidence=config['independent_full_replay_verification'],
                      all_original_experiments_complete=False)
    validation['source_receipts'] += config['source_receipts'] + [dict(path=args.config, sha256=sha(ROOT / args.config))]
    validation['outputs'] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    write(target / 'publication_validation.json', validation)
    print(dict(rows=7283, measured=1790, new_performance_rows=0, corrected_full_replay_verified=True))


if __name__ == '__main__':
    main()
