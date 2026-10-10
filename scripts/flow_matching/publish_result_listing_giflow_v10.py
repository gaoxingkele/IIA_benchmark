"""Publish the first verified full GiFlow seed without recapturing other tracks."""
import argparse
import copy
from datetime import datetime, timezone
import shutil

from scripts.flow_matching import publish_result_listing_audited_v9 as previous

legacy = previous.legacy
ROOT = previous.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = legacy.read(ROOT / args.config)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve every immutable result publication')
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Pinned GiFlow result publisher changed')
    base, capture = ROOT / config['base_capture'], ROOT / config['audited_capture']
    old_receipt = previous.verified_capture(base, 'publication_validation.json')
    previous.verified_capture(capture, 'validation.json')
    previous.verified_capture(ROOT / config['full_array_audit'], 'validation.json')
    original = previous.load_rows(base / 'all_algorithm_dataset_metrics.csv')
    updates = previous.load_rows(capture / 'algorithm_dataset_metrics.csv')
    if len(updates) != config['expected_replaced_metric_rows'] or any(r['task'] != 'imputation' for r in updates):
        raise ValueError('Only the complete canonical GiFlow imputation capture required')
    rows = previous.merge_audited_rows(original, updates)
    target.mkdir(parents=True)
    legacy.csv_write(target / 'all_algorithm_dataset_metrics.csv', rows)
    for name in ('strict_algorithm_dataset.md', 'overview.md', 'original_generation_jobs.csv'):
        shutil.copyfile(base / name, target / name)
    headers = ['任务', '算法／消融', '数据集', '协议', '轮数', '掩码／缺失率', '推断配方', '指标',
               '完成／预定', '均值', 'STD', 'SE', '95%下界', '95%上界', '论文值', '核验时间UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], r.get('epochs',''),
        str(r.get('pattern',''))+' / '+str(r.get('missing_ratio','')), r.get('score_recipe',''), r['metric'],
        f"{r['n']}/{r['required_repeats']}", *[legacy.fmt(r.get(k)) for k in
        ('mean','std','se','ci95_low','ci95_high','paper_mean')], r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text('# 全部算法—数据集技术指标及空缺\n\n各协议分别列出；—为缺失，0为实测零。'
        '逐行来源及核验时间见CSV。全部原始实验仍未完成。\n\n'+legacy.table(headers,values)+'\n',encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows),encoding='utf-8')
    measured = sum(r['mean'] is not None for r in rows)
    published = datetime.now(timezone.utc).isoformat()
    lines = (base / 'README.md').read_text(encoding='utf-8').splitlines()
    lines[2] = '发布：'+published+'。本版更新 GiFlow 原始插补结果，其余轨道沿用已有快照，各自核验时间见CSV。'
    lines[4] = f'**{len(rows)}条汇总指标：{measured}条有实测值，{len(rows)-measured}条暂无完整结果。指标行数不等于训练次数。**'
    readme = '\n'.join(lines)+'\n\n## 首个完整 GiFlow 原生种子\n\n'
    readme += ('Air36／released_mirror／main／seed393：按原生早停实际完成247轮、11856次更新（预算300轮，'
        '每轮48个drop_last批次）。全部14个测试批次的已保存预测、目标和掩码位置已独立核验，'
        '完整数组重算MAE9.6546902、MSE359.7984835、MAPE21.5168332%、sqrt(native mean batch MSE)18.9683548。'
        '保持重叠窗口、不加权batch均值、测试参与验证及EMA重置行为。\n\n'
        '这是1/5种子，没有种子STD或区间；140个原始槽位目前1个完整，28个五种子组尚无全部完成组。'
        '完整数组复核不等于训练模型重推断；未声明作者等价或严格TAB结果。\n\n'
        '[完整规范槽位和指标](../giflow_canonical_listing_v2/README.md)、'
        '[独立全测试数组审计](../giflow_first_full_array_audit_v2/README.md)。\n\n'
        'LS4 PhysioNet 已接入原生500轮训练和50次全测试评测，10个任务共750000次预定更新；'
        '完整CPU批量诊断达到内存上限的失败已保留，后继队列执行完整GPU容量验证，'
        '尚无该队列的完整500轮训练分数。'
        '[执行状态及协议限制](../ls4_native_physionet_execution_v2/README.md)。\n')
    (target / 'README.md').write_text(readme,encoding='utf-8')
    receipt = copy.deepcopy(old_receipt)
    receipt.update(published_utc=published,base_publication_utc=old_receipt['published_utc'],
        replaced_metric_rows=len(updates), aggregate_metric_rows=len(rows), measured_metric_rows=measured,
        missing_metric_rows=len(rows)-measured, newly_measured_metric_rows=measured-old_receipt['measured_metric_rows'],
        all_original_experiments_complete=False,other_tracks_recaptured=False)
    receipt['source_receipts'] += config['source_receipts'] + [dict(path=args.config,sha256=legacy.sha(ROOT/args.config))]
    receipt['outputs'] = {p.name:legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target/'publication_validation.json',receipt)
    print({k:receipt[k] for k in ('aggregate_metric_rows','measured_metric_rows','missing_metric_rows','newly_measured_metric_rows')})


if __name__ == '__main__':
    main()
