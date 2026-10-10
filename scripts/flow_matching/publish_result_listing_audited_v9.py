"""Merge immutable result captures without promoting diagnostics to performance."""
import argparse
import copy
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

from scripts.flow_matching import publish_result_listing_v7 as legacy

ROOT = legacy.ROOT
KEY_FIELDS = ('task', 'algorithm', 'dataset', 'protocol', 'metric',
              'epochs', 'pattern', 'missing_ratio', 'score_recipe')
NUMERIC_FIELDS = ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean',
                  'paper_std', 'paper_spread', 'evaluator_std', 'evaluator_se')


def key(row):
    return tuple(str(row.get(k) or '') for k in KEY_FIELDS)


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ('n', 'required_repeats'):
            row[field] = int(row[field])
        for field in NUMERIC_FIELDS:
            if field in row:
                row[field] = None if row[field] == '' else float(row[field])
    legacy.validate_rows(rows)
    return rows


def merge_audited_rows(base, audited):
    legacy.validate_rows(base)
    legacy.validate_rows(audited)
    positions = {key(row): i for i, row in enumerate(base)}
    result = copy.deepcopy(base)
    for row in audited:
        identity = key(row)
        if identity not in positions:
            raise ValueError('Audited replacement has no canonical slot')
        old = base[positions[identity]]
        if int(row['required_repeats']) != int(old['required_repeats']):
            raise ValueError('Frozen repeat budget changed')
        if int(row['n']) < int(old['n']):
            raise ValueError('Audited replacement loses completed repeats')
        if row['captured_utc'] <= old['captured_utc']:
            raise ValueError('Audited replacement is not newer')
        if str(row.get('author_equivalence_certified')).lower() != 'false':
            raise ValueError('Unsupported author equivalence promotion')
        result[positions[identity]].update(copy.deepcopy(row))
    legacy.validate_rows(result)
    return result


def verified_capture(directory, validation_name):
    receipt = legacy.read(directory / validation_name)
    for name, expected in receipt['outputs'].items():
        if legacy.sha(directory / name) != expected:
            raise ValueError('Captured export changed: ' + name)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = legacy.read(ROOT / args.config)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Published captures are immutable')
    base = ROOT / config['base_capture']
    audit = ROOT / config['audited_capture']
    old_receipt = verified_capture(base, 'publication_validation.json')
    verified_capture(audit, 'validation.json')
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Pinned publication source changed')
    original = load_rows(base / 'all_algorithm_dataset_metrics.csv')
    updates = load_rows(audit / 'algorithm_dataset_metrics.csv')
    if len(updates) != config['expected_replaced_metric_rows']:
        raise ValueError('Unexpected audited replacement scope')
    rows = merge_audited_rows(original, updates)
    target.mkdir(parents=True)
    legacy.csv_write(target / 'all_algorithm_dataset_metrics.csv', rows)
    for name in ('strict_algorithm_dataset.md', 'overview.md', 'original_generation_jobs.csv'):
        shutil.copyfile(base / name, target / name)
    headers = ['任务', '算法 / 消融', '数据集', '协议', '轮数', '掩码 / 缺失率',
               '推断配方', '指标', '完成/预定', '均值', 'STD', 'SE', '95%下界',
               '95%上界', '论文值', '核验时间UTC']
    values = [[r['task'], r['algorithm'], r['dataset'], r['protocol'], r.get('epochs', ''),
               str(r.get('pattern', '')) + ' / ' + str(r.get('missing_ratio', '')),
               r.get('score_recipe', ''), r['metric'], f"{r['n']}/{r['required_repeats']}",
               *[legacy.fmt(r.get(k)) for k in ('mean', 'std', 'se', 'ci95_low', 'ci95_high', 'paper_mean')],
               r['captured_utc']] for r in rows]
    (target / 'all_metrics.md').write_text(
        '# 全部算法—数据集技术指标及空缺\n\n各协议分别列出；—为缺失，0为实测零。'
        '逐行来源与核验时间见CSV。本表不代表全部实验已完成。\n\n' +
        legacy.table(headers, values) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    published = datetime.now(timezone.utc).isoformat()
    readme = (base / 'README.md').read_text(encoding='utf-8')
    readme = readme.replace('发布：' + old_receipt['published_utc'], '发布：' + published)
    # Previous README publication and its receipt differ by a few milliseconds.
    lines = readme.splitlines()
    lines[2] = ('发布：' + published + '。各轨道核验时间保留在CSV；'
                '本版合并既有快照与较新的Pi双种子独立分数审计，未重新采集其他轨道。')
    readme = '\n'.join(lines) + '\n'
    readme += ('\n## 最新Pi双种子指标\n\n'
               'PSM causal_axis_repaired/full：2/6种子；作者阈值＋PA F1 97.8869%，'
               '同阈值无PA F1 1.9109%，无PA AUROC 0.487928、AP 0.275030。'
               '保留原测试早停和训练／测试联合校准，不进入严格泛化排名。'
               '更新8条已有指标，未新增训练种子槽位。'
               '[完整审计](../pi_two_full_seed_audit_v2/README.md)。\n\n'
               '新注册LS4共轭修正与SaShiMi原始数据任务尚无本次核验的完整训练分数；'
               '其完整CPU数据审计和数学／接口诊断不列为实验性能。'
               '[LS4状态](../ls4_cauchy_corrected_resume_v3/README.md)、'
               '[SaShiMi状态](../sashimi_monash_gaussian_resume_v2/README.md)。\n')
    (target / 'README.md').write_text(readme, encoding='utf-8')
    receipt = copy.deepcopy(old_receipt)
    receipt.update(published_utc=published, base_publication_utc=old_receipt['published_utc'],
                   replaced_metric_rows=len(updates), aggregate_metric_rows=len(rows),
                   measured_metric_rows=sum(r['mean'] is not None for r in rows),
                   missing_metric_rows=sum(r['mean'] is None for r in rows),
                   task_rows=dict(Counter(r['task'] for r in rows)),
                   all_original_experiments_complete=False,
                   other_tracks_recaptured=False)
    sources = [args.config, config['base_capture'] + '/publication_validation.json',
               config['base_capture'] + '/all_algorithm_dataset_metrics.csv',
               config['audited_capture'] + '/validation.json',
               config['audited_capture'] + '/algorithm_dataset_metrics.csv']
    receipt['source_receipts'] += config['source_receipts'] + [
        dict(path=p, sha256=legacy.sha(ROOT / p)) for p in sources]
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print({k: receipt[k] for k in ('aggregate_metric_rows', 'measured_metric_rows',
                                  'missing_metric_rows', 'replaced_metric_rows')})


if __name__ == '__main__':
    main()
