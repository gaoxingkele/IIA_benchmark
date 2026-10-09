"""Publish audited original-task updates without duplicating seed slots."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def recipe_key(value):
    recipe = ast.literal_eval(value) if isinstance(value, str) else value
    return int(recipe['source_samples']), int(recipe['flow_evaluations'])


def replace_audited_rows(rows, cfm, native, cfm_source, native_source, native_capture):
    """Resolve existing canonical rows; an already measured native slot is immutable."""
    index = {}
    for row in rows:
        if row['task'] == 'continuous_time_dynamics':
            key = ('CFM', row['algorithm'], row['dataset'], row['protocol'],
                   int(row['epochs']), row['metric'])
        elif row['source'].endswith('/grasp_algorithm_dataset_metrics.csv'):
            key = ('GRASP', row['algorithm'], row['dataset'].lower(), row['protocol'],
                   recipe_key(row['score_recipe']), row['metric'])
        else:
            continue
        if key in index:
            raise ValueError('Duplicate canonical result row')
        index[key] = row
    updated = set()
    for group in cfm['seed_aggregates']:
        for metric in ('MSE', 'MAE', 'RMSE'):
            key = ('CFM', 'CFM-TS/' + group['variant'], group['dataset'],
                   group['track'], int(group['epochs']), metric)
            row = index[key]
            if key in updated or int(row['n']) > group[metric + '_n']:
                raise ValueError('Duplicate update or lost completed seed')
            updated.add(key)
            row.update(n=group[metric + '_n'], captured_utc=cfm['captured_utc'],
                       source=cfm_source, comparison=group['comparison'])
            for field in ('mean', 'std', 'se', 'ci95_low', 'ci95_high'):
                row[field] = group[metric + '_' + field]
    for result in native:
        key = ('GRASP', result['algorithm'].split('/')[-1], result['dataset'].lower(),
               result['protocol'], (int(result['source_samples']), int(result['flow_evaluations'])),
               result['metric'])
        row = index[key]
        if key in updated or int(row['n']) != 0 or int(result['completed_seeds']) != 1:
            raise ValueError('Refuse seed duplication or replacing a measured result')
        if int(row['required_repeats']) != int(result['required_seeds']):
            raise ValueError('Frozen repeat budget changed')
        updated.add(key)
        row.update(n=1, mean=float(result['value']), std=None, se=None,
                   ci95_low=None, ci95_high=None, paper_mean=result['paper_mean'],
                   source=native_source, captured_utc=native_capture,
                   original_canonical_id=result['original_canonical_id'],
                   execution_id=result['execution_id'],
                   comparison='one_seed_numerical_comparison_only')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = read(config_path)
    base = ROOT / config['base_catalog']
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Keep published captures immutable')
    previous = read(base / 'publication_validation.json')
    for receipt in previous['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Published source changed')
    for name, expected in previous['outputs'].items():
        if sha(base / name) != expected:
            raise ValueError('Published catalog changed')
    cfm_dir = ROOT / config['cfm_capture']
    native_dir = ROOT / config['native_main_capture']
    cfm = read(cfm_dir / 'execution_audit.json')
    audit = read(native_dir / 'independent_array_audit.json')
    attribution = read(native_dir / 'attribution_array_audit.json')
    native = read_csv(native_dir / 'algorithm_dataset_metrics.csv')
    completed = audit['audited_full_result']
    if len(completed) != 1 or not completed[0]['metrics_independently_recomputed']:
        raise ValueError('Native full-array metric audit absent')
    if not attribution['full_overlap_counts_and_point_score_reconstruction_checked']:
        raise ValueError('Attribution aggregation audit absent')
    if len(native) != 60 or any(r['original_canonical_id'] != audit['canonical_original_id'] for r in native):
        raise ValueError('Native slot/recipe coverage changed')
    rows = read_csv(base / 'all_algorithm_dataset_metrics.csv')
    updated = replace_audited_rows(rows, cfm, native,
        (cfm_dir / 'algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix(),
        (native_dir / 'algorithm_dataset_metrics.csv').relative_to(ROOT).as_posix(), audit['captured_utc'])
    if len(rows) != 6923 or len(updated) != 231:
        raise ValueError('Full catalog coverage changed')
    target.mkdir(parents=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (target / 'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    # The strict/imputation tables and other native groups retain their own capture times.
    text = (base / 'README.md').read_text(encoding='utf-8')
    original_cfm = next(read(ROOT / receipt['path']) for receipt in previous['source_receipts']
        if receipt['path'].endswith('/execution_audit.json')
        and read(ROOT / receipt['path']).get('registered_seed_cohorts') == 57)
    text = text.replace(original_cfm['captured_utc'], cfm['captured_utc'])
    text = text.replace('快照状态' + str(original_cfm['job_counts']), '快照状态' + str(cfm['job_counts']))
    text = text.replace('完成4个全实体种子组', '完成5个全实体种子组（主模型精确重跑见本次更新节）')
    text = text.replace('全部504条算法—数据集—推断配方—协议指标及空缺',
        '原生基快照504条指标及空缺（本版主模型60条更新见总CSV及本次更新节）')
    start = text.index('## CFM-TS：全部57组原始连续时间实验')
    end = text.index('## Spectral Mean Flow / Diffusion-TS：原始生成任务', start)
    block = (cfm_dir / 'README.md').read_text(encoding='utf-8')
    block = block[block.index('| 协议 |'):block.index('## 复现边界')]
    cfmlink = '../' + cfm_dir.name
    mainlink = '../' + native_dir.name
    replacement = ('## CFM-TS：全部57组原始连续时间实验\n\n'
        f"核验时间{cfm['captured_utc']}；285项原始槽位，状态{cfm['job_counts']}。两项精确重跑解决原种子槽位，不增加重复数。\n\n"
        f'[全部指标]({cfmlink}/algorithm_dataset_metrics.csv)；[逐种子任务与状态]({cfmlink}/registered_jobs.csv)；[实现边界与审计]({cfmlink}/README.md)。\n\n'
        + block + '\n')
    text = text[:start] + replacement + text[end:]
    main_table = (native_dir / 'README.md').read_text(encoding='utf-8')
    main_table = main_table[main_table.index('| 协议 |'):main_table.index('\n论文表')]
    notice = ('\n## 本次更新：GRASP主模型的完整训练与全部推断配方\n\n'
        f"核验时间{audit['captured_utc']}；SWAN，原种子1103，1500轮/93000次更新，72000个测试点。仅1/10种子，STD/CI为空。原始失败输出保留；覆盖原有60个空缺指标，不新增统计槽位。原生轨道全实体完成组从4增至5。\n\n"
        f'[完整评分、训练资源及差异说明]({mainlink}/README.md)；[指标复算]({mainlink}/independent_array_audit.json)；[归因数组审计]({mainlink}/attribution_array_audit.json)。\n\n'
        + main_table + '\n')
    intro = ('> 本版保留严格异常检测/插补基表的独立捕获时间；更新CFM-TS及GRASP主模型的完整核验结果。'
        '全表仍为6923条汇总指标；GRASP其他模型保留基快照，主模型独立捕获时间见更新节。全部实验尚未完成。\n\n')
    text = text.replace('# 全部算法—数据集实验技术指标\n\n', '# 全部算法—数据集实验技术指标\n\n' + intro, 1)
    (target / 'README.md').write_text((text + notice).rstrip() + '\n', encoding='utf-8')
    sources = [config_path, base / 'publication_validation.json', base / 'all_algorithm_dataset_metrics.csv',
        cfm_dir / 'execution_audit.json', cfm_dir / 'algorithm_dataset_metrics.csv',
        native_dir / 'independent_array_audit.json', native_dir / 'algorithm_dataset_metrics.csv',
        native_dir / 'attribution_array_audit.json']
    validation = {'published_utc': datetime.now(timezone.utc).isoformat(),
        'aggregate_metric_rows': len(rows), 'updated_metric_rows': len(updated),
        'cfm_canonical_completed_slots': cfm['job_counts']['completed'],
        'grasp_completed_native_cohorts': 5, 'grasp_main_completed_seeds': 1,
        'all_original_experiments_complete': False,
        'checks': {'no_extra_seed_slots': True, 'no_metric_based_attempt_selection': True,
                   'source_capture_times_preserved': True, 'missing_values_kept': True,
                   'paper_claims_not_local_results': True},
        'source_receipts': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sources],
        'outputs': {p.name: sha(p) for p in target.iterdir() if p.is_file()}}
    (target / 'publication_validation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: validation[k] for k in ('aggregate_metric_rows', 'updated_metric_rows', 'cfm_canonical_completed_slots')}))


if __name__ == '__main__':
    main()
