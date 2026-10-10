"""Extend frozen publication with canonical GiFlow imputation, not TSAD rows."""
import argparse
import copy
import json
from pathlib import Path
import types

from scripts.flow_matching import publish_result_listing_v7 as legacy
from scripts.flow_matching.giflow_native_protocol import ROOT, sha


def merge_rows(report, base, cfm, cfm_directory, grasp, native):
    # Older native tables called every native task anomaly detection. Remove
    # those GiFlow entries before adding the strictly verified imputation rows.
    report = copy.deepcopy(report)
    report['native_summary'] = [r for r in report['native_summary'] if r['algorithm'] != 'giflow']
    rows = legacy.captured_rows(report,base,cfm,cfm_directory,grasp)
    extra = native['metric_rows']
    if any(r['task'] != 'imputation' or not r['protocol'].startswith('giflow_native/') for r in extra):
        raise ValueError('GiFlow native publication task/protocol changed')
    rows.extend(copy.deepcopy(extra))
    legacy.validate_rows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=legacy.read(ROOT/args.config)
    for item in config['extension_source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Pinned native publication extension changed')
    directory=ROOT/config['giflow_capture']
    validation=legacy.read(directory/'validation.json')
    for name,value in validation['outputs'].items():
        if sha(directory/name) != value:
            raise ValueError('GiFlow canonical capture changed')
    native=legacy.read(directory/'execution_audit.json')
    for item in native['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('GiFlow captured source binding changed')
    if native['canonical_slots'] != 140 or native['seed_groups'] != 28:
        raise ValueError('GiFlow full canonical scope missing')
    namespace=dict(vars(legacy))
    namespace['captured_rows']=lambda report,base,cfm,cfm_directory,grasp:merge_rows(
        report,base,cfm,cfm_directory,grasp,native)
    types.FunctionType(legacy.main.__code__,namespace)()
    target=ROOT/config['output_directory']
    note=('\n## GiFlow 原始插补任务\n\n'
          '本版本纳入 GiFlow 的 140 个原始种子槽位、28 个五种子组和 112 条插补指标及空缺。'
          '修正观察计数的重试不增加种子；未完成或缺失实际更新证据的旧结果不填数值。'
          '作者镜像的测试参与验证分支与修正的验证检查点分支分别保留。'
          'MAE、MSE、MAPE（百分数）和 sqrt(native mean batch MSE) 保持原代码单位；'
          '这些指标来自重叠测试窗口、batch 均值的不加权平均，不能当作异常检测 F1。'
          '训练模型独立重推断未完成。\n')
    with (target/'README.md').open('a',encoding='utf-8') as stream:stream.write(note)
    receipt=legacy.read(target/'publication_validation.json')
    receipt.update(giflow_canonical_slots=140,giflow_five_seed_groups=28,
        giflow_complete_slots=native['job_counts'].get('completed',0),
        giflow_independent_model_reinference_complete=False)
    paths=[str(p.relative_to(ROOT)).replace('\\','/') for p in directory.iterdir() if p.is_file()]
    receipt['source_receipts'] += config['extension_source_receipts'] + [dict(path=p,sha256=sha(ROOT/p)) for p in paths]
    receipt['outputs']={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='publication_validation.json'}
    legacy.write(target/'publication_validation.json',receipt)


if __name__=='__main__':
    main()
