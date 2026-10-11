"""Preserve original TSAD splits and repair strictly constant fit-channel scaling.

Original inputs, queues, sources and running workers are never modified.
Only truly constant fit channels use their exact first value and unit scale;
nonconstant channels retain every original mean/std operation without epsilon.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType

from scripts.flow_matching import prepare_tsad_execution as original
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api

ROOT = original.ROOT


def constant_scaling_ast(source):
    tree = ast.parse(source)
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_and_normalize'))
    replaced = 0
    for index, item in enumerate(node.body):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id == 'safe':
            replacement = ast.parse("""
constant = np.ptp(arrays['train'], axis=0) == 0
legacy_std, legacy_mean = std.copy(), mean.copy()
mean = np.where(constant, arrays['train'][0], mean)
std = np.where(constant, 0., std)
safe = np.where(std > 0, std, 1.)
""").body
            node.body[index:index + 1] = replacement
            replaced += 1
            break
    if replaced != 1:
        raise ValueError('Original full fit scaling boundary changed')
    index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Return))
    node.body[index:index] = ast.parse("""
audit.update(legacy_mean=legacy_mean.tolist(), legacy_std=legacy_std.tolist(),
    changed_constant_channels=np.flatnonzero(constant & (legacy_std > 0)).tolist(),
    exact_constant_fit_channels=np.flatnonzero(constant).tolist(),
    normalization_variant='constant_fit_v2_exact_first_value_unit_scale_no_test_fit')
""").body
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


namespace = dict(vars(original))
exec(compile(constant_scaling_ast(Path(original.__file__).read_text('utf-8')), __file__, 'exec'), namespace)
split_and_normalize = namespace['split_and_normalize']
prepare = FunctionType(original.prepare.__code__, namespace)


def verify(config):
    for receipt in config['source_receipts']:
        if original.sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen constant-fit preprocessing source changed')
    before = json.loads((ROOT / config['original_settings']).read_text('utf-8'))
    after = config['preparation_settings']
    unchanged = set(before) - {'id', 'recipe', 'prepared_root'}
    if any(after[key] != before[key] for key in unchanged):
        raise ValueError('Preserve all datasets, split budgets and method obligations')
    for key, value in before['recipe'].items():
        if key != 'scaler' and after['recipe'].get(key) != value:
            raise ValueError('Only the explicit fit-only constant scaling rule may differ')
    if after['prepared_root'] == before['prepared_root'] or len(after['datasets']) != 7:
        raise ValueError('Full original seven datasets and fresh prepared version required')


def worker(config):
    before = json.loads((ROOT / config['original_manifest']).read_text('utf-8'))
    for dataset in before['datasets']:
        if original.sha(ROOT / dataset['input_npz']) != dataset['input_sha256']:
            raise ValueError('Original full prepared input changed')
        for source in dataset['source_files']:
            if original.sha(ROOT / source['path']) != source['sha256']:
                raise ValueError('Original full raw source changed')
    after = prepare(ROOT, config['preparation_settings'])
    evidence = []
    import numpy as np
    for old, new in zip(before['datasets'], after['datasets']):
        if old['id'] != new['id'] or old['source_files'] != new['source_files']:
            raise ValueError('All original raw sources and dataset identity must remain')
        if len(old['entities']) != len(new['entities']):
            raise ValueError('Every original entity required')
        changed = []
        with np.load(ROOT / old['input_npz'], allow_pickle=False) as previous, np.load(ROOT / new['input_npz'], allow_pickle=False) as current:
            if set(previous.files) != set(current.files):
                raise ValueError('Original complete arrays required')
            for old_entity, entity in zip(old['entities'], new['entities']):
                keys = ('entity_id', 'array_prefix', 'fit_interval', 'purged_interval', 'validation_interval', 'test_points', 'features', 'repairs')
                if any(old_entity[k] != entity[k] for k in keys):
                    raise ValueError('Full original grouped splits and cleaning must remain')
                prefix = entity['array_prefix']
                constant = entity['exact_constant_fit_channels']
                varying = [i for i in range(entity['features']) if i not in constant]
                for split in ('train', 'validation', 'test'):
                    key = prefix + '_' + split
                    if previous[key].shape != current[key].shape or not np.isfinite(current[key]).all():
                        raise ValueError('Full finite original input shape required')
                    if not np.array_equal(previous[key][:, varying], current[key][:, varying]):
                        raise ValueError('Nonconstant input channels changed')
                    if split == 'train' and np.any(current[key][:, constant] != 0):
                        raise ValueError('Exact constant fit channels must be exactly zero')
                if not np.array_equal(previous[prefix + '_labels'], current[prefix + '_labels']):
                    raise ValueError('Original full test labels changed')
                if entity['changed_constant_channels']:
                    changed.append(dict(entity=entity['entity_id'], channels=entity['changed_constant_channels'],
                        legacy_std=[entity['legacy_std'][i] for i in entity['changed_constant_channels']],
                        full_train_points=len(current[prefix + '_train']), full_validation_points=len(current[prefix + '_validation']),
                        full_test_points=len(current[prefix + '_test']),
                        old_test_max_abs=float(np.max(np.abs(previous[prefix + '_test']))),
                        corrected_test_max_abs=float(np.max(np.abs(current[prefix + '_test'])))))
        evidence.append(dict(dataset=old['id'], full_entities=len(new['entities']),
            full_test_points=sum(e['test_points'] for e in new['entities']), original_input_sha256=old['input_sha256'],
            corrected_input_sha256=new['input_sha256'], all_original_grouped_splits_labels_and_source_files_equal=True,
            nonconstant_channels_bitwise_equal=True, changed_constant_fit_entities=changed))
        # Recheck after writing the new version; preserve the original snapshot.
        if original.sha(ROOT / old['input_npz']) != old['input_sha256']:
            raise ValueError('Original prepared archive was modified')
    manifest = ROOT / config['preparation_settings']['prepared_root'] / 'manifest.json'
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), full_original_datasets_verified=7,
        source_configuration=config['configuration_path'], original_manifest_sha256=original.sha(ROOT / config['original_manifest']),
        corrected_manifest=manifest.relative_to(ROOT).as_posix(), corrected_manifest_sha256=original.sha(manifest),
        actual_corrected_prepared_root=str(manifest.parent.resolve()), datasets=evidence,
        full_recomparison_training_still_required=True, new_completed_training_slots=0, author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False,
        boundary='Full original seven-dataset raw preprocessing completed. Exact fit constants anchored without epsilon; original nonconstant arrays, grouped splits, labels, original data and queues preserved. Requires separate complete training comparison for every affected original model/ablation/seed.')
    output = ROOT / config['report_directory'] / 'full_preparation_and_data_differential.json'
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(full_original_datasets_verified=7, affected_datasets=[d['dataset'] for d in evidence if d['changed_constant_fit_entities']]),ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text('utf-8'))
    verify(config)
    if args.worker:
        return worker(config)
    api = resource_api()
    base = ROOT / config['controller_directory']
    base.mkdir(parents=True, exist_ok=True)
    status = dict(pid=os.getpid(), state='starting', active_pid=None, config_sha256=original.sha(ROOT / args.config))
    def update(values):
        status.update(values)
        original.write_json(base / 'status.json', status)
    with api['resource_slot'](ROOT / config['readonly_resource_lock'], config['settings'], update):
        command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.prepare_tsad_constant_fit_v2', '--config', args.config, '--worker']
        with (base / 'worker.stdout.log').open('xb') as stdout, (base / 'worker.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            update(dict(state='preparing_all_original_full_datasets', active_pid=child.pid))
            code, usage = capped_wait(child, config, api, update)
        update(dict(state='completed' if code == 0 else 'failed_preserved', active_pid=None, exit_code=code, resources=usage))
        if code:
            raise RuntimeError('Full constant-fit preprocessing failed; preserve every attempt')


if __name__ == '__main__':
    main()
