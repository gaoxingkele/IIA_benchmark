"""Full117 classification preprocessing control fitted only on official TRAIN.

Keep the completed source-transductive cache unchanged. Reuse its complete
TRAIN arrays after checking every coordinate against the actual source. TEST
uses the TRAIN standardizer; differing native lengths are interpolated onto
the TRAIN-native grid before transformation, never a grid chosen from TEST.
This explicit control is not claimed to reproduce the historical paper.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import types
from scripts.flow_matching import prepare_moment_icm_classification_v1 as source

ROOT, read, sha, write = source.ROOT, source.read, source.sha, source.write
MODULE = 'scripts.flow_matching.prepare_moment_icm_classification_train_only_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_full117_classification_train_only.2026-10-15_v1.json'
PARENT = 'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json'
TEST = 'tests/test_fm_moment_icm_classification_train_only_v1.py'


class FittedTrainingScaler:
    """Expose fixed TRAIN statistics and reject every held-out fit attempt."""
    def __init__(self, fitted):
        self.fitted = deepcopy(fitted)
        self.test_fit_calls = 0
        self.transform_calls = 0
        self.resampled_test_grid = None

    def fit(self, *args, **kwargs):
        self.test_fit_calls += 1
        raise ValueError('TEST cannot fit or update the training standardizer')

    def transform(self, values):
        self.transform_calls += 1
        return self.fitted.transform(values)

    def __getattr__(self, name):
        return getattr(self.fitted, name)


def heldout_class(cls, fitted_training_scaler):
    """Change one fit statement in a private _read_data copy, nothing upstream."""
    original_path = Path(cls._read_data.__code__.co_filename)
    tree = ast.parse(original_path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClassificationDataset')
    node = deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_read_data'))
    changes = []
    class FixedTrainingGrid(ast.NodeTransformer):
        def visit_Expr(self, candidate):
            if ast.unparse(candidate) == 'self.scaler.fit(self.data)':
                changes.append(candidate)
                return ast.copy_location(ast.parse('self.data = _training_grid(self.data, self.scaler)').body[0], candidate)
            return self.generic_visit(candidate)
    FixedTrainingGrid().visit(node)
    if len(changes) != 1:
        raise ValueError('Exactly the original single scaler-fit statement required')
    # The new class is TEST-only, so the training branch does not need adapting.
    proxy = FittedTrainingScaler(fitted_training_scaler)
    namespace = dict(cls._read_data.__globals__)
    interpolate = namespace['interpolate_timeseries']
    def training_grid(values, scaler):
        length = scaler.n_features_in_
        if values.shape[-1] != length:
            scaler.resampled_test_grid = dict(original_test_native_length=values.shape[-1], training_native_length=length)
            return interpolate(timeseries=values, interp_length=length)
        return values
    namespace.update(StandardScaler=lambda: proxy, _training_grid=training_grid)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(original_path), 'exec'), namespace)
    controlled = type('TrainOnlyHeldoutClassificationDataset', (cls,), dict(_read_data=namespace['_read_data']))
    return controlled, proxy


def verify_training_coordinates(dataset, directory, instances):
    import numpy as np
    directory = Path(directory)
    x = np.load(directory/'values.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(directory/'input_mask.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(directory/'labels.npy', mmap_mode='r', allow_pickle=False)
    rows = np.load(directory/'original_instance_rows.npy', mmap_mode='r', allow_pickle=False)
    channels = dataset.n_channels
    if (x.shape != (instances, channels, 256) or masks.shape != (instances, 256)
            or not np.array_equal(rows, np.arange(instances)) or len(dataset) != instances*channels):
        raise ValueError('All unchanged TRAIN cases and channels must be retained')
    for index in range(len(dataset)):
        item = dataset[index]; case, channel = divmod(index, channels)
        if (not np.array_equal(x[case, channel], item.timeseries[0])
                or not np.array_equal(masks[case], item.input_mask)
                or labels[case] != int(np.asarray(item.labels).item())):
            raise ValueError('Source TRAIN coordinates,mask or label differ')
    return dict(all_original_train_coordinates_checked=True, checked_value_coordinates=instances*channels*256,
                directory=str(directory), files_sha256={p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()})


def compare_complete_heldout(current, original):
    import numpy as np
    current, original = Path(current), Path(original)
    report = {}
    for file in ['values.npy', 'input_mask.npy', 'labels.npy', 'original_instance_rows.npy']:
        a, b = np.load(current/file, mmap_mode='r', allow_pickle=False), np.load(original/file, mmap_mode='r', allow_pickle=False)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Both complete tracks require identical shape and dtype')
        changed = 0; maximum = 0.
        for start in range(0, len(a), 8):
            left, right = a[start:start+8], b[start:start+8]
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError('Full finite source and control values required')
            changed += int(np.count_nonzero(left != right))
            maximum = max(maximum, float(np.max(np.abs(left.astype('float64')-right.astype('float64')), initial=0.)))
        if file in ['labels.npy', 'original_instance_rows.npy'] and changed:
            raise ValueError('Official TEST labels and instance order must be identical')
        report[file] = dict(shape=list(a.shape), dtype=str(a.dtype), changed_coordinates=changed, maximum_absolute_difference=maximum)
    return report


def registration():
    parent = read(ROOT/PARENT); jobs = source.verify(parent)
    bindings = []
    for job in jobs:
        identity = job['dataset']; proof = Path(parent['artifact_root'])/identity/'preprocessing.json'
        guard = ROOT/parent['state_root']/'attempts'/identity/'guard_receipt.json'
        p, g = read(proof), read(guard)
        if (not p['passed'] or p['job'] != job or p['configuration_sha256'] != sha(ROOT/PARENT)
                or not g['passed'] or g['exit_code'] != 0 or g['resource_guard_aborted'] or g['proof_sha256'] != sha(proof)):
            raise ValueError('Actual successful full original117 preprocessing required')
        bindings.append(dict(dataset=identity, proof_path=str(proof), proof_sha256=sha(proof),
                             guard_path=str(guard), guard_sha256=sha(guard)))
    own = Path(__file__).resolve().relative_to(ROOT).as_posix()
    return dict(schema_version=1, original_configuration=PARENT, original_configuration_sha256=sha(ROOT/PARENT),
        paper_citation=parent['paper_citation'], data_citation='https://www.timeseriesclassification.com/',
        environment=parent['environment'], python=parent['python'], settings=parent['settings'],
        resource_lock=parent['resource_lock'], jobs=jobs, full_original_source_preprocessing_bindings=bindings,
        source_root=parent['source_root'],
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in [PARENT, own, TEST, 'scripts/flow_matching/prepare_moment_icm_classification_v1.py']],
        policy=dict(scaler_fit='official_TRAIN_only', TEST_statistic_fit=False,
                    unequal_native_lengths='source_interpolate_TEST_to_TRAIN_fitted_native_length_before_transform',
                    source_no_normalization_exceptions_retained=True, unchanged_all_TRAIN_arrays_referenced=True,
                    raw_TRAIN_TEST_cases_and_all_channels_retained=True, sequence_length=256),
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_classification_train_only_v1',
        state_root='experiments/runs/fm_moment_icm_full117_classification_train_only_v1',
        boundary='Separate control for all117 datasets,not a replacement for source TEST-fitted normalization. '
            'Preserve original exclusions,NaN interpolation,labels,all instances/channels and source resampling. '
            'Different TEST-native grids are explicitly aligned to the TRAIN grid before applying frozen statistics. '
            'No neural inference,SVM result,TAB certification or historical paper equivalence is implied.')


def verify(config):
    if config != registration():
        raise ValueError('Full117 frozen source bindings,train-only policy and unchanged resource policy required')
    return config['jobs']


def worker(config, job):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']:
        raise ValueError('Exact preprocessing interpreter and libraries required')
    parent = read(ROOT/PARENT)
    prior = Path(parent['artifact_root'])/job['dataset']
    proof = read(prior/'preprocessing.json')
    for split in ['TRAIN', 'TEST']:
        if sha(ROOT/job['splits'][split]['path']) != job['splits'][split]['sha256']:
            raise ValueError('Raw split changed')
        for name, digest in proof['records'][split]['files_sha256'].items():
            if sha(prior/split/name) != digest:
                raise ValueError('Complete original cached split changed')
    cls, small, unscaled = source.source_components(ROOT/config['source_root'])
    train = source.source_dataset(cls, ROOT/job['splits']['TRAIN']['path'], job['dataset'], small, 'train')
    training = verify_training_coordinates(train, prior/'TRAIN', job['splits']['TRAIN']['instances'])
    fixed, proxy = heldout_class(cls, train.scaler)
    heldout = source.source_dataset(fixed, ROOT/job['splits']['TEST']['path'], job['dataset'], small, 'test')
    output = Path(config['artifact_root'])/job['dataset']; output.mkdir(parents=True, exist_ok=False)
    test = source.export_split(heldout, job['splits']['TEST']['instances'], output/'TEST')
    comparison = compare_complete_heldout(output/'TEST', prior/'TEST')
    if proxy.test_fit_calls != 0:
        raise ValueError('No held-out fitting allowed')
    if job['dataset'] in unscaled and any(r['changed_coordinates'] for r in comparison.values()):
        raise ValueError('Original no-normalization exceptions must remain byte-identical')
    write(output/'preprocessing.json', dict(passed=True, job=job, configuration_sha256=sha(ROOT/CONFIG),
        environment=actual, strict_train_only_standardization=True, test_scaler_fit_calls=proxy.test_fit_calls,
        test_transform_calls=proxy.transform_calls, test_native_grid_adjustment=proxy.resampled_test_grid,
        original_no_normalization_exception=job['dataset'] in unscaled,
        unchanged_TRAIN=training, TEST=test, complete_TEST_comparison_to_source_transductive=comparison,
        original_source_preprocessing_proof_sha256=sha(prior/'preprocessing.json'),
        neural_model_training_or_inference_or_SVM_performed=False, raw_data_overwritten=False,
        historical_ICM_pipeline_equivalence_certified=False, all_original_experiments_complete=False))
    del train, heldout; gc.collect()
    for split in ['TRAIN', 'TEST']:
        if sha(ROOT/job['splits'][split]['path']) != job['splits'][split]['sha256']:
            raise ValueError('Raw data changed during preprocessing')


def controller(config, jobs):
    # Exact original metadata/resource/guard controller, bound only to the new
    # module name so its scientific child calls this separate control worker.
    fn = types.FunctionType(source.controller.__code__, dict(source.controller.__globals__, MODULE=MODULE),
                            source.controller.__name__, source.controller.__defaults__, source.controller.__closure__)
    fn(config, jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    if args.config != CONFIG:
        raise ValueError('Use the exact separate train-only control registration')
    if args.register:
        target = ROOT/CONFIG
        if target.exists():
            raise FileExistsError('Preserve frozen train-only registration')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(registration(), stream, indent=2); stream.write('\n')
        print(json.dumps(dict(configuration=CONFIG, datasets=117, original_source_track_preserved=True)))
        return
    config = read(ROOT/CONFIG); jobs = verify(config); config['_config_path'] = CONFIG
    if args.probe_only:
        print(json.dumps(dict(passed=True, datasets=len(jobs), splits=234, original_source_datasets_completed=117,
                             torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
                             new_data_workers_started=0, all_original_experiments_complete=False)))
    elif args.dataset:
        worker(config, next(j for j in jobs if j['dataset'] == args.dataset))
    else:
        controller(config, jobs)


if __name__ == '__main__':
    main()
