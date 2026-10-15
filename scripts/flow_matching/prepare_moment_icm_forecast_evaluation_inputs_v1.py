"""Full six-dataset downstream forecast inputs,all three reported horizons.

Keep source fixed ETT months and other-dataset70/10/20 splits distinct from
the60/10/30 pretraining caches. Store exact TC matrices plus row indices;
never expand or truncate the complete overlapping forecast-window population.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from scripts.flow_matching import prepare_moment_icm_PILE_long_forecast_v1 as original

ROOT, read, sha, write = original.ROOT, original.read, original.sha, original.write
MODULE = 'scripts.flow_matching.prepare_moment_icm_forecast_evaluation_inputs_v1'
CONFIG = 'configs/reproducibility/fm_moment_ICM_all6_forecast_inputs_all18_horizons.2026-10-15_v1.json'
SCOPE = 'configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json'
TEST = 'tests/test_fm_moment_icm_forecast_evaluation_inputs_v1.py'
RECIPE = 'experiments/runs/mtsad_protocol_audit/sources/moment_research/original/configs/forecasting/linear_probing.yaml'
REPORT = 'complete_forecast_input_audit.json'
HORIZONS = [96, 192, 384]


def source_dataset(cls, path, split, horizon):
    if split not in ['train', 'val', 'test'] or horizon not in HORIZONS:
        raise ValueError('All three original horizons and chronological roles required')
    return cls(seq_len=256, forecast_horizon=horizon, full_file_path_and_name=Path(path).as_posix(),
        data_split=split, target_col='OT', scale=True, data_stride_len=1,
        task_name='long-horizon-forecasting', train_ratio=.7, val_ratio=.1, test_ratio=.2,
        output_type='multivariate', random_seed=13)


def export_dataset(cls, raw, output):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    roles = {}; all_targets = {}; fitted_stats = None
    for role in ['train', 'val', 'test']:
        data = source_dataset(cls, raw, role, HORIZONS[0])
        matrix = data.data
        if (matrix.ndim != 2 or matrix.shape != (data.length_timeseries, data.n_channels)
                or not np.isfinite(matrix).all()):
            raise ValueError('Entire finite source time-channel matrix required')
        borders = data._get_borders(); source_slice = getattr(borders, role)
        rows = np.fromiter(range(*source_slice.indices(data.length_timeseries_original)), dtype=np.int64)
        if len(rows) != len(matrix) or not np.array_equal(np.diff(rows), np.ones(len(rows)-1, dtype=np.int64)):
            raise ValueError('Complete contiguous source CSV row mapping required')
        target = output/role; target.mkdir()
        values = original.save_source_array(target/'values_TC.npy', matrix)
        row_proof = original.save_source_array(target/'original_CSV_rows.npy', rows)
        mask = data[0].input_mask
        if mask.shape != (256,) or not np.array_equal(mask, np.ones(256)):
            raise ValueError('Exact full source256observed input mask required')
        mask_proof = original.save_source_array(target/'input_mask_window.npy', mask)
        stats = dict(mean=data.scaler.mean_.tolist(), var=data.scaler.var_.tolist(), scale=data.scaler.scale_.tolist(),
                     train_rows=int(data.scaler.n_samples_seen_))
        if fitted_stats is None: fitted_stats = stats
        elif fitted_stats != stats: raise ValueError('Same training-only fitted scaler required for all roles')
        ranges = []
        for horizon in HORIZONS:
            data.forecast_horizon = horizon
            windows = len(data)
            if windows <= 0 or windows != len(matrix)-256-horizon+1:
                raise ValueError('Complete original stride1forecast window count required')
            # Check every start/target bound analytically,not only three samples.
            starts = np.arange(windows, dtype=np.int64)
            target_starts = starts+256; target_ends = target_starts+horizon
            if (int(starts.min()) != 0 or int(starts.max()) != windows-1
                    or np.any(target_ends > len(matrix)) or np.any(target_starts != starts+256)
                    or int(target_ends.max()) != len(matrix)):
                raise ValueError('All histories and complete horizon targets must be in bounds')
            checked = sorted({0, windows//2, windows-1})
            for index in checked:
                item = data[index]
                if (item.timeseries.shape != (data.n_channels, 256)
                        or item.forecast.shape != (data.n_channels, horizon)
                        or item.timeseries.tobytes() != matrix[index:index+256].T.tobytes()
                        or item.forecast.tobytes() != matrix[index+256:index+256+horizon].T.tobytes()
                        or not np.array_equal(item.input_mask, mask)):
                    raise ValueError('Pinned actual source input/forecast axes and endpoints must match')
            first_target, final_target = int(rows[256]), int(rows[-1])+1
            # val/test history overlaps the prior role;their forecast targets do not.
            boundary = borders.train.stop if role == 'val' else borders.val.stop
            if role in ['val', 'test'] and first_target != boundary:
                raise ValueError('Heldout forecast targets cannot enter the preceding split')
            ranges.append(dict(horizon=horizon, windows=windows, input_coordinates=windows*data.n_channels*256,
                target_coordinates=windows*data.n_channels*horizon, source_target_row_start=first_target,
                source_target_row_end_exclusive=final_target, every_window_bound_checked=True,
                actual_source_windows_value_checked_at=checked, every_window_value_independently_rebuilt=False))
            all_targets[role, horizon] = (first_target, final_target)
        roles[role] = dict(time_rows=len(matrix), channels=data.n_channels, original_CSV_rows=data.length_timeseries_original,
            raw_time_start=int(rows[0]), raw_time_end_exclusive=int(rows[-1])+1,
            files=dict(values_TC=values, original_CSV_rows=row_proof, input_mask_window=mask_proof),
            windows_by_horizon=ranges, training_scaler=stats,
            all_TC_coordinates_saved_byte_exact=True, repeated_window_tensor_materialized=False)
        del data, matrix; gc.collect()
    for horizon in HORIZONS:
        if (all_targets['train', horizon][1] > all_targets['val', horizon][0]
                or all_targets['val', horizon][1] > all_targets['test', horizon][0]):
            raise ValueError('Forecast target roles must be chronologically separate')
    original_rows = roles['train']['original_CSV_rows']
    selected_end = roles['test']['raw_time_end_exclusive']
    return dict(roles=roles, total_source_raw_rows=original_rows, selected_source_time_end_exclusive=selected_end,
        original_rows_outside_source_fixed_month_selection=original_rows-selected_end,
        all_raw_rows_preserved_on_disk=True, training_scaler_fixed_across_all_roles=True,
        full_window_populations_and_horizons_retained=True, forecast_target_roles_overlap=False,
        validation_and_test_include256historical_context_rows=True, original_cubic_interpolation_precedes_split=True,
        strict_temporal_independence_of_interpolation_certified=False)


def registration():
    scope = read(ROOT/SCOPE)
    source = read(ROOT/'configs/reproducibility/fm_moment_ICM_full117_classification_preprocessing.2026-10-15_v1.json')
    raw_audit = read(ROOT/source['raw_audit'])
    records = {r['dataset']: r for r in raw_audit['files'] if 'table' not in r}
    jobs = []
    for dataset, file in scope['forecast_files'].items():
        row = records[dataset]
        if row['status'] != 'verified' or Path(row['logical_path']).name != file:
            raise ValueError('Actual verified original forecast raw CSV binding required')
        jobs.append(dict(dataset=dataset, path=row['logical_path'], sha256=row['sha256'], raw_rows=row['observed_instances']))
    if len(jobs) != 6: raise ValueError('All original six forecast datasets required')
    paths = [SCOPE, source['raw_audit'], RECIPE, MODULE.replace('.', '/')+'.py', TEST,
             'scripts/flow_matching/prepare_moment_icm_PILE_long_forecast_v1.py',
             'scripts/flow_matching/prepare_moment_icm_classification_v1.py',
             'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
             'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
             'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
             'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
             'scripts/flow_matching/cached_resource_snapshot_v1.py',
             'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
             'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py',
             'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py']
    paths += [p['path'] for p in source['source_receipts'] if '/moment_research/original/' in p['path']]
    return dict(schema_version=1, paper_citation=scope['paper_citation'], data_citation='https://github.com/thuml/Autoformer',
        paper_scope=SCOPE, original_source_root=source['source_root'], related_forecast_recipe=RECIPE,
        jobs=jobs, required_datasets=6, required_horizon_dataset_pairs=18, roles=['train', 'val', 'test'],
        sequence_length=256, horizons=HORIZONS, output_type='multivariate', stride=1,
        other_dataset_ratios=dict(train=.7, validation=.1, test=.2), source_fixed_ETT_month_borders=True,
        source_scaler_fit='original_train_rows_only', source_cubic_interpolation_precedes_split=True,
        python=source['python'], environment=source['environment'], settings=source['settings'], resource_lock=source['resource_lock'],
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_forecast_evaluation_inputs_v1',
        state_root='experiments/runs/fm_moment_icm_forecast_evaluation_inputs_v1',
        boundary='All6raw datasets/all18horizons,256lookback and all multivariate channels;source downstream70/10/20 '
            'and fixed ETT months remain separate from pretraining60/10/30. Exact TC matrices,all stride1window '
            'bounds and source endpoint samples,not tensor-expanded copies. Historical preprocessing equivalence '
            'and global cubic interpolation temporal independence remain unproven;no trained forecaster metrics.')


def verify(config):
    if config != registration():
        raise ValueError('All6datasets/all18horizons,original downstream splits and data-resource policy required')
    return config['jobs']


def worker(config):
    actual = dict(python=platform.python_version(), **{k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn']})
    if actual != config['environment']: raise ValueError('Exact source preprocessing environment required')
    cls = original.source_long_class(ROOT/config['original_source_root'])
    output = Path(config['artifact_root']); output.mkdir(parents=True, exist_ok=False)
    complete = []
    for job in config['jobs']:
        raw = ROOT/job['path']
        if sha(raw) != job['sha256']: raise ValueError('Full original raw CSV changed')
        record = export_dataset(cls, raw, output/job['dataset'])
        if record['total_source_raw_rows'] != job['raw_rows']: raise ValueError('All source raw rows required')
        complete.append(dict(job=job, **record))
        if sha(raw) != job['sha256']: raise ValueError('Raw CSV changed during complete export')
        write(output/(job['dataset']+'.json'), complete[-1])
    write(output/REPORT, dict(passed=True, configuration_sha256=sha(ROOT/CONFIG), environment=actual,
        datasets=6, horizon_dataset_pairs=18, role_horizon_groups=54, source_records=complete,
        same_as_pretraining_input_split=False, raw_data_overwritten=False,
        model_training_or_prediction_performed=False, all_original_experiments_complete=False))


def guarded_controller():
    # Keep the existing successful single-worker data guard's full body. Only
    # its module/report and descriptive state labels refer to this new task.
    import ast
    from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as guard
    node = next(n for n in ast.parse(Path(guard.__file__).read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'controller')
    namespace = dict(guard.controller.__globals__, MODULE=MODULE, CONFIG=CONFIG, REPORT=REPORT)
    replacements = {'closing_all1949_anomaly_sources': 'preparing_all6_forecast_downstream_sources',
                    'all1949_anomaly_cache_closure_passed': 'all6_forecast_inputs_all18_horizons_prepared'}
    class StateNames(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str) and n.value in replacements:
                return ast.copy_location(ast.Constant(replacements[n.value]), n)
            return n
    node = StateNames().visit(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(guard.__file__), 'exec'), namespace)
    return namespace['controller']


def controller(config):
    guarded_controller()(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true'); args = parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists(): raise FileExistsError('Preserve frozen forecast inputs registration')
        with (ROOT/CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(registration(), stream, indent=2); stream.write('\n')
        print(json.dumps(dict(configuration=CONFIG, datasets=6, horizon_dataset_pairs=18)))
        return
    config = read(ROOT/CONFIG); verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, datasets=6, horizon_dataset_pairs=18,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            forecast_predictions_produced=0, all_original_experiments_complete=False)))
    elif args.worker: worker(config)
    else: controller(config)


if __name__ == '__main__': main()
