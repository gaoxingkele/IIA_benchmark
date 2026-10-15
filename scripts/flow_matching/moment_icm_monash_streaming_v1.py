"""Bounded per-series PILE Monash preprocessing using pinned source arithmetic.

Registered files contain no missing/nonfinite raw tokens. Preserve exact
PandasArray versus special-chunk NumPyArray normalization and native getitem.
Cache raw-order items plus the exact source shuffle/split index;no eager frame.
"""
from __future__ import annotations
import ast
from datetime import datetime
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as sources
from scripts.flow_matching import moment_icm_streaming_scope_v2 as metadata
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = sources.ROOT, sources.read, sources.sha, sources.write


def numeric_tokens(text):
    start = 0
    while True:
        end = text.find(',', start)
        token = text[start:] if end < 0 else text[start:end]
        if token == '?': raise ValueError('Registered source snapshot has no missing tokens;do not silently coerce missing strings')
        yield float(token)
        if end < 0: break
        start = end + 1


def source_normalizer(source_root):
    path = Path(source_root) / 'moment/data/forecasting_datasets.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ShortForecastingDataset')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_read_data')
    functions = [n for n in ast.walk(method) if isinstance(n, ast.Lambda)]
    if len(functions) != 1 or ast.unparse(functions[0]) != 'lambda i: (i - i.mean()) / (i.std(ddof=0) + 1e-07)':
        raise ValueError('Exact original per-series normalization lambda required')
    return eval(compile(ast.Expression(functions[0]), str(path), 'eval'), {})


def tsf_rows(path):
    import numpy as np
    import pandas as pd
    attributes = []; info = dict(frequency=None, forecast_horizon=None,
        contain_missing_values=None, contain_equal_length=None)
    body = False; count = 0
    with Path(path).open('r', encoding='cp1252') as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line.startswith('@'):
                if line.startswith('@data'):
                    if not attributes: raise ValueError('Original attributes must precede data')
                    body = True; continue
                parts = line.split(' ')
                if line.startswith('@attribute'):
                    if len(parts) != 3: raise ValueError('Invalid original attribute')
                    attributes.append((parts[1], parts[2]))
                else:
                    if len(parts) != 2: raise ValueError('Invalid original metadata')
                    if parts[0] == '@frequency': info['frequency'] = parts[1]
                    elif parts[0] == '@horizon': info['forecast_horizon'] = int(parts[1])
                    elif parts[0] == '@missing': info['contain_missing_values'] = metadata.boolean(parts[1])
                    elif parts[0] == '@equallength': info['contain_equal_length'] = metadata.boolean(parts[1])
                continue
            if not body: raise ValueError('Original data tag required')
            if info['forecast_horizon'] is not None and info['forecast_horizon'] <= 0:
                raise ValueError('Original forecasting horizon must be positive')
            fields = line.split(':')
            if len(fields) != len(attributes) + 1: raise ValueError('Complete original attributes and series values required')
            values = np.fromiter(numeric_tokens(fields[-1]), dtype=np.float64, count=fields[-1].count(',') + 1)
            if not np.isfinite(values).all(): raise ValueError('Registered complete snapshot contains only finite raw values')
            labels = {}
            for (name, kind), value in zip(attributes, fields[:-1]):
                if kind == 'numeric': labels[name] = int(value)
                elif kind == 'string': labels[name] = str(value)
                elif kind == 'date': labels[name] = datetime.strptime(value, '%Y-%m-%d %H-%M-%S')
                else: raise ValueError('Invalid original attribute kind')
            if 'series_name' not in labels: raise ValueError('Original getitem requires series_name')
            yield pd.Series(values).array, labels, dict(info), list(attributes), count
            count += 1
    if not body or count == 0: raise ValueError('Complete nonempty original TSF required')


def series_rows(cls, path):
    rows = tsf_rows(path)
    special = Path(path).stem in cls._list_of_long_timeseries(None, None)
    for values, attributes, info, columns, raw_index in rows:
        if special:
            if raw_index != 0 or len(columns) != 2:
                raise ValueError('Original special splitter requires one raw series and value column2')
            import numpy as np
            original = np.asarray(values); count, tail = divmod(len(original), 256)
            if count <= 0: raise ValueError('Original special splitter cannot produce zero chunks')
            for i in range(count):
                label = dict(attributes, series_name='T' + str(i + 1))
                yield original[i * 256:(i + 1) * 256], label, info, dict(raw_series_index=0,
                    source_chunk_index=i, raw_time_start=i * 256, raw_time_end_exclusive=(i + 1) * 256,
                    original_special_tail_omitted=tail)
        else:
            yield values, attributes, info, dict(raw_series_index=raw_index, source_chunk_index=None,
                raw_time_start=0, raw_time_end_exclusive=len(values), original_special_tail_omitted=0)


class OneRow:
    def __init__(self, name, values): self.row = SimpleNamespace(series_name=name, series_value=values)
    def __getitem__(self, index):
        if index != (0, slice(None, None, None)): raise IndexError('Single original getitem row expected')
        return self.row


def original_item(cls, path, values, name, info):
    item = object.__new__(cls)
    item.seq_len = 256; item.n_channels = 1; item.task_name = 'pre-training'
    item.dataset_name = Path(path).stem; item.forecast_horizon = info['forecast_horizon'] or 8
    item.upsampling_pad_direction = 'backward'; item.upsampling_type = 'pad'
    item.downsampling_type = 'interpolate'; item.pad_mode = 'constant'
    item.meta_data = dict(info); item.data = SimpleNamespace(iloc=OneRow(name, values))
    return item[0]


def create_array(path, shape, dtype):
    import numpy as np
    if Path(path).exists(): raise FileExistsError('Preserve previous array')
    array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
    array.flush(); array._mmap.close(); del array


def prepare(path, cls, normalize, profile, output):
    import numpy as np
    import pandas as pd
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = series_rows(cls, path); first = next(rows); count = profile['source_length_field']
    values, labels, info, provenance = first
    first_item = original_item(cls, path, normalize(values), labels['series_name'], info)
    create_array(output / 'values.npy', (count, 1, 256), first_item.timeseries.dtype)
    create_array(output / 'input_mask.npy', (count, 256), first_item.input_mask.dtype)
    lengths = np.empty(count, dtype=np.int64)
    block = max(1, 16 * 2**20 // (256 * first_item.timeseries.dtype.itemsize))
    signals = masks = None; completed = 0
    publish = status_writer(output / 'progress.json', write)
    try:
        with (output / 'series_metadata.jsonl').open('x', encoding='utf-8', newline='\n') as descriptions:
            for index, (values, labels, info, provenance) in enumerate(itertools.chain([first], rows)):
                if index >= count: raise ValueError('Original complete metadata row count exceeded')
                if index % block == 0:
                    if signals is not None:
                        signals.flush(); masks.flush(); signals._mmap.close(); masks._mmap.close()
                    signals = np.load(output / 'values.npy', mmap_mode='r+', allow_pickle=False)
                    masks = np.load(output / 'input_mask.npy', mmap_mode='r+', allow_pickle=False)
                    publish(output / 'progress.json', dict(completed_series=index, required_series=count))
                item = first_item if index == 0 else original_item(cls, path, normalize(values), labels['series_name'], info)
                if (item.timeseries.shape != (1, 256) or item.input_mask.shape != (256,)
                        or not np.isfinite(item.timeseries).all() or not np.isin(item.input_mask, [0, 1]).all()
                        or item.timeseries.dtype != signals.dtype or item.input_mask.dtype != masks.dtype):
                    raise ValueError('Complete finite native source signals and binary masks required')
                signals[index] = item.timeseries; masks[index] = item.input_mask; lengths[index] = len(values)
                if signals[index].tobytes() != item.timeseries.tobytes() or masks[index].tobytes() != item.input_mask.tobytes():
                    raise ValueError('Saved native source coordinates differ')
                descriptions.write(json.dumps(dict(source_row=index, series_name=labels['series_name'],
                    pre_resampling_length=len(values), **provenance), separators=(',', ':')) + '\n')
                completed += 1
    finally:
        if signals is not None:
            signals.flush(); masks.flush(); signals._mmap.close(); masks._mmap.close(); del signals, masks
    if completed != count: raise ValueError('Every original source series required')
    order = pd.DataFrame({'source': np.arange(count, dtype=np.int64)}).sample(frac=1, random_state=13).source.to_numpy()
    n_train, n_test = int(.6 * count), int(.3 * count)
    train_lengths = lengths[order[:n_train]]
    observed = dict(actual_training_rows=n_train, training_temporal_min=int(train_lengths.min()) if n_train else None,
        training_temporal_max=int(train_lengths.max()) if n_train else None,
        training_temporal_total_points=int(train_lengths.sum()), training_rows_length_ge_window=int((train_lengths >= 256).sum()))
    if any(profile[k] != value for k, value in observed.items()): raise ValueError('Complete source shuffle/length metadata differs')
    for name, array in [('source_series_rows.npy', np.arange(count, dtype=np.int64)),
                         ('source_shuffle_order.npy', order), ('source_series_lengths.npy', lengths)]:
        np.save(output / name, array, allow_pickle=False)
    publish(output / 'progress.json', dict(completed_series=completed, required_series=count))
    return dict(series=count, channels=1, sequence_length=256, source_seed=13,
        source_count_gate_selected=profile['original_source_field_selected'],
        separate_temporal_control_selected=profile['observed_training_time_selected'],
        original_split_positions=dict(train=[0, n_train], val=[n_train, count - n_test], test=[count - n_test, count]),
        observed_source_training_length_metadata=observed,
        source_order_cache_with_exact_shuffle_index=True, dtype=str(first_item.timeseries.dtype),
        mask_dtype=str(first_item.input_mask.dtype), all_source_coordinates_and_masks_byte_equal=True,
        source_normalization_lambda_reused=True, native_getitem_reused=True,
        ordinary_PandasArray_and_special_NumpyArray_representations_preserved=True,
        exact_historical_ICM_pipeline_certified=False, normalized_complete_series_before_resampling_and_masking=True,
        strict_train_only_or_masked_target_independent_normalization=False,
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'progress.json'},
        neural_fit_inference_SVM=0)
