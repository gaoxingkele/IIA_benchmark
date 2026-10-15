"""Enumerate complete related-source pretraining inputs and audit temporal gates.

Keep the original length-field policy and a separate training-time-length control.
No neural training,normalization,example subsampling or implicit corpus reduction.
"""
from __future__ import annotations
import ast
import gc
import os
from pathlib import Path
import posixpath
import types
from typing import Optional
import warnings

from scripts.flow_matching import prepare_moment_icm_classification_v1 as components

ROOT,read,sha,write=components.ROOT,components.read,components.sha,components.write


def posix_walk(path):
    for root,dirs,files in os.walk(path):yield Path(root).as_posix(),dirs,files


def getter(source_root,relative,function,pile_root,portable=True):
    path=Path(source_root)/relative;tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function]
    if len(nodes)!=1:raise ValueError('Exact original getter required')
    namespace=dict(PATHS=types.SimpleNamespace(DATA_DIR=Path(pile_root).as_posix()),Optional=Optional,
        os=types.SimpleNamespace(path=posixpath,walk=posix_walk) if portable else os,
        DATASETS_EXTENSIONS=['.tsf','.csv','.npy'])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[function]


def enumerate_inputs(source_root,pile_root):
    groups={
        'classification':getter(source_root,'moment/data/classification_datasets.py','get_classification_datasets',pile_root)(collection='UCR'),
        'long_forecast':getter(source_root,'moment/data/forecasting_datasets.py','get_forecasting_datasets',pile_root)(collection='autoformer'),
        'short_forecast':getter(source_root,'moment/data/forecasting_datasets.py','get_forecasting_datasets',pile_root)(collection='monash'),
        'optional_FRED':getter(source_root,'moment/data/forecasting_datasets.py','get_forecasting_datasets',pile_root)(collection='fred/preprocessed'),
        'anomaly':getter(source_root,'moment/data/anomaly_detection_datasets.py','get_anomaly_detection_datasets',pile_root)(collection='TSB-UAD-Public')}
    native=getter(source_root,'moment/data/classification_datasets.py','get_classification_datasets',pile_root,portable=False)(collection='UCR')
    native={Path(p).as_posix() for p in native}
    portable=set(groups['classification'])
    return {k:sorted(v) for k,v in groups.items()},dict(
        native_only_classification_paths=sorted(native-portable),portable_only_classification_paths=sorted(portable-native),
        portable_path_handling_preserves_original_forward_slash_exclusion=True,
        missing_optional_FRED_path_does_not_establish_missing_paper_required_data=True)


def short_forecast_class(source_root):
    cls,_,_=components.source_components(source_root)
    namespace=dict(cls._read_data.__globals__)
    loader=namespace['load_from_tsfile'].__globals__
    namespace.update(Optional=Optional,warnings=warnings,
        convert_tsf_to_dataframe=loader['convert_tsf_to_dataframe'])
    path=Path(source_root)/'moment/data/forecasting_datasets.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ShortForecastingDataset']
    if len(nodes)!=1:raise ValueError('Original full short forecasting class required')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['ShortForecastingDataset']


def selected(source_length_field,training_lengths,sequence_length,policy):
    if (isinstance(sequence_length,bool) or not isinstance(sequence_length,int) or sequence_length<=0
            or isinstance(source_length_field,bool) or not isinstance(source_length_field,int) or source_length_field<0
            or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in training_lengths)):
        raise ValueError('Actual nonnegative cardinalities and positive temporal window required')
    if policy=='original_source_field':return source_length_field>=sequence_length
    if policy=='observed_training_time_length':return max(training_lengths,default=0)>=sequence_length
    raise ValueError('Explicit original or separate temporal-length control required')


def profile_short(cls,path,sequence_length=256,seed=13):
    dataset=cls(seq_len=sequence_length,full_file_path_and_name=Path(path).as_posix(),
        data_split='train',scale=False,task_name='pre-training',train_ratio=.6,val_ratio=.1,test_ratio=.3,
        output_type='univariate',random_seed=seed)
    lengths=[len(series) for series in dataset.data.series_value]
    source_field=int(dataset.length_timeseries)
    if len(dataset)!=len(lengths):raise ValueError('Every actual source training row required')
    result=dict(path=str(path),source_length_field=source_field,source_field_definition='dataframe_series_rows_before_train_val_test_split',
        actual_training_rows=len(lengths),training_temporal_min=min(lengths,default=None),
        training_temporal_max=max(lengths,default=None),training_temporal_total_points=sum(lengths),
        training_rows_length_ge_window=sum(v>=sequence_length for v in lengths),
        sequence_length=sequence_length,source_seed=seed,source_ratios=dict(train=.6,validation=.1,test=.3),
        original_source_field_selected=selected(source_field,lengths,sequence_length,'original_source_field'),
        observed_training_time_selected=selected(source_field,lengths,sequence_length,'observed_training_time_length'),
        source_scaling_disabled_for_shape_only_audit=True,neural_dataset_sample_getitem_called=False,
        source_preprocessing_or_model_equivalence_certified=False)
    del dataset;gc.collect()
    return result
