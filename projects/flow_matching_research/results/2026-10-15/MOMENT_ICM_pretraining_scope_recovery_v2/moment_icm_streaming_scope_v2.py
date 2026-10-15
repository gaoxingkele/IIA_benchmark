"""Streaming TSF value validation and exact length-only source-split profiling.

Every numeric token is parsed,then discarded. Only lengths are retained;this
replaces eager value-frame allocation in the failed metadata audit,not a model.
"""
from datetime import datetime
import math
from pathlib import Path

from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as parent


def boolean(value):
    value=value.lower()
    if value in ('y','yes','t','true','on','1'):return True
    if value in ('n','no','f','false','off','0'):return False
    raise ValueError('Invalid source Boolean metadata')


def validate_series(text):
    count=missing=nonfinite=0;start=0
    while True:
        end=text.find(',',start)
        token=text[start:] if end<0 else text[start:end]
        if token=='?':missing+=1
        else:
            value=float(token)
            nonfinite+=int(not math.isfinite(value))
        count+=1
        if end<0:break
        start=end+1
    if count==missing:raise ValueError('All source series values are missing')
    return count,missing,nonfinite


def scan_tsf(path):
    attributes=[];lengths=[];found_data=False;nonempty=0;missing=nonfinite=0;horizon=None
    with Path(path).open('r',encoding='cp1252') as stream:
        for line in stream:
            line=line.strip()
            if not line:continue
            nonempty+=1
            if line.startswith('@'):
                if line.startswith('@data'):
                    if not attributes:raise ValueError('Attributes must precede data')
                    found_data=True;continue
                parts=line.split(' ')
                if line.startswith('@attribute'):
                    if len(parts)!=3:raise ValueError('Invalid source attribute metadata')
                    attributes.append((parts[1],parts[2]))
                else:
                    if len(parts)!=2:raise ValueError('Invalid source metadata')
                    if line.startswith('@horizon'):horizon=int(parts[1])
                    elif line.startswith(('@missing','@equallength')):boolean(parts[1])
                continue
            if line.startswith('#'):continue
            if not attributes or not found_data:raise ValueError('Missing source attributes or data tag')
            fields=line.split(':')
            if len(fields)!=len(attributes)+1:raise ValueError('Missing source attributes/series values')
            size,m,n=validate_series(fields[-1]);lengths.append(size);missing+=m;nonfinite+=n
            for (_,kind),value in zip(attributes,fields[:-1]):
                if kind=='numeric':int(value)
                elif kind=='date':datetime.strptime(value,'%Y-%m-%d %H-%M-%S')
                elif kind!='string':raise ValueError('Invalid original attribute type')
    if not nonempty or not attributes or not lengths:raise ValueError('Complete source TSF data required')
    if horizon is not None and horizon<=0:raise ValueError('Original forecasting horizon must be positive')
    return dict(lengths=lengths,attributes=attributes,validated_values=sum(lengths),missing_tokens=missing,
        parsed_nonfinite_numeric_tokens=nonfinite)


def profile_stream(cls,path,sequence_length=256,seed=13):
    import pandas as pd
    scanned=scan_tsf(path);lengths=scanned['lengths'];tail=0
    special=cls._list_of_long_timeseries(None,None)
    if Path(path).stem in special:
        # Preserve the source's singular-series splitter exactly for metadata.
        if len(lengths)!=1 or len(scanned['attributes'])!=2 or 'series_name' not in dict(scanned['attributes']):
            raise ValueError('Original special splitter requires one series and value column2')
        chunks,tail=divmod(lengths[0],sequence_length)
        if chunks<=0:raise ValueError('Original splitter cannot create zero sections')
        lengths=[sequence_length]*chunks
    source_field=len(lengths)
    # Source DataFrame.sample selects row positions independently of value cells.
    frame=pd.DataFrame({'_length':lengths}).sample(frac=1,random_state=seed).reset_index(drop=True)
    selected=frame.iloc[:int(.6*source_field),0].tolist()
    selected=[int(v) for v in selected]
    return dict(path=str(path),source_length_field=source_field,
        source_field_definition='dataframe_series_rows_before_train_val_test_split',
        actual_training_rows=len(selected),training_temporal_min=min(selected,default=None),
        training_temporal_max=max(selected,default=None),training_temporal_total_points=sum(selected),
        training_rows_length_ge_window=sum(v>=sequence_length for v in selected),
        sequence_length=sequence_length,source_seed=seed,source_ratios=dict(train=.6,validation=.1,test=.3),
        original_source_field_selected=parent.selected(source_field,selected,sequence_length,'original_source_field'),
        observed_training_time_selected=parent.selected(source_field,selected,sequence_length,'observed_training_time_length'),
        source_scaling_disabled_for_shape_only_audit=True,neural_dataset_sample_getitem_called=False,
        source_preprocessing_or_model_equivalence_certified=False,
        streaming_all_numeric_tokens_validated=True,raw_values_parsed_and_discarded=scanned['validated_values'],
        raw_missing_tokens=scanned['missing_tokens'],raw_nonfinite_numeric_tokens=scanned['parsed_nonfinite_numeric_tokens'],
        original_special_splitter_tail_omitted_points=tail,eager_value_frame_constructed=False)
