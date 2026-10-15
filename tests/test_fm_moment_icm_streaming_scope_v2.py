from pathlib import Path
import os
import subprocess
import pytest
from scripts.flow_matching import moment_icm_streaming_scope_v2 as s


@pytest.mark.parametrize('text,expected',[('1,2,3',(3,0,0)),('1,?,NaN,inf',(4,1,2)),(' 1,2 ',(2,0,0))])
def test_every_numeric_token_is_validated_without_retaining_series_values(text,expected):
    assert s.validate_series(text)==expected


@pytest.mark.parametrize('text',['?,?,?','1,not_numeric,3','1,2,'])
def test_invalid_or_all_missing_source_series_are_rejected(text):
    with pytest.raises(ValueError):s.validate_series(text)


@pytest.mark.parametrize('filename',[
    'nn5_daily_dataset_without_missing_values.tsf','fred_md_dataset.tsf','cif_2016_dataset.tsf',
    'm1_monthly_dataset.tsf','us_births_dataset.tsf','saugeenday_dataset.tsf',
    'solar_weekly_dataset.tsf','bitcoin_dataset_without_missing_values.tsf'])
def test_streaming_metadata_matches_complete_actual_source_profile(filename):
    root=s.parent.ROOT
    code=r'''
import sys,json
from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as original
from scripts.flow_matching import moment_icm_streaming_scope_v2 as stream
source=original.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
path=original.ROOT/'data/public_datasets/flow_matching/project_additions/moment_timeseries_pile/forecasting/monash'/sys.argv[1]
cls=original.short_forecast_class(source)
old=original.profile_short(cls,path);new=stream.profile_stream(cls,path)
assert all(new[key]==value for key,value in old.items()),[(k,v,new.get(k)) for k,v in old.items() if new.get(k)!=v]
assert new['streaming_all_numeric_tokens_validated'] and not new['eager_value_frame_constructed']
assert 'torch' not in sys.modules
'''
    p=subprocess.run([str(root/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,filename],cwd=root,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout[-1000:]+p.stderr[-4000:]
