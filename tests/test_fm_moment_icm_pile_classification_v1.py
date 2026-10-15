import os
from pathlib import Path
import subprocess
import pytest

from scripts.flow_matching import prepare_moment_icm_pile_classification_v1 as s


@pytest.mark.parametrize('cases,channels', [(3, 1), (3, 2), (10, 9), (275, 9), (1, 1345)])
def test_disclosed_casewise_group_split_retains_all_validation_cases_and_source_boundary(cases, channels):
    r = s.split_counts(cases, channels)
    assert r['grouped_casewise_training_cases'] + r['grouped_casewise_validation_cases'] == cases
    assert r['grouped_casewise_training_channel_rows'] % channels == 0
    assert r['original_flatten_first_training_channel_rows'] >= r['grouped_casewise_training_channel_rows']
    assert r['same_casewise_split_required_for_independent_and_ICM']
    assert not r['exact_historical_ICM_split_certified']


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_invalid_case_counts_are_not_silently_reinterpreted(value):
    with pytest.raises(ValueError): s.split_counts(value, 2)


@pytest.mark.parametrize('dataset', ['ArticularyWordRecognition', 'AtrialFibrillation', 'JapaneseVowels'])
def test_real_source_pretraining_uses_only_TRAIN_and_all_cases_channels_with_pad_policy(tmp_path, dataset):
    code = r'''
import json,sys
from pathlib import Path
from scripts.flow_matching import prepare_moment_icm_pile_classification_v1 as p
from scripts.flow_matching import prepare_moment_icm_classification_v1 as o
root=p.ROOT
source=root/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
path=root/'data/public_datasets/flow_matching/project_additions/moment_timeseries_pile/classification/UCR'/sys.argv[1]/(sys.argv[1]+'_TEST.ts')
cls,_,_=o.source_components(source)
calls=[];namespace=cls._read_data.__globals__;original=namespace['load_from_tsfile']
def capture(path,*args,**kw):
 calls.append(Path(path).name)
 return original(path,*args,**kw)
namespace['load_from_tsfile']=capture
data=p.source_pretraining_dataset(cls,path)
assert calls and all(name.endswith('_TRAIN.ts') for name in calls)
assert data.task_name=='pre-training' and data.upsampling_type=='pad'
assert data.train_ratio==1 and data.val_ratio==0
assert len(data)%data.n_channels==0
cases=len(data)//data.n_channels
r=p.export_pretraining_split(data,cases,Path(sys.argv[2])/'TRAIN')
assert r['instances']==cases and r['channels']==data.n_channels
assert r['source_coordinates_copied_and_checked']==cases*data.n_channels*256
assert not r['label_values_exported']
assert set(r['files_sha256'])=={'values.npy','input_mask.npy','original_instance_rows.npy'}
assert 'torch' not in sys.modules
'''
    result = subprocess.run([str(s.ROOT / '.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8',
        '-c', code, dataset, str(tmp_path)], cwd=s.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-4000:]
