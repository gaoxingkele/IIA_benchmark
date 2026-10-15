import os
import subprocess
import pytest
from scripts.flow_matching import moment_icm_monash_streaming_v1 as s


@pytest.mark.parametrize('filename', ['nn5_daily_dataset_without_missing_values.tsf', 'fred_md_dataset.tsf',
    'us_births_dataset.tsf', 'saugeenday_dataset.tsf', 'bitcoin_dataset_without_missing_values.tsf',
    'm1_monthly_dataset.tsf'])
def test_every_real_original_normalized_item_in_all_three_splits_is_byte_equal(tmp_path, filename):
    code = r'''
from pathlib import Path
import sys,numpy as np
from scripts.flow_matching import moment_icm_monash_streaming_v1 as p
from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as src
source=p.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
cls=src.short_forecast_class(source)
path=p.ROOT/'data/public_datasets/flow_matching/project_additions/moment_timeseries_pile/forecasting/monash'/sys.argv[1]
profile=src.profile_short(cls,path)
result=p.prepare(path,cls,p.source_normalizer(source),profile,Path(sys.argv[2])/'prepared')
base=Path(sys.argv[2])/'prepared'
values=np.load(base/'values.npy',mmap_mode='r');masks=np.load(base/'input_mask.npy',mmap_mode='r');order=np.load(base/'source_shuffle_order.npy')
assert result['source_normalization_lambda_reused'] and result['native_getitem_reused']
for split in ['train','val','test']:
 data=cls(seq_len=256,full_file_path_and_name=path.as_posix(),data_split=split,scale=True,task_name='pre-training',train_ratio=.6,val_ratio=.1,test_ratio=.3,output_type='univariate',random_seed=13,upsampling_pad_direction='backward',upsampling_type='pad',downsampling_type='interpolate',pad_mode='constant',pad_constant_values=0)
 start,end=result['original_split_positions'][split]
 assert len(data)==end-start
 for index in range(len(data)):
  item=data[index];row=int(order[start+index])
  assert values[row].tobytes()==item.timeseries.tobytes(),(sys.argv[1],split,index)
  assert masks[row].tobytes()==item.input_mask.tobytes(),(sys.argv[1],split,index)
assert 'torch' not in sys.modules
'''
    p = subprocess.run([str(s.ROOT / '.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8',
        '-c', code, filename, str(tmp_path)], cwd=s.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout[-1000:] + p.stderr[-4000:]


def test_missing_string_sentinel_is_not_silently_coerced_to_a_different_numeric_policy():
    with pytest.raises(ValueError): list(s.numeric_tokens('1,?,3'))
    with pytest.raises(ValueError): list(s.numeric_tokens('1,no,3'))
