import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scripts.flow_matching import prepare_moment_icm_PILE_long_forecast_v1 as s


@pytest.mark.parametrize('dataset', ['ETTh1', 'exchange_rate', 'national_illness'])
def test_real_original_source_all_tracks_splits_scaler_rows_and_CT_window_axes(tmp_path, dataset):
    code = r'''
from pathlib import Path
import sys,numpy as np
from scripts.flow_matching import prepare_moment_icm_PILE_long_forecast_v1 as p
root=p.ROOT
cls=p.source_long_class(root/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original')
path=root/'data/public_datasets/flow_matching/project_additions/moment_timeseries_pile/forecasting/autoformer'/(sys.argv[1]+'.csv')
for track in p.TRACKS:
 for split in p.SPLITS:
  data=p.source_dataset(cls,path,track,split)
  out=Path(sys.argv[2])/track/split
  record=p.export_split(data,out)
  saved=np.load(out/'values_TC.npy',allow_pickle=False)
  assert saved.tobytes()==data.data.tobytes(order='C')
  assert record['complete_source_window_count']==len(data)
  assert record['source_scaler_fitted_rows']==record['source_scaler_n_samples_seen']
  assert all(r['all_source_coordinates_byte_equal'] for r in record['files'].values())
  rows=np.load(out/'original_CSV_rows.npy',allow_pickle=False)
  bounds=getattr(data._get_borders(),split)
  assert np.array_equal(rows,np.arange(data.length_timeseries_original)[bounds])
  assert np.array_equal(saved[-256:].T,data[len(data)-1].timeseries)
  if track=='source_univariate_OT':assert saved.shape[1]==1
  else:assert saved.shape[1]>1
assert 'torch' not in sys.modules
'''
    result = subprocess.run([str(s.ROOT / '.venv/moment-author-py310/Scripts/python.exe'), '-B', '-X', 'utf8',
        '-c', code, dataset, str(tmp_path)], cwd=s.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-4000:]


def test_chunked_storage_preserves_fortran_source_coordinates_and_refuses_overwrite(tmp_path):
    data=np.asfortranarray(np.arange(1500,dtype=np.float64).reshape(500,3))
    path=tmp_path/'array.npy';record=s.save_source_array(path,data,block_bytes=1024)
    assert record['write_mapping_blocks']>1
    assert np.load(path).tobytes()==data.tobytes(order='C')
    with pytest.raises(FileExistsError):s.save_source_array(path,data)


def test_no_unknown_track_or_split_is_silently_selected():
    with pytest.raises(ValueError):s.source_dataset(None,'none','other','train')
    with pytest.raises(ValueError):s.source_dataset(None,'none','source_univariate_OT','other')
