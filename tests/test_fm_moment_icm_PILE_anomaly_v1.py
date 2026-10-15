"""Native source preprocessing checks. No neural benchmark results."""
import os
from pathlib import Path
import subprocess
import pytest
from scripts.flow_matching import prepare_moment_icm_PILE_anomaly_v1 as p


@pytest.mark.parametrize('family',['Daphnet','NASA-MSL','KDD21'])
def test_real_original_all_splits_header_rows_scaler_full_values_and_item_axes(tmp_path,family):
    code=r'''
import json,sys,numpy as np
from pathlib import Path
from scripts.flow_matching import prepare_moment_icm_PILE_anomaly_v1 as p
scope=p.read(p.ROOT/p.storage.SCOPE)
registry={r['path']:r for r in p.read(p.ROOT/scope['integrity'])['records']}
path=next(x for x in scope['all_source_getter_paths']['anomaly'] if Path(x).parent.name==sys.argv[1])
raw=registry[path]; pair=None
if sys.argv[1]=='NASA-MSL': pair=registry[(Path(path).parent/(Path(path).name.split('.')[0]+'.train.out')).as_posix()]
job=dict(family=sys.argv[1],raw=raw,train_raw=pair)
audit,rows=p.csv_identity(raw)
if pair:train_audit,train_rows=p.csv_identity(pair)
cls=p.source_anomaly_class(p.ROOT/scope['source_root'])
for split in p.SPLITS:
 data=p.source_dataset(cls,p.ROOT/path,split)
 ids=train_rows if pair and split!='test' else rows
 out=Path(sys.argv[2])/split
 result=p.export_split(data,job,ids,out,NASA_train_rows=len(train_rows) if pair else None)
 assert np.load(out/'values_T.npy').tobytes()==data.data.tobytes()
 actual=np.load(out/'raw_CSV_record_indices.npy')
 assert len(actual)==len(data.data)
 assert result['source_dunder_len']==data.__len__()
 assert result['source_scaler_fit_rows']==int(data.scaler.n_samples_seen_)
 if sys.argv[1]=='Daphnet' and split=='test':assert actual[-1]==rows[-2]
 if pair and split=='test':assert actual[-1]==rows[-1]
 if pair:assert result['source_scaler_fit_rows']==len(train_rows)
assert audit['source_discards_numeric_first_record']
assert 'torch' not in sys.modules
'''
    result=subprocess.run([str(p.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8',
        '-c',code,family,str(tmp_path)],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]


def test_short_native_source_preserves_negative_len_exclusion_and_raw_data(tmp_path):
    code=r'''
from pathlib import Path
import sys,numpy as np
from scripts.flow_matching import prepare_moment_icm_PILE_anomaly_v1 as p
root=Path(sys.argv[1]); folder=root/'Daphnet';folder.mkdir()
raw=folder/'short.out';raw.write_text(''.join(f'{i},0\n' for i in range(100)),encoding='utf-8')
record=dict(path=str(raw),size_bytes=raw.stat().st_size,sha256=p.sha(raw))
audit,rows=p.csv_identity(record)
scope=p.read(p.ROOT/p.storage.SCOPE)
cls=p.source_anomaly_class(p.ROOT/scope['source_root'])
data=p.source_dataset(cls,raw,'train');assert data.__len__()<0
out=root/'output';r=p.export_split(data,dict(family='Daphnet',raw=record,train_raw=None),rows,out)
assert not r['source_length_gate_selected'] and r['eligible_stride1_window_count']==0
assert np.load(out/'values_T.npy').shape==(59,)
assert not r['window_endpoint_indices_checked']
assert p.sha(raw)==record['sha256']
try:p.export_split(data,dict(family='Daphnet',raw=record,train_raw=None),rows,out)
except FileExistsError:pass
else:raise AssertionError('overwrite accepted')
'''
    result=subprocess.run([str(p.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8',
        '-c',code,str(tmp_path)],cwd=p.ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def test_unknown_split_is_rejected_before_reading():
    with pytest.raises(ValueError):p.source_dataset(None,'absent','other')
