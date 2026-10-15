import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.flow_matching import prepare_moment_icm_classification_v1 as s


def test_every_source_coordinate_channel_label_and_mask_is_exported(tmp_path):
    class Dataset:
        n_channels=3
        scaler=object()
        def __len__(self):return 6
        def __getitem__(self,i):
            return type('Sample',(),dict(timeseries=(np.arange(256,dtype=np.float64)+i)[None,:],
                input_mask=np.ones(256,dtype=np.float64),labels=np.array([i//3])))()
    result=s.export_split(Dataset(),2,tmp_path/'arrays')
    x=np.load(tmp_path/'arrays/values.npy');labels=np.load(tmp_path/'arrays/labels.npy')
    assert x.shape==(2,3,256) and x.dtype==np.float64 and result['source_coordinates_copied_and_checked']==1536
    assert np.array_equal(x[1,2],np.arange(256)+5) and np.array_equal(labels,[0,1])
    assert np.array_equal(np.load(tmp_path/'arrays/original_instance_rows.npy'),[0,1])


@pytest.mark.parametrize('error',['missing_instance','label','mask','nonfinite'])
def test_incomplete_or_changed_instance_grouping_is_rejected(tmp_path,error):
    class Dataset:
        n_channels=2
        def __len__(self):return 3 if error=='missing_instance' else 4
        def __getitem__(self,i):
            values=np.ones((1,256));mask=np.ones(256);label=i//2
            if i==1:
                if error=='label':label=99
                if error=='mask':mask[-1]=0
                if error=='nonfinite':values[0,-1]=np.inf
            return type('Sample',(),dict(timeseries=values,input_mask=mask,labels=np.array([label])))()
    with pytest.raises(ValueError):s.export_split(Dataset(),2,tmp_path/'arrays')


@pytest.mark.parametrize('scenario',['equal','ragged_nan','long','spectro'])
def test_real_pinned_source_full_TRAIN_equals_original_train_plus_val_fixture(tmp_path,scenario):
    # Actual pinned parser/class/Scipy/StandardScaler in exact source interpreter;
    # fixture inputs only,never a benchmark or pretrained model experiment.
    source=(s.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original').as_posix()
    code=r'''
import sys,pathlib,numpy as np
from scripts.flow_matching import prepare_moment_icm_classification_v1 as s
root=pathlib.Path(sys.argv[1]);source=pathlib.Path(sys.argv[2]);scenario=sys.argv[3]
name='Wine' if scenario=='spectro' else 'fixture';d=root/name;d.mkdir()
header='@problemName fixture\n@timestamps false\n@univariate false\n@classLabel true A B\n@data\n'
if scenario=='ragged_nan':header=header.replace('@classLabel','@equalLength false\n@classLabel')
def raw(count,offset):
 rows=[]
 for i in range(count):
  length=263 if scenario=='long' else (5+i if scenario=='ragged_nan' else 9)
  channels=[]
  for channel in range(3):
   channels.append(','.join('?' if scenario=='ragged_nan' and i==1 and k==2 else str(offset+(channel+1)*i+(channel+1)*k) for k in range(length)))
  rows.append(':'.join(channels)+':'+('A' if i%2==0 else 'B')+'\n')
 return header+''.join(rows)
train=d/(name+'_TRAIN.ts');test=d/(name+'_TEST.ts');train.write_text(raw(4,0));test.write_text(raw(4,100))
cls,small,unscaled=s.source_components(source)
whole=s.source_dataset(cls,train,name,small,'train')
left=s.source_dataset(cls,train,name,small,'train',.6,.1)
right=s.source_dataset(cls,train,name,small,'val',.6,.1)
joined=[left[i] for i in range(len(left))]+[right[i] for i in range(len(right))]
assert len(whole)==len(joined)==12 and len(left)%3!=0
for index,row in enumerate(joined):
 w=whole[index]
 assert np.array_equal(w.timeseries,row.timeseries) and np.array_equal(w.input_mask,row.input_mask) and np.array_equal(w.labels,row.labels)
out=s.export_split(whole,4,root/'export')
assert out['values_shape']==[4,3,256]
heldout=s.source_dataset(cls,test,name,small,'test')
# Preserve and disclose the source's TEST-specific standardization.
assert not np.array_equal(whole.scaler.mean_,heldout.scaler.mean_)
assert abs(float(heldout.data.mean()))<1e-12
assert 'torch' not in sys.modules
'''
    p=subprocess.run([str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,str(tmp_path),source,scenario],
        cwd=s.ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout[-1000:]+p.stderr[-4000:]


def test_all117_raw_dataset_bindings_have_two_complete_splits():
    config=s.read(s.ROOT/'configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json')
    names=list(dict.fromkeys(r['dataset'] for r in config['all630_reported_targets'] if r['table'] in [6,7]))
    assert len(names)==117
    report=s.read(s.ROOT/(config['output_directory']+'/complete_raw_data_audit.json'))
    records={(r['dataset'],r['split']) for r in report['files'] if 'table' in r}
    assert all((name,split) in records for name in names for split in ['TRAIN','TEST'])
