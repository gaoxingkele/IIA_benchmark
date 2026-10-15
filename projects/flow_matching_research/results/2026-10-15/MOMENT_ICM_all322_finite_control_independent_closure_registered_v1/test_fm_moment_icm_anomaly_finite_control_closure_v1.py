import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_anomaly_finite_control_v1 as v


@pytest.mark.parametrize('slot',[0,160,321])
def test_real_all3splits_independent_full_numeric_reconstruction_and_mask_corruption_rejection(tmp_path,slot):
    code=r'''
import sys
from pathlib import Path
from scripts.flow_matching import verify_moment_icm_anomaly_finite_control_v1 as v
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as p
c=p.read(p.ROOT/p.CONFIG);jobs=p.verify(c);job=jobs[int(sys.argv[1])]
p.source.install_private_path(c)
import numpy as np,pandas as pd
rows,raw=v.template.raw_records(job['raw'])
values,observed,borders,scaler=v.independently_reconstruct(p.ROOT/job['raw']['path'])
cls=p.source.producer.source_anomaly_class(p.ROOT/c['source_root'])
for split,bounds in borders.items():
 data=p.source.producer.source_dataset(cls,p.ROOT/job['raw']['path'],split)
 directory=Path(sys.argv[2])/split
 record=p.export_split(data,job,rows,observed,directory)
 result=v.inspect_split(directory,record,values[bounds],rows[bounds],observed[bounds],scaler)
 assert result['all7arrays_independently_reconstructed_byte_equal']
 assert result['all_model_inputs_finite']
 if split=='test':
  file=directory/v.NAMES['valid_targets'];target=np.load(file);target[0]=~target[0];np.save(file,target,allow_pickle=False)
  record['files']['valid_targets']['sha256']=v.sha(file)
  try:v.inspect_split(directory,record,values[bounds],rows[bounds],observed[bounds],scaler)
  except ValueError:pass
  else:raise AssertionError('A valid-looking hash cannot certify a wrong target mask')
assert 'torch' not in sys.modules
'''
    result=subprocess.run([str(v.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,str(slot),str(tmp_path)],
        cwd=v.ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]


def test_prerequisite_checks_are_cold_before_the_single_resource_API_install():
    import sys
    code=('import sys;from scripts.flow_matching.verify_moment_icm_anomaly_finite_control_v1 import registration,verify,preparation_ready;'
        'c=registration();verify(c);preparation_ready(c);'
        'from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades;'
        'a=install_facades();a["resource_snapshot"]();assert not any(k in sys.modules for k in ["numpy","pandas","torch"])')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=v.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr[-4000:]
