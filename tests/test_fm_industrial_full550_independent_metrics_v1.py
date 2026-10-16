"""Fixtures test the independent metric audit, never benchmark performance."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as audit
from scripts.flow_matching.timegan_native_common_v1 import sha, read, write


def npy_bytes(values):
    out = io.BytesIO(); np.save(out,values,allow_pickle=False); return out.getvalue()


@pytest.mark.parametrize('dtype', [np.float64,np.float32,np.int64,np.uint8,'<U12'])
def test_pickle_free_reader_accepts_full_native_numeric_and_fault_arrays(dtype):
    values = np.asarray([1,2,3],dtype=dtype)
    header,payload = audit.npy(io.BytesIO(npy_bytes(values)))
    assert header['shape']==values.shape and payload==values.tobytes(order='C')


@pytest.mark.parametrize('fault',['truncated','trailing','magic','version','object','fortran','oversize'])
def test_reader_rejects_unsafe_incomplete_or_unbounded_payload(fault):
    values = np.arange(4,dtype=np.float64)
    blob = npy_bytes(values)
    if fault=='truncated': blob = blob[:-1]
    elif fault=='trailing': blob += b'x'
    elif fault=='magic': blob = b'BADNPY'+blob[6:]
    elif fault=='version': blob = blob[:6]+b'\x09\x00'+blob[8:]
    elif fault=='object':
        out = io.BytesIO();np.save(out,np.asarray([{}],dtype=object),allow_pickle=True);blob = out.getvalue()
    elif fault=='fortran': blob = npy_bytes(np.asfortranarray(np.arange(6).reshape(2,3)))
    else:
        header = str(dict(descr='<f8',fortran_order=False,shape=(100000000,))).encode('ascii')
        blob = b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header
    with pytest.raises(ValueError): audit.npy(io.BytesIO(blob))


@pytest.mark.parametrize('fault',['nonfinite','nonbinary','shape','dtype'])
def test_score_and_label_vectors_require_complete_finite_native_payload(fault):
    dtype = '|u1' if fault=='nonbinary' else '<f8'
    values = np.asarray([0,2] if fault=='nonbinary' else [0.,np.nan] if fault=='nonfinite' else [0.,1.],dtype=dtype)
    header,payload = audit.npy(io.BytesIO(npy_bytes(values)))
    if fault=='shape': header['shape']=(1,)
    elif fault=='dtype': header['descr']='<f4'
    with pytest.raises(ValueError): audit.vector(header,payload,2,dtype)


def fixture(tmp_path, monkeypatch):
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    groups = dict(training_groups=[dict(array_prefix='t',group_id='train',source_interval=[0,6],points=6,features=2)],
        validation_groups=[dict(array_prefix='v',group_id='validation',source_interval=[0,5],points=5,features=2)],
        entities=[dict(array_prefix='a',group_id='testA',entity_id='testA',source_interval=[0,7],points=7,test_points=7,features=2),
                  dict(array_prefix='b',group_id='testB',entity_id='testB',source_interval=[0,8],points=8,test_points=8,features=2)])
    dataset = dict(id='fixture',input_npz='input.npz',source_files=[],scaler=dict(fit_points=6),**groups)
    inputs,saved = {},{}
    for role,g in audit.roles(dataset):
        prefix = g['array_prefix'];n=g['points']
        inputs[prefix+('_test' if role=='test' else '_values')]=np.arange(n*2,dtype=np.float32).reshape(n,2)
        saved[prefix+'_'+role]=np.linspace(0,8,n,dtype=np.float64)
        for suffix,values in [('_labels',np.asarray([(i%3)!=0 for i in range(n)],dtype=np.uint8)),
                ('_timestamps',np.arange(n,dtype=np.float64)),('_source_rows',np.arange(n,dtype=np.int64)),
                ('_fault_labels',np.asarray(['Normal' if i%3==0 else 'Fault A' for i in range(n)]))]:
            inputs[prefix+suffix]=values;saved[prefix+suffix]=values.copy()
    np.savez_compressed(tmp_path/'input.npz',**inputs);dataset['input_sha256']=sha(tmp_path/'input.npz')
    write(tmp_path/'model.json',dict(parameters=dict(epochs=2)))
    job = dict(id='fixture',dataset='fixture',seed=1103,window=3,model_config='model.json',output_directory='output',
        frozen_files=[dict(path='model.json',sha256=sha(tmp_path/'model.json'))])
    settings = dict(strict_false_alarm_percent=[.1,1,5],primary_false_alarm_percent=1,
        bootstrap_resamples=2,bootstrap_block_points=2,tab_threshold_ratios_percent=[1],tab_commit='fixture')
    out = tmp_path/'output';out.mkdir()
    (out/'model.pt').write_bytes(b'fixture placeholder; this test never loads a model')
    np.savez_compressed(out/'scores.npz',**saved)
    entities = {g['entity_id']:(saved[g['array_prefix']+'_labels'],saved[g['array_prefix']+'_test']) for g in dataset['entities']}
    metrics = evaluate_saved(entities,saved['v_validation'],saved['t_train'],dataset['input_sha256'],sha(out/'model.pt'),settings,1103)
    budget = audit.expected_budget(dataset,3)
    result = dict(id='fixture',job=job,status='completed',checkpoint_sha256=sha(out/'model.pt'),scores_sha256=sha(out/'scores.npz'),
        experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),fit_group_budget=budget,
        unique_fit_windows=2,epoch_audit=dict(declared_epochs=2,recorded_epochs=2),training_losses=[1.,.5],
        test_points=15,validation_points=5,test_entities=2,metrics=metrics)
    write(out/'result.json',result)
    return dict(numerical_metric_tolerance=dict(rtol=2e-6,atol=2e-6)),dict(evaluation=settings),dict(datasets=[dataset]),job,inputs,saved


def test_full_multi_entity_audit_uses_original_labels_metadata_and_independent_metrics(tmp_path,monkeypatch):
    c,q,manifest,job,inputs,saved = fixture(tmp_path,monkeypatch)
    report = audit.audit(c,q,manifest,job)
    assert report['passed'] and report['full_test_points']==15 and report['full_validation_points']==5
    assert len(report['all_array_bindings'])==20
    assert set(report['independent_strict_metrics'])=={'0.1','1','5'}
    assert not report['source_checkpoint_neural_replay'] and report['new_complete_model_results']==0
    assert not report['source_historical_resource_guard_certified']


@pytest.mark.parametrize('fault',['source_row','fault_label','missing_tail','extra_array','duplicate_array','source_hash','threshold','PA','training_budget','epochs'])
def test_complete_audit_rejects_coordinate_protocol_budget_and_archive_corruption(tmp_path,monkeypatch,fault):
    c,q,manifest,job,inputs,saved = fixture(tmp_path,monkeypatch)
    out = tmp_path/'output';result=read(out/'result.json')
    if fault in ['source_row','fault_label','missing_tail','extra_array','duplicate_array']:
        if fault=='source_row': saved['a_source_rows'][0]+=1
        elif fault=='fault_label': saved['a_fault_labels'][0]='X'
        elif fault=='missing_tail': saved['a_test']=saved['a_test'][:-1]
        elif fault=='extra_array': saved['unexpected']=np.array([0.])
        np.savez_compressed(out/'scores.npz',**saved)
        if fault=='duplicate_array':
            with zipfile.ZipFile(out/'scores.npz','a') as z:
                with pytest.warns(UserWarning): z.writestr('a_test.npy',npy_bytes(saved['a_test']))
        result['scores_sha256']=sha(out/'scores.npz')
    elif fault=='source_hash': result['checkpoint_sha256']='wrong'
    elif fault=='threshold': result['metrics']['strict']['1']['calibration']['threshold']+=1
    elif fault=='PA': result['metrics']['strict']['1']['point_adjustment']=True
    elif fault=='training_budget': result['unique_fit_windows']=1
    else: result['training_losses'].pop()
    write(out/'result.json',result)
    with pytest.raises(ValueError): audit.audit(c,q,manifest,job)


def test_cold_registration_preserves_full_scope_and_imports_no_neural_libraries():
    code = ('import sys;from scripts.flow_matching.audit_industrial_full550_independent_metrics_v1 import registration,verify;'
        'c=registration();q,m=verify(c);assert len(q["jobs"])==550;'
        'assert c["original_industrial_GPU_scope_unchanged"]==850;'
        'assert c["settings"]["minimum_commit_headroom_bytes"]==22*2**30;'
        'assert c["settings"]["emergency_commit_headroom_bytes"]==16*2**30;'
        'assert not any(n in sys.modules for n in ["numpy","torch","sklearn","scipy","pandas"])')
    p = subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
