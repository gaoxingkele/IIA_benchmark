import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as s


@pytest.mark.parametrize('field,lengths,original,temporal',[
    (111,[791]*66,False,True),(266,[212]*159,True,False),
    (1,[],False,False),(256,[256],True,True),(1000,[12,96],True,False)])
def test_count_and_temporal_eligibility_are_separate_controls(field,lengths,original,temporal):
    assert s.selected(field,lengths,256,'original_source_field') is original
    assert s.selected(field,lengths,256,'observed_training_time_length') is temporal


@pytest.mark.parametrize('field,lengths,window',[(True,[1],256),(1,[True],256),(1,[1],0),(1,[-1],256)])
def test_invalid_or_ambiguous_cardinalities_are_rejected(field,lengths,window):
    with pytest.raises(ValueError):s.selected(field,lengths,window,'original_source_field')


def test_getter_uses_author_explicit_exclusion_without_modifying_source(tmp_path):
    root=s.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
    for name in ['CharacterTrajectories','BasicMotions']:
        path=tmp_path/'classification/UCR'/name;path.mkdir(parents=True)
        (path/(name+'_TEST.ts')).write_text('fixture file enumeration only')
        (path/(name+'_TRAIN.ts')).write_text('fixture file enumeration only')
    result,_=s.enumerate_inputs(root,tmp_path)
    assert [Path(p).name for p in result['classification']]==['BasicMotions_TEST.ts']


def test_actual_NN5_source_metadata_exposes_wrong_axis_without_training():
    code=r'''
import json,sys
from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as s
root=s.ROOT/'experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
path=s.ROOT/'data/public_datasets/flow_matching/project_additions/moment_timeseries_pile/forecasting/monash/nn5_daily_dataset_without_missing_values.tsf'
row=s.profile_short(s.short_forecast_class(root),path)
assert row['source_length_field']==111 and row['actual_training_rows']==66 and row['training_temporal_min']==791
assert row['original_source_field_selected'] is False and row['observed_training_time_selected'] is True
assert 'torch' not in sys.modules
print(json.dumps(row))
'''
    p=subprocess.run([str(s.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code],cwd=s.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout[-1000:]+p.stderr[-4000:]


def test_entire_snapshot_file_hash_and_size_changes_are_rejected(tmp_path):
    import hashlib
    from scripts.flow_matching.audit_moment_icm_pretraining_corpus_v1 import check_file
    p=tmp_path/'raw.tsf';raw=b'complete raw fixture bytes\n';p.write_bytes(raw)
    reference=dict(path=p.name,status='verified',size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        git_blob_sha1=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest())
    assert check_file(p,reference)['bytes']==len(raw)
    p.write_bytes(raw.replace(b'raw',b'new'))
    with pytest.raises(ValueError):check_file(p,reference)


def test_unspecified_policy_is_not_silently_replaced_by_the_easier_one():
    with pytest.raises(ValueError):s.selected(111,[791]*66,256,'guess')


def test_entire_corpus_registration_is_cold_and_keeps_every_source_path():
    code='from scripts.flow_matching.register_moment_icm_pretraining_corpus_audit_v1 import registration;from scripts.flow_matching import audit_moment_icm_pretraining_corpus_v1 as a;import sys;c=registration();g,_=a.verify(c);assert len(g["short_forecast"])==47 and c["snapshot_files"]==2441;assert not any(k in sys.modules for k in ["numpy","torch","pandas"])'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
