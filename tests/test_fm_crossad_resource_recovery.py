import json
import pytest
from scripts.flow_matching.register_crossad_resource_recovery import build


def fixture(root,aborted=True):
    settings={'minimum_commit_headroom_bytes':16*1024**3,'emergency_commit_headroom_bytes':8*1024**3,'cpu_threads':2}
    jobs=[]
    for dataset in ('PSM','SMAP','GECCO'):
        job={'id':dataset,'dataset':dataset,'seed':2025,'mode':'release_checkpoint','output_directory':'old/'+dataset,
             'model_parameters':{'seq_len':192},'train_parameters':{'batch_size':128,'train_epochs':20},
             'release_checkpoint':'original.pth','frozen_files':[{'path':'author.py','sha256':'fixed-source'}]}
        jobs.append(job)
        out=root/job['output_directory'];out.mkdir(parents=True)
        if dataset=='GECCO':(out/'result.json').write_text('{}')
        else:(out/'failure.json').write_text(json.dumps({'resources':{'resource_guard_aborted':aborted}}))
    path=root/'configs/experiments/fm_crossad_complete_queue.v7.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'settings':settings,'jobs':jobs}))
    return settings,jobs


def test_recovers_only_failed_original_release_slots_and_preserves_protocol(tmp_path):
    settings,jobs=fixture(tmp_path)
    failures={p:p.read_bytes() for p in tmp_path.glob('old/*/failure.json')}
    config,native=build(tmp_path)
    assert [c['original_id'] for c in config['jobs']]==['PSM','SMAP']
    assert native['settings']==settings
    assert config['settings']==dict(settings,minimum_commit_headroom_bytes=24*1024**3)
    for case,original in zip(config['jobs'],jobs):
        assert case['original_job']==original and case['settings']==settings
        assert {k:v for k,v in case['job'].items() if k not in ('id','output_directory')}=={k:v for k,v in original.items() if k not in ('id','output_directory')}
    assert all(p.read_bytes()==value for p,value in failures.items())


def test_source_error_requires_repair_instead_of_unchanged_retry(tmp_path):
    fixture(tmp_path,False)
    with pytest.raises(ValueError,match='source failure'):build(tmp_path)
