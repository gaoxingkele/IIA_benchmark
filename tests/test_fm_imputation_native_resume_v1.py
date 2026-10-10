import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flow_matching import run_imputation_native_resume_queue_v1 as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_original_scope_gate_remains_pending_even_after_physio_completion(tmp_path):
    write(tmp_path/'prior.json', dict(jobs=[]))
    write(tmp_path/'campaign.json', dict(execution=dict(original_scope_status='in_progress')))
    queue=dict(requires_complete_queue='prior.json', requires_original_scope_review='campaign.json')
    assert not runner.original_gates(tmp_path, queue)['ready']
    write(tmp_path/'campaign.json', dict(execution=dict(original_scope_status='completed_or_documented_unavailable')))
    assert runner.original_gates(tmp_path, queue)['ready']


def test_missing_full_original_result_keeps_grin_gate_pending(tmp_path):
    config=dict(id='saits_original', output_root='run', diagnostic=False)
    write(tmp_path/'config.json', config)
    write(tmp_path/'prior.json', dict(jobs=[dict(config='config.json',config_sha256=runner.digest(tmp_path/'config.json'))]))
    gate=runner.original_gates(tmp_path, dict(requires_complete_queue='prior.json'))
    assert not gate['ready'] and gate['prerequisite_pending']==1


def test_model_command_is_exact_original_full_queue_command(tmp_path):
    queue=dict(python='author/python.exe')
    job=dict(runner='scripts/run.py',config='configs/full.json')
    assert runner.command_for(tmp_path,queue,job)==[str(tmp_path/'author/python.exe'),
        str(tmp_path/'scripts/run.py'),'--config',str(tmp_path/'configs/full.json')]


def test_lane_wait_holds_no_idle_resource_slot_and_releases_owner(tmp_path,monkeypatch):
    calls=[]
    handle=SimpleNamespace(close=lambda:calls.append('closed'))
    answers=iter([None,handle])
    api=dict(exclusive_lock=lambda path:next(answers))
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:calls.append(('sleep',seconds)))
    with runner.original_lane(api,tmp_path/'queue.lock',lambda value:calls.append(value['state'])):
        calls.append('model')
    assert calls==['waiting_original_imputation_lane',('sleep',10),'model','closed']


def test_resume_preserves_existing_checkpoint_bytes_without_overwrite(tmp_path):
    output=tmp_path/'model';output.mkdir();(output/'last.ckpt').write_bytes(b'original optimizer and RNG')
    config=dict(id='full',output_root='model')
    runner.preserve_inputs(tmp_path,config,tmp_path/'aux')
    assert (output/'last.ckpt').read_bytes()==b'original optimizer and RNG'
    assert (tmp_path/'aux/preserved_resume_inputs/full/last.ckpt').read_bytes()==(output/'last.ckpt').read_bytes()
    with pytest.raises(FileExistsError):runner.preserve_inputs(tmp_path,config,tmp_path/'aux')


def test_grin_reduced_budget_is_rejected_and_missing_result_is_not_zero(tmp_path):
    config=dict(id='full',paper_id='grin',output_root='model',seed=2026,dataset='air36',diagnostic=False)
    write(tmp_path/'config.json',config)
    job=dict(config='config.json',config_sha256=runner.digest(tmp_path/'config.json'))
    assert runner.runner_result(tmp_path,job,dict(epochs=300)) is False
    result=dict(status='completed',config_sha256=job['config_sha256'],config=config,
        metrics=dict(mae=1,mse=1,rmse=1,mre=1,mape=1),author_arguments=dict(seed=2026,
        dataset_name='air36',model_name='grin',epochs=1,batch_size=32,samples_per_epoch=5120),completed_epochs=1)
    write(tmp_path/'model/result.json',result)
    with pytest.raises(ValueError,match='author arguments'):
        runner.runner_result(tmp_path,job,dict(epochs=300,batch_size=32,samples_per_epoch=5120))


def test_diagnostic_cannot_satisfy_original_gate(tmp_path):
    config=dict(id='short',output_root='model',diagnostic=True)
    write(tmp_path/'config.json',config)
    digest=runner.digest(tmp_path/'config.json')
    write(tmp_path/'model/result.json',dict(status='completed',config_sha256=digest,config=config))
    write(tmp_path/'prior.json',dict(jobs=[dict(config='config.json',config_sha256=digest)]))
    with pytest.raises(ValueError,match='Diagnostic'):
        runner.original_gates(tmp_path,dict(requires_complete_queue='prior.json'))


def test_registration_checks_full_jobs_and_frozen_source_not_just_status(tmp_path):
    write(tmp_path/'config.json',dict(id='full',diagnostic=False,device='cuda:0'))
    job=dict(config='config.json',config_sha256=runner.digest(tmp_path/'config.json'),runner='run.py')
    write(tmp_path/'queue.json',dict(jobs=[job]))
    (tmp_path/'run.py').write_text('full_original_training()',encoding='utf-8')
    registration=dict(source_receipts=[dict(path='run.py',sha256=runner.digest(tmp_path/'run.py'))],
        queue='queue.json',queue_sha256=runner.digest(tmp_path/'queue.json'),allowed_runners=['run.py'],full_job_count=1)
    assert len(runner.verify_registration(tmp_path,registration)['jobs'])==1
    (tmp_path/'run.py').write_text('short_training()',encoding='utf-8')
    with pytest.raises(ValueError,match='source differs'):runner.verify_registration(tmp_path,registration)


def test_cfmi_uses_original_explicit_gpu_cli_without_inventing_config_device(tmp_path):
    write(tmp_path/'config.json',dict(id='full',diagnostic=False,epochs=200))
    job=dict(config='config.json',config_sha256=runner.digest(tmp_path/'config.json'),
             runner='scripts/flow_matching/run_cfmi.py')
    write(tmp_path/'queue.json',dict(jobs=[job]))
    registration=dict(source_receipts=[],queue='queue.json',queue_sha256=runner.digest(tmp_path/'queue.json'),
        allowed_runners=[job['runner']],full_job_count=1)
    assert runner.verify_registration(tmp_path,registration)['jobs']==[job]
    job['runner']='scripts/flow_matching/run_csdi.py'
    write(tmp_path/'queue.json',dict(jobs=[job]))
    registration.update(queue_sha256=runner.digest(tmp_path/'queue.json'),allowed_runners=[job['runner']])
    with pytest.raises(ValueError,match='CUDA'):runner.verify_registration(tmp_path,registration)
