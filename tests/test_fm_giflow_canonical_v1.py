import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching import summarize_giflow_canonical_v1 as capture
from scripts.flow_matching.giflow_native_protocol import fingerprint, sha


@pytest.fixture
def result_case(tmp_path):
    job=dict(id='native',output_directory='attempt',track='reviewed_corrected',
             arguments=dict(batch_size=128,training_epoch=300))
    out=tmp_path/'attempt';out.mkdir();queue=tmp_path/'queue.json';queue.write_text('{}')
    loader=dict(samples=6267,batch_size=128,batches=48,drop_last=True)
    steps={str(i):48 for i in range(300)}
    observation=dict(native_train_loader=loader,optimizer_steps=14400,epoch_optimizer_steps=steps)
    (out/'training_observation.json').write_text(json.dumps(observation))
    artifacts=[dict(path='attempt/training_observation.json',sha256=sha(out/'training_observation.json'))]
    for i in range(2):
        p=out/f'prediction_{i}.npz';p.write_bytes(b'receipt-test-only')
        artifacts.append(dict(path=p.relative_to(tmp_path).as_posix(),sha256=sha(p),targets=50))
    result=dict(status='completed',diagnostic=False,experiment_sha256=fingerprint(job),
        job=job,queue_sha256=sha(queue),artifacts=artifacts,native_train_loader=loader,
        epoch_optimizer_steps=steps,epochs_completed=300,optimizer_updates=14400,
        data_binding=dict(windows=dict(train=6267,test=200)),full_test_batches=2,
        test_targets_with_window_repetitions=100,
        metrics=dict(native_mean_batch_mae=1.,native_mean_batch_mse=4.,
                     native_mean_batch_mape_percent=5.,sqrt_native_mean_batch_mse=2.),
        selection=dict(test_used_in_validation=False,tested_weights_match_validation_checkpoint=True))
    (out/'native.stdout.log').write_text('\n'.join(f'train_loss=1.0 after epoch {i}' for i in range(300)))
    (out/'data_binding.json').write_text(json.dumps(result['data_binding']))
    (out/'evaluated_model.pt').write_bytes(b'weight-receipt-test-only')
    for name in ['native.stdout.log','data_binding.json','evaluated_model.pt']:
        artifacts.append(dict(path='attempt/'+name,sha256=sha(out/name)))
    def save(value):
        (out/'result.json').write_text(json.dumps(value))
    save(result)
    return tmp_path,job,result,save


def test_full_drop_last_result_accepted(result_case):
    root,job,result,_=result_case
    assert capture.observed_result(job,'queue.json',root)==result


@pytest.mark.parametrize('change', ['queue','job','epochs','batch','coverage','sqrt','checkpoint','diagnostic'])
def test_invalid_full_result_rejected(result_case,change):
    root,job,result,save=result_case
    if change=='queue':result['queue_sha256']='bad'
    if change=='job':result['job']=dict(job,id='another')
    if change=='epochs':result['epoch_optimizer_steps'].pop('299')
    if change=='batch':result['native_train_loader']['batch_size']=2
    if change=='coverage':result['full_test_batches']=1
    if change=='sqrt':result['metrics']['sqrt_native_mean_batch_mse']=1
    if change=='checkpoint':result['selection']['tested_weights_match_validation_checkpoint']=False
    if change=='diagnostic':result['diagnostic']=True
    save(result)
    with pytest.raises(ValueError):capture.observed_result(job,'queue.json',root)


def test_missing_result_blank_and_single_seed_has_no_uncertainty(tmp_path):
    assert capture.observed_result(dict(output_directory='missing'),'queue.json',tmp_path) is None
    assert capture.aggregate([])['mean'] is None
    value=capture.aggregate([0.])
    assert value['mean']==0 and value['n']==1
    assert all(value[k] is None for k in ['std','se','ci95_low','ci95_high'])


def test_short_run_requires_original_early_stop_evidence(result_case):
    root,job,result,save=result_case
    out=root/'attempt'
    result.update(epochs_completed=2,optimizer_updates=96,epoch_optimizer_steps={'0':48,'1':48})
    def update_file(name,content):
        (out/name).write_text(content)
        for a in result['artifacts']:
            if a['path']=='attempt/'+name:a['sha256']=sha(out/name)
    update_file('training_observation.json',json.dumps(dict(native_train_loader=result['native_train_loader'],
        optimizer_steps=96,epoch_optimizer_steps=result['epoch_optimizer_steps'])))
    log='train_loss=1.0 after epoch 0\ntrain_loss=1.0 after epoch 1\n'
    update_file('native.stdout.log',log);save(result)
    with pytest.raises(ValueError,match='early-stop'):capture.observed_result(job,'queue.json',root)
    update_file('native.stdout.log',log+'Early stopping\n');save(result)
    assert capture.observed_result(job,'queue.json',root)['epochs_completed']==2


def test_live_worker_requires_exact_queue_and_worker(tmp_path):
    class Process:
        pid=123
        def __init__(self,cmd):self.cmd=cmd
        def cmdline(self):return self.cmd
        def create_time(self):return 42.
    command=['python','-m','scripts.flow_matching.run_giflow_native_job_v2','--queue','q.json','--job-id','native']
    p=Process(command)
    assert capture.live_workers(tmp_path,'q.json',command[2],[p])['native']['create_time']==42.
    assert capture.live_workers(tmp_path,'another.json',command[2],[p])=={}
    assert capture.live_workers(tmp_path,'q.json','scripts.flow_matching.run_giflow_native_job',[p])=={}


def test_retry_not_an_additional_seed_and_imputation_task(monkeypatch,tmp_path):
    fields=dict(id='native',dataset='air36',track='released_mirror',recipe='main',seed=393,
        arguments=dict(missing_rate=.2,missing_type='point'),author_source='source',model_config='model')
    original=dict(fields,output_directory='old')
    attempt=dict(fields,output_directory='new',original_canonical_id='native',original_output_directory='old')
    old=dict(jobs=[original],settings={},resource_lock='lock')
    (tmp_path/'old.json').write_text(json.dumps(old))
    new=dict(old,jobs=[attempt],original_canonical_queue='old.json',original_canonical_queue_sha256=sha(tmp_path/'old.json'))
    (tmp_path/'new.json').write_text(json.dumps(new))
    monkeypatch.setattr(capture,'verify',lambda *args:None)
    monkeypatch.setattr(capture,'live_workers',lambda *args:{})
    record=dict(metrics=dict(zip(capture.METRICS,[1.,4.,5.,2.])),epochs_completed=300,optimizer_updates=14400,
        total_native_seconds=100.,parameters=200,peak_cuda_allocated_bytes=300)
    (tmp_path/'new').mkdir();(tmp_path/'new/result.json').write_text(json.dumps(record))
    monkeypatch.setattr(capture,'observed_result',lambda *args:copy.deepcopy(record))
    settings=dict(canonical_queue='old.json',observed_attempt_queue='new.json',required_canonical_slots=1,required_seeds_per_group=1)
    report=capture.summarize(settings,tmp_path)
    assert report['canonical_slots']==1 and report['job_counts']=={'completed':1}
    assert len(report['metric_rows'])==4
    assert all(r['task']=='imputation' and r['n']==1 for r in report['metric_rows'])
    settings['required_seeds_per_group']=2
    with pytest.raises(ValueError,match='seed scope'):capture.summarize(settings,tmp_path)
