import json

import pytest

from scripts.flow_matching import run_spectral_table2_method_gated_queue_v1 as controller
from scripts.flow_matching.run_light_controller import ROOT


def proof_fixture(tmp_path):
    queue=controller.read(ROOT/'configs/experiments/fm_spectral_table2_original_queue.v1.json')
    records=[]
    for dataset in ('sine','stock','mujoco'):
        job=next(j for j in queue['jobs'] if j['method']=='sdformer_ar' and j['dataset']==dataset)
        for method,index in [('sdformer_vq',0),('sdformer_ar',1)]:
            flags=job['stages'][index]['original_arguments'];size=int(flags[flags.index('--batch-size')+1])
            records.append(dict(method=method,dataset=dataset,batch_shape=[size,*job['data']['shape'][1:]],
                backward_and_optimizer_step=True,loss=1.0))
    proof=dict(passed=True,diagnostic_only=True,queue_sha256=controller.fingerprint(queue),
        native_full_shape_checks=records,full_original_eight_worker_loaders=[dict(dataset=d,num_workers=8,
            drop_last=False,worker_exit_codes=[0]*8,worker_pids=list(range(1,9)),sampler='RandomSampler',
            full_shape=next(j['data']['shape'] for j in queue['jobs'] if j['dataset']==d),
            rows=next(j['data']['shape'][0] for j in queue['jobs'] if j['dataset']==d))
            for d in ('sine','stock','mujoco')])
    path=tmp_path/'diagnostic_fixture_only.json';path.write_text(json.dumps(proof))
    return queue,proof,path,{'method_proofs':{'sdformer_ar':str(path)}}


def test_full_method_proof_rejects_reduced_batch_or_missing_dataset(tmp_path):
    queue,proof,path,registration=proof_fixture(tmp_path)
    assert controller.method_ready(queue,'sdformer_ar',registration)
    proof['native_full_shape_checks'][1]['batch_shape'][0]-=1
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='full model batch'):
        controller.method_ready(queue,'sdformer_ar',registration)
    proof['native_full_shape_checks'].pop()
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='All full original'):
        controller.method_ready(queue,'sdformer_ar',registration)


def test_gpu_slot_released_when_shared_cpu_slot_is_busy(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    monkeypatch.setattr(controller.time,'sleep',lambda _:None)
    opened=[];closed=[];cpu_attempts=0
    class Handle:
        def __init__(self,name):self.name=name;opened.append(name)
        def close(self):closed.append(self.name)
    def lock(path):
        nonlocal cpu_attempts
        if path.name=='cpu.lock':
            cpu_attempts+=1
            return None if cpu_attempts==1 else Handle('cpu')
        return Handle('gpu')
    api=dict(resource_snapshot=lambda:{'physical_available_bytes':100,'commit_available_bytes':100},
        gpu_snapshot=lambda _:{'gpu_free_bytes':100},has_headroom=lambda *_:True,exclusive_lock=lock)
    registration=dict(settings={'minimum_gpu_free_bytes':1},resource_lock='gpu.lock',cpu_reservation_lock='cpu.lock')
    with controller.complete_native_slot(api,registration,lambda _:None):
        assert closed==['gpu']
        assert opened==['gpu','gpu','cpu']
    assert closed==['gpu','cpu','gpu']


@pytest.mark.parametrize('field,value', [('worker_exit_codes',[0]),('worker_pids',[1]*8),
    ('rows',1),('full_shape',[1,24,6]),('sampler','SequentialSampler')])
def test_full_loader_rejects_missing_workers_or_truncated_data(tmp_path,field,value):
    queue,proof,path,registration=proof_fixture(tmp_path)
    proof['full_original_eight_worker_loaders'][0][field]=value
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='full loader'):
        controller.method_ready(queue,'sdformer_ar',registration)
