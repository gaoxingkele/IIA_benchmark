from scripts.flow_matching import run_native_grasp_gpu_memory_recovery_v3 as runtime
from scripts.flow_matching import run_grasp_protocol_queue as original


def test_fixed_manifest_and_original_cold_cuda_worker_remain_bound():
    from scripts.flow_matching.native_exact_recovery_manifest_v2 import verified_manifest
    bound=runtime.full_controller()
    assert bound.__globals__['verify'] is verified_manifest
    assert bound.__globals__['guarded_gpu_wait'] is original.guarded_gpu_wait
    assert 'scripts.flow_matching.run_native_grasp_recovery_v2' in bound.__code__.co_consts
    assert bound.__globals__['load_controller'] is runtime.load_recovery_gpu_controller


def test_admission_accepts_original_native_poll_argument(monkeypatch):
    calls=[]
    api={'resource_slot':lambda *args:calls.append(args)}
    monkeypatch.setattr(runtime,'load_gpu_controller',lambda *args:(1,api,3))
    _,wrapped,_=runtime.load_recovery_gpu_controller('root',{},'crossad')
    settings=object();update=object()
    wrapped['resource_slot']('gpu.lock',settings,update,poll_seconds=.25)
    assert calls==[('gpu.lock',settings,update)]
