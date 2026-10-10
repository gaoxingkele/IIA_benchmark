from scripts.flow_matching import run_grasp_gpu_memory_fair_queue_v6 as runtime


def test_gpu_dispatch_uses_original_settings_and_mutex_without_mutating_api(monkeypatch,tmp_path):
    original_api={'resource_slot':object(),'resource_snapshot':object()}
    original_settings={'minimum_available_memory_bytes':12,'minimum_commit_headroom_bytes':20,
                       'minimum_gpu_free_bytes':8,'emergency_commit_headroom_bytes':10}
    called=[]
    monkeypatch.setattr(runtime.original,'load_controller',lambda *args:('selection',original_api,'controller'))
    monkeypatch.setattr(runtime,'full_gpu_memory_slot',lambda api,registration,update:called.append((api,registration,update)))
    _,api,_=runtime.load_gpu_controller(tmp_path,{},'crossad')
    update=object()
    api['resource_slot'](tmp_path/'original-gpu.lock',original_settings,update)
    assert called[0][1]==dict(resource_lock='original-gpu.lock',settings=original_settings)
    assert called[0][0]['gpu_snapshot'] is runtime.original.gpu_snapshot
    assert called[0][2] is update
    assert original_api['resource_slot'] is not api['resource_slot']


def test_full_bound_function_keeps_training_guards_and_worker_globals():
    function=runtime.full_main(35)
    assert function.__globals__['guarded_gpu_wait'] is runtime.original.guarded_gpu_wait
    assert function.__globals__['complete'] is runtime.original.complete
    assert function.__globals__['verify'] is runtime.original.verify
    assert function.__globals__['load_controller'] is runtime.load_gpu_controller
