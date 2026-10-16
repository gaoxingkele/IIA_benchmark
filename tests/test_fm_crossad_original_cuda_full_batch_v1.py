import ast
from scripts.flow_matching import preflight_crossad_original_cuda_full_batch_v1 as s


def test_full_existing_native_batch_model_seed_and_host_gates_are_preserved():
    c=s.registration();q=s.read(s.ROOT/s.QUEUE)
    assert c['full_native_batch_shape']==[128,128,9] and c['parameters']==1327625
    assert c['all_source_training_windows']==55281 and c['job']['seed']==1105
    for k,v in c['settings'].items():assert v==q['settings'][k]
    assert c['settings']['minimum_available_memory_bytes']==24*2**30
    assert c['settings']['minimum_commit_headroom_bytes']==16*2**30
    assert c['settings']['emergency_commit_headroom_bytes']==8*2**30
    assert c['GPU']['minimum_free_bytes']==18*2**30 and c['GPU']['allocator_fraction']==.75
    assert c['CPU_GPU_array_tolerance']==dict(rtol=2e-6,atol=2e-6)
    assert c['optimizer_updates']==0 and not c['benchmark_performance']


def test_original_GPU_masks_are_preserved_and_CPU_reference_is_a_distinct_runtime():
    c=s.registration()
    original=s.ROOT/c['original_source_root']/'models/CrossAD/Basic_CrossAD.py'
    cpu=s.ROOT/c['cpu_reference_root']/'models/CrossAD/Basic_CrossAD.py'
    text=original.read_text(encoding='utf-8')
    assert text.count('self.ms_utils.scale_ind_mask(self.ms_p_lens).cuda()')==1
    assert text.count('self.ms_utils.next_scale_mask(self.ms_p_lens).cuda()')==1
    assert cpu.read_text(encoding='utf-8')==text.replace('self.ms_utils.scale_ind_mask(self.ms_p_lens).cuda()',
        'self.ms_utils.scale_ind_mask(self.ms_p_lens)').replace('self.ms_utils.next_scale_mask(self.ms_p_lens).cuda()',
        'self.ms_utils.next_scale_mask(self.ms_p_lens)')


def test_probe_executes_actual_full_backward_and_compares_full_array_without_optimizer_step():
    source=(s.ROOT/(s.MODULE.replace('.','/')+'.py')).read_text()
    tree=ast.parse(source)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    cuda_calls=[ast.unparse(n.func) for n in ast.walk(functions['cuda_worker']) if isinstance(n,ast.Call)]
    assert 'loss.backward' in cuda_calls and not any(name.endswith('.step') for name in cuda_calls)
    assert 'torch.cuda.set_per_process_memory_fraction' in cuda_calls
    assert 'torch.autocast' not in cuda_calls and 'torch.compile' not in cuda_calls
    cpu_calls=[ast.unparse(n.func) for n in ast.walk(functions['cpu_reference']) if isinstance(n,ast.Call)]
    assert 'np.allclose' in cpu_calls and 'model.infer' in cpu_calls


def test_GPU_emergency_guard_stops_only_its_own_child_and_marks_failure(monkeypatch):
    from scripts.flow_matching import heavy_resources as original
    class Child:
        def __init__(self):self.stopped=False
        def poll(self):return 1 if self.stopped else None
        def wait(self):return 1
        pid=-100
    child=Child();stopped=[]
    monkeypatch.setattr(original,'resource_snapshot',lambda:dict(physical_available_bytes=40*2**30,commit_available_bytes=30*2**30))
    def stop(target):assert target is child;target.stopped=True;stopped.append(target)
    monkeypatch.setattr(original,'terminate_tree',stop)
    def gone(pid):raise original.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(original.psutil,'Process',gone)
    monkeypatch.setattr(original.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(s,'gpu_snapshot',lambda:dict(free_bytes=1))
    c=dict(GPU=dict(emergency_free_bytes=2*2**30),settings=dict(emergency_commit_headroom_bytes=8*2**30))
    code,usage=s.guarded_GPU_wait(child,c,lambda values:None)
    assert code==1 and usage['resource_guard_aborted'] and usage['GPU_resource_guard_aborted']
    assert stopped==[child]
