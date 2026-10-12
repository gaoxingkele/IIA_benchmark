"""Freeze additional GPU reservation for the unchanged413 original replay jobs."""
import copy
import json

import psutil

from scripts.flow_matching import run_shcl_full413_completion_v2 as replay
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    native=replay.original.replay.native
    path='configs/runtime/fm_shcl_full413_completion_replay.v1.json';old=native.read(native.ROOT/path)
    replay.original.verify(old)
    state=native.read(native.ROOT/old['state_root']/'status.json');actor=psutil.Process(state['pid'])
    if actor.create_time()!=state['create_time'] or actor.cmdline()!=state['command']:
        raise ValueError('Exact idle original full413 replay identity required')
    reference='configs/reproducibility/fm_full_image_training_execution_recovery.2026-10-11_v3.json'
    original=native.read(native.ROOT/reference)
    new=copy.deepcopy(old);new.update(runtime_configuration='configs/runtime/fm_shcl_full413_completion_replay.v2.json',
        state_root='experiments/runs/fm_shcl_full413_completion_v2',output_root=old['output_root'].replace('_v1','_v2'),
        GPU_resource_lock=original['resource_lock'],GPU_mutex_reference_configuration=reference,
        handoff=dict(previous_owner=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline()),
            previous_state_path=old['state_root']+'/status.json',previous_configuration_sha256=native.sha(native.ROOT/path)))
    paths=[path,reference,'scripts/flow_matching/combined_readonly_gpu_slot_v1.py',
        'scripts/flow_matching/run_shcl_full413_completion_v2.py','scripts/flow_matching/register_shcl_full413_GPU_mutex_v2.py',
        'tests/test_fm_shcl_full413_GPU_mutex_v2.py']
    new['source_receipts']+=[dict(path=p,sha256=native.sha(native.ROOT/p)) for p in paths]
    replay.original.verify(new);replay.verify_gpu_mutex(new)
    if not replay.safe_phase(new,state,[dict(name=p.name(),exe=p.exe()) for p in actor.children(recursive=True)]):
        raise ValueError('Only unstarted idle waiting full413 owner may be replaced')
    exclusive(native.ROOT/new['runtime_configuration'],new)
    print(json.dumps(dict(original_full_training_replays=49,original_full_test_replays=364,GPU_mutex_added=True,scientific_sources_unchanged=True)))


if __name__=='__main__':main()
