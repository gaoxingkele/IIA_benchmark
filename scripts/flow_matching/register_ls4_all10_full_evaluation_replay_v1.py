"""Freeze the full ten-model,500-evaluation original LS4 replay obligations."""
import copy
from datetime import datetime,timezone
import json
import psutil
from scripts.flow_matching.run_ls4_all10_full_evaluation_replay_v1 import ROOT,read,sha,verify,exclusive


def main():
    observation_path='configs/runtime/fm_ls4_physio_full_observation_dispatch.v2.json'
    observation=read(ROOT/observation_path);queue=read(ROOT/observation['original_queue'])
    launch=read(ROOT/observation['state_root']/'launch.json');p=psutil.Process(launch['pid'])
    if p.create_time()!=launch['create_time'] or p.cmdline()!=launch['command']:
        raise ValueError('Actual original full observation/training owner identity required')
    config=dict(observation_configuration=observation_path,original_queue=observation['original_queue'],
        original_jobs=copy.deepcopy(queue['jobs']),settings=copy.deepcopy(queue['settings']),resource_lock=queue['resource_lock'],
        worker_python=queue['python'],training_state_root=observation['state_root'],original_training_owner=launch,
        original_scope=copy.deepcopy(observation['original_scope']),required_full_native_evaluations=500,
        required_original_model_calls=37500,required_original_training_updates=750000,additional_training_seed_slots=0,
        poll_seconds=10,state_root='experiments/runs/fm_ls4_all10_full_evaluation_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/ls4_all10_full_evaluation_replay_v1',
        proof_root='projects/flow_matching_research/results/2026-10-12/ls4_all10_full_evaluation_replay_v1/completed_models')
    sources={p['path'] for p in observation['source_receipts']}
    sources.update([observation_path,'scripts/flow_matching/ls4_full_saved_evaluation_replay_v1.py',
        'scripts/flow_matching/run_ls4_all10_full_evaluation_replay_v1.py',
        'scripts/flow_matching/register_ls4_all10_full_evaluation_replay_v1.py',
        'tests/test_fm_ls4_all10_full_evaluation_replay_v1.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py'])
    config['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)]
    verify(config)
    path='configs/runtime/fm_ls4_all10_full_evaluation_replay.v1.json';exclusive(ROOT/path,config)
    folder=ROOT/'projects/flow_matching_research/results/2026-10-12/ls4_all10_full_evaluation_replay_v1'
    exclusive(folder/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path,configuration_sha256=sha(ROOT/path),source_pins=len(config['source_receipts']),
        full_original_jobs=10,full_original_native_evaluations_required=500,full_original_model_calls_required=37500,
        full_original_training_optimizer_updates_required=750000,original_scope=config['original_scope'],
        original_complete_constructor_prefix_and_original_eval_required=True,
        whole_saved_model_optimizer_scheduler_CPU_CUDA_NumPy_Python_loader_rng_and_captured_backend_flags_required=True,
        all_full_original_method_inputs_must_match_exactly=True,all_numeric_output_mismatches_remain_failed_strict_comparisons=True,
        original_all_resource_settings_and_gpu_mutex_preserved=True,actual_original_training_owner=launch,
        actual_complete_native_replay_evaluations=0,new_numeric_results_published=0,additional_training_seed_slots=0,
        all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=path,source_pins=len(config['source_receipts']),
        full_original_jobs=10,required_full_native_evaluations=500,required_original_model_calls=37500)))


if __name__=='__main__':main()
