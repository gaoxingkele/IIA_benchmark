"""Freeze all413 released commands; no reduced workers/epochs or discarded grid."""
import json
import math
from pathlib import Path

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    registry_path='configs/experiments/fm_shcl_released_script_registry.v1.json'; registry=native.read(native.ROOT/registry_path)
    native_data='configs/runtime/fm_shcl_author_data.v1.json'; data_config=native.read(native.ROOT/native_data)
    data_path=Path(data_config['view_root'])/'full_data_preparation_audit.json'; data=native.read(data_path)
    fixture_config='configs/runtime/fm_shcl_native_observer_audit.v1.json'; fixture=native.read(native.ROOT/fixture_config)
    fixture_path=native.ROOT/fixture['output_root']/'full_observer_fidelity_fixture.json'
    fixture_report=native.read(fixture_path)
    if not data['passed'] or data['full_native_loader_modes_audited']!=52 or not fixture_report['passed']:
        raise ValueError('All52 actual native data modes and observed/unobserved full-batch equivalence required')
    environment_path='configs/acquisition/fm_shcl_author_environment.v1.json'; env=native.read(native.ROOT/environment_path)
    base=native.read(native.ROOT/registry['model_config'])['runtime_output_root']; jobs=[]; training={}
    for index,command in enumerate(registry['commands']):
        dataset=command['dataset']; window=command['win_size']; epochs=command['num_epochs']
        stem='shcl_released__'+dataset.lower()+'__w'+str(window)+'__e'+str(epochs)+'__seed1'
        key=(dataset,window,epochs)
        if command['mode']=='train':
            identifier=stem+'__train'; dependency=None; training[key]=identifier
        else:
            if key not in training: raise ValueError('Every exact source test requires its prior full training')
            identifier=stem+'__test_ratio'+format(command['anormly_ratio'],'.8g').replace('.','p'); dependency=training[key]
        audit=next(r for r in data['records'] if r['dataset']==dataset)
        modes={r['mode']:r for r in audit['native_loaders'] if r['window']==window}
        if set(modes)!={'train','val','test','thre'}: raise ValueError('Complete full-window source profile required')
        checkpoint=base+'/checkpoints/'+stem; argv=list(command['argv'])
        for name,value in [('--data_path',audit['data_path']),('--model_save_path',checkpoint)]:
            position=argv.index(name)+1; argv[position]=value
        # Every non-filesystem CLI token remains exactly the registered source.
        for i,token in enumerate(command['argv']):
            if i and command['argv'][i-1] in ('--data_path','--model_save_path'): continue
            if argv[i]!=token: raise ValueError('Original CLI seed/budget/ratio/architecture flags changed')
        jobs.append(dict(id=identifier,source_command_index=index,source_script=command['script'],
            mode=command['mode'],dataset=dataset,win_size=window,num_epochs=epochs,seed=1,batch_size=64,
            anormly_ratio=command['anormly_ratio'],unused_source_threshold=command['ignored_script_threshold'],
            argv=argv,depends_on=dependency,checkpoint_directory=checkpoint,output_directory=base+'/jobs/'+identifier,
            expected_optimizer_updates=math.ceil(modes['train']['full_native_windows']/64)*epochs,
            expected_validation_batches=math.ceil(modes['val']['full_native_windows']/64)*epochs,
            expected_threshold_windows=modes['thre']['full_native_windows'],expected_test_windows=modes['test']['full_native_windows'],
            source_dropped_tail_rows=modes['test']['source_dropped_tail_rows'],native_DataLoader_workers=8,
            full_data_files=audit['output_files'],synthetic_fixture=False))
    if len(jobs)!=413 or len({j['id'] for j in jobs})!=413 or len(training)!=49:
        raise ValueError('Every unique49train/364test source command required')
    files=[registry_path,native_data,str(data_path),fixture_config,fixture_path.relative_to(native.ROOT).as_posix(),
        environment_path,env['root']+'/environment_acquisition_report.json',registry['model_config'],
        'scripts/flow_matching/shcl_native_observer_v1.py','scripts/flow_matching/run_shcl_native_job_v1.py',
        'scripts/flow_matching/register_shcl_native_script_queue_v1.py','scripts/flow_matching/run_shcl_native_script_queue_v1.py',
        'scripts/flow_matching/shcl_author_audits_v1.py','scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py']
    queue=dict(schema_version=1,id='fm_shcl_released_native_full413_v1',model_config=registry['model_config'],script_registry=registry_path,
        original_scope=registry['original_scope'],source_commit=registry['source_commit'],jobs=jobs,
        environment=data_config['environment'],python=env['python'],
        state_root='experiments/runs/fm_shcl_native_script_queue_v1',resource_lock='experiments/runs/fm_shcl_native_eight_loader_workers.lock',
        settings=dict(minimum_available_memory_bytes=32*2**30,minimum_commit_headroom_bytes=56*2**30,
            emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=32*2**30,maximum_worker_private_bytes=40*2**30,
            gpu_uuid='GPU-b62da557-0167-4b14-25ef-7b6eab536ef9',minimum_gpu_free_bytes=8*2**30,
            emergency_gpu_free_bytes=2*2**30,cpu_threads=2),
        capacity_basis='First original8-worker SHCL native lane: actual single native Torch1.11/CUDA11.3 '
            'operator peak private bytes '+str(native.read(native.ROOT/'experiments/runs/fm_shcl_author_operators_v1/receipt.json')['peak_tree_private_bytes'])+
            '. Nine native interpreter processes plus full dataset replication must be reserved.56GiB '
            'commit admission retains16GiB system emergency reserve and permits40GiB tree private limit. '
            'No existing controller policy is changed; workers, full datasets, complete epochs and source CLI stay exact.',
        source_receipts=[dict(path=p,sha256=native.sha(native.ROOT/p)) for p in files],
        boundary='All49train/364test exact released VAE commands, separate original seed1 processes, '
            'original8 DataLoader workers and all full dataset rows. CUDA device visibility remapped to '
            'the sole registered UUID; source --gpu0 unchanged. This is the refactored VAE release; '
            'paper-era Transformer/LSTM, strict full-point TAB and independent trained-checkpoint '
            'replay remain separate requirements. No original-source test-grid result is called a strict benchmark.')
    exclusive(native.ROOT/'configs/experiments/fm_shcl_native_script_queue.v1.json',queue)
    print(json.dumps(dict(original_training_commands=49,original_test_commands=364,native_workers_each=8,
        original_full_epochs_across_trainings=sum(j['num_epochs'] for j in jobs if j['mode']=='train'))))


if __name__=='__main__':main()
