"""Register three exact idle MaelNet owners and preserve their pending log stages."""
import json
from pathlib import Path
import psutil

from scripts.flow_matching import run_cached_maelnet_prepared_stage_monitor_v1 as runtime
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    root=runtime.base.ROOT; observed=runtime.base.read(root/'tmp/maelnet_prepared_stage_identity_20261012_v1.json')
    for name,row in zip(('original150','seed1105','seed1106'),observed):
        original=runtime.base.read(root/row['runtime']); state=runtime.base.read(root/row['state_path'])
        process=psutil.Process(row['owner']['pid'])
        if process.create_time()!=row['owner']['create_time'] or process.cmdline()!=row['owner']['command']:
            raise ValueError('Exact originally observed idle process required')
        if state['state']!='waiting_for_memory' or state['active_pid'] is not None or state['requested_job']!=row['pending_job']['id']:
            raise ValueError('Exact originally requested unstarted stage required')
        if len(row['pending_stage_directories'])!=1: raise ValueError('Only one exact prepared first stage required')
        output=root/row['pending_job']['output_directory']; workspace=output/'author_workspace'
        corrected=root/runtime.base.read(root/row['queue'])['settings']['corrected_source']
        pins=[]
        for path in workspace.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts: continue
            relative=path.relative_to(workspace).as_posix()
            # Historical sample results/logs were copied from the corrected
            # source. A new training artifact or modified sample must reject.
            if not (corrected/relative).is_file() or runtime.base.sha(path)!=runtime.base.sha(corrected/relative):
                raise ValueError('Pending workspace contains actual new/modified output: '+relative)
            pins.append(dict(relative_path=relative,sha256=runtime.base.sha(path)))
        filename='configs/runtime/fm_maelnet_'+name+'_cached_prepared_stage_monitor.v1.json'
        files=[row['runtime'],row['queue'],'scripts/flow_matching/run_cached_maelnet_prepared_stage_monitor_v1.py',
            'scripts/flow_matching/register_cached_maelnet_prepared_stage_monitors_v1.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v2.py',
            'scripts/flow_matching/cached_resource_snapshot_v1.py']
        config=dict(runtime_configuration=filename,entry_module=row['owner']['command'][5],
            original_runtime_configuration=row['runtime'],original_arguments=['--runtime-config',row['runtime']],
            original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
            settings=original['settings'],resource_lock=original['resource_lock'],historical_identity=None,
            expected_state_identity=dict(runtime_sha256=runtime.base.sha(root/row['runtime']),queue_sha256=runtime.base.sha(root/row['queue'])),
            pending_job=row['pending_job'],workspace_snapshot=pins,
            prepared_stage=dict(path=row['pending_stage_directories'][0]['path'],requested_command=state['requested_command']),
            prepared_stage_backup='F:/aicoding/IIA_Data/experiments/maelnet_cached_prepared_stage_monitors_v1/'+name+'/prepared_stage',
            monitor_state_root='experiments/runs/fm_maelnet_'+name+'_cached_prepared_stage_monitor_v1',
            handoff=dict(previous_owner=row['owner'],previous_state_path=row['state_path']),
            source_receipts=[dict(path=p,sha256=runtime.base.sha(root/p)) for p in files],
            boundary='All150 canonical jobs and unchanged native stage functions/settings/source/code/data/seeds '
                'remain. Only a verified idle parent is replaced. Exact empty pre-admission stage logs are '
                'moved to a preserved F: backup; the unchanged original stage scanner recreates its exact '
                'original path. Historical source sample outputs stay byte-identical. No native child stopped.')
        runtime.verify(config)
        if not runtime.pending_unstarted(config): raise ValueError('Prepared stage failed exact unstarted manifest proof')
        exclusive(root/filename,config)
        print(json.dumps(dict(role=name,pid=process.pid,full_canonical_jobs=150,workspace_files=len(pins),native_training_stopped=False)))


if __name__=='__main__':main()
