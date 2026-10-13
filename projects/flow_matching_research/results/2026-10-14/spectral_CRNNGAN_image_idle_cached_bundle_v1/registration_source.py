import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_idle_full_scope_cached_bundle_v1 as current

root=current.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/spectral_CRNNGAN_image_idle_cached_bundle_v1'
folder.mkdir(exist_ok=False)
specs=[('table2',24296,1791651863.9159133,'experiments/runs/fm_spectral_table2_method_gated_v1/status.json'),
       ('table2_full_preflight',30560,1791651863.8515537,'experiments/runs/fm_spectral_table2_spectral_preflight_v3/status.json'),
       ('table3',27496,1791684391.27287,'experiments/runs/fm_spectral_table3_original_v3/status.json'),
       ('table45',32612,1791652953.3819592,'experiments/runs/fm_spectral_table45_original_v1/status.json'),
       ('crnngan',11904,1791679562.3354588,'experiments/runs/fm_crnngan_native_classical_released_v3/status.json'),
       ('image325',37664,1791715156.9441707,'experiments/runs/fm_flow_mismatching_full_image_evaluation_v1/status.json')]
roles={}
pins=['scripts/flow_matching/run_idle_full_scope_cached_bundle_v1.py','tests/test_idle_full_scope_cached_bundle_v1.py',
      'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py','scripts/flow_matching/cached_resource_snapshot_v1.py',
      'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
      'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py','scripts/flow_matching/run_spectral_original_queue.py']
for name,pid,ctime,statepath in specs:
    p=psutil.Process(pid);assert p.create_time()==ctime
    command=p.cmdline();index=command.index('-m');module=command[index+1];args=command[index+2:]
    assert module in current.ALLOWED
    state=current.cold.original.read(root/statepath)
    assert state['pid']==pid and state['create_time']==ctime
    descendants=[dict(name=c.name(),exe=c.exe()) for c in p.children(recursive=True)]
    assert current.idle_allowed('original',state,False,descendants)
    key=next(k for k in ['--runtime-config','--config','--queue'] if k in args)
    parameter=args[args.index(key)+1]
    data=current.cold.original.read(root/parameter)
    queuepath=parameter if key=='--queue' else data['queue']
    queue=current.cold.original.read(root/queuepath)
    itemskey='cases' if name=='image325' else 'jobs'
    target=folder/name;target.mkdir()
    (target/'original_live_status_before_handoff.json').write_bytes((root/statepath).read_bytes())
    roles[name]=dict(entry_module=module,original_arguments=args,previous_owner=dict(pid=pid,create_time=ctime,command=command),
                     previous_state_path=statepath,previous_private_bytes=p.memory_info().private,
                     original_queue=queuepath,queue_items_key=itemskey,required_original_slots=len(queue[itemskey]),
                     performance_queue_role=name!='table2_full_preflight')
    pins += [parameter,queuepath,module.replace('.','/')+'.py']
unique={role['original_queue']:role['required_original_slots'] for role in roles.values()}
assert sum(unique.values())==412
config=dict(original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),roles=roles,
            unique_original_scientific_slots=412,additional_full_preflight_roles=1,
            handoff_state_root='experiments/runs/fm_idle_full_scope_cached_bundle_v1',
            source_receipts=[dict(path=path,sha256=current.cold.streaming_sha(root/path)) for path in sorted(set(pins))])
cfgpath='configs/runtime/fm_spectral_CRNNGAN_image_idle_cached_bundle.v1.json'
with (root/cfgpath).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
(folder/'validation.xml').write_bytes((root/'tmp/idle_full_scope_cached_bundle_validation_20261014.xml').read_bytes())
print(json.dumps(dict(configuration=cfgpath,unique_original_scientific_slots=412,previous_private_bytes=sum(r['previous_private_bytes'] for r in roles.values()))))
