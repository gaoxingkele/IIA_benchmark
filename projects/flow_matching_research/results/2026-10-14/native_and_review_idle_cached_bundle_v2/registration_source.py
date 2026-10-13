import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_idle_native_and_review_cached_bundle_v2 as current

root=current.previous.ROOT
folder=root/'projects/flow_matching_research/results/2026-10-14/native_and_review_idle_cached_bundle_v2'
folder.mkdir(exist_ok=False)
specs=[('GTGAN_energy',2976,1791671245.228836,'experiments/runs/fm_gtgan_native_original_energy_v1/status.json'),
       ('TCCM_capacity',32692,1791741754.077063,'experiments/runs/fm_tccm_full_image_capacity_recovery_v1/status.json'),
       ('GRASP_tau0_review',32236,1791720506.5041308,'experiments/runs/fm_grasp_tau01104_full_array_audit_v1/status.json'),
       ('GRASP_tau05_review',21956,1791729456.3561015,'experiments/runs/fm_grasp_tau051104_full_array_audit_v1/status.json')]
roles={}
pins=['scripts/flow_matching/run_idle_native_and_review_cached_bundle_v2.py','tests/test_idle_native_and_review_cached_bundle_v2.py',
      'scripts/flow_matching/run_idle_full_scope_cached_bundle_v1.py','scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
      'scripts/flow_matching/cached_resource_snapshot_v1.py','scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
      'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py']
for name,pid,ctime,statepath in specs:
    p=psutil.Process(pid);assert p.create_time()==ctime
    command=p.cmdline();index=command.index('-m');module=command[index+1];arguments=command[index+2:]
    state=current.previous.cold.original.read(root/statepath)
    assert state['pid']==pid and state['create_time']==ctime
    descendants=[dict(name=c.name(),exe=c.exe()) for c in p.children(recursive=True)]
    assert current.previous.idle_allowed('original',state,False,descendants)
    assert module in current.EXTRA_HOSTS
    key=next(k for k in ['--queue','--runtime-config','--config'] if k in arguments)
    parameter=arguments[arguments.index(key)+1]
    data=current.previous.cold.original.read(root/parameter)
    if name=='TCCM_capacity':
        queuepath=data['configuration'];itemskey='capacity_cases'
    elif name.startswith('GRASP'):
        queuepath=parameter;itemskey='full_results'
        assert not (root/data['output_directory']).exists()
        assert not (root/data['state_root']/'stdout.log').exists()
        pins.append(data['queue'])
    else:
        queuepath=parameter;itemskey='jobs'
    queue=current.previous.cold.original.read(root/queuepath)
    target=folder/name;target.mkdir()
    (target/'original_live_status_before_handoff.json').write_bytes((root/statepath).read_bytes())
    roles[name]=dict(entry_module=module,original_arguments=arguments,previous_owner=dict(pid=pid,create_time=ctime,command=command),
                     previous_state_path=statepath,previous_private_bytes=p.memory_info().private,original_queue=queuepath,
                     queue_items_key=itemskey,required_original_slots=len(queue[itemskey]))
    pins += [parameter,queuepath,module.replace('.','/')+'.py']
assert roles['GTGAN_energy']['required_original_slots']==4
assert roles['TCCM_capacity']['required_original_slots']==1
assert all(roles[name]['required_original_slots']==1 for name in ['GRASP_tau0_review','GRASP_tau05_review'])
config=dict(original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),roles=roles,
            native_Energy_commands=4,complete_original_TCCM_capacity_cases=1,selected_full_original_GRASP_reviews=2,
            handoff_state_root='experiments/runs/fm_idle_native_review_cached_bundle_v2',
            source_receipts=[dict(path=path,sha256=current.previous.cold.streaming_sha(root/path)) for path in sorted(set(pins))])
cfgpath='configs/runtime/fm_native_and_review_idle_cached_bundle.v2.json'
with (root/cfgpath).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
(folder/'validation.xml').write_bytes((root/'tmp/idle_native_review_bundle_v2_validation.xml').read_bytes())
print(json.dumps(dict(configuration=cfgpath,previous_private_bytes=sum(r['previous_private_bytes'] for r in roles.values()),original_commands=4,full_capacity_cases=1,full_reviews=2)))
