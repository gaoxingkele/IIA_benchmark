from pathlib import Path
import json
from scripts.flow_matching import audit_cfm_gp600_seed1103_full80_failure_v2 as current

root=current.ROOT
relative='projects/flow_matching_research/results/2026-10-14/cfm_GP6001103_actual_full80_failure_v2'
folder=root/relative
if folder.exists():
    assert {p.name for p in folder.iterdir()}=={'validation.xml'}, 'Preserve any actual execution'
    (folder/'validation.xml').rename(folder/'preserved_pre_registration_validation_failure.xml')
else:
    folder.mkdir(exist_ok=False)
guardpath='configs/runtime/fm_cfm_completion_replay_queue.v4.json'
guard=current.read(root/guardpath)
queuepath='configs/experiments/fm_cfm_ts_original_queue.v1.json'
queue=current.read(root/queuepath)
job=next(j for j in queue['jobs'] if j['id']==current.IDENTITY)
receipt=queue['state_root']+'/receipts/'+job['id']+'.json'
progress=job['output_directory']+'/progress.json'
stderr=queue['state_root']+'/logs/'+job['id']+'.stderr.log'
checkpoint=Path(job['artifact_directory'])/'model.pt'
model=current.read(root/job['model_config'])
config={k:guard[k] for k in ['settings','resource_lock','python','cpu_threads','environment_lock','original_scope']}
config.update(original_queue=queuepath,canonical_job=job,checkpoint_path=str(checkpoint),checkpoint_sha256=current.sha(checkpoint),
              guard_reference_configuration=guardpath,original_failed_receipt=receipt,original_failed_progress=progress,
              original_failed_stderr=stderr,solver_parameters={k:model['parameters'][k] for k in ['solver','rtol','atol','max_num_steps']},
              state_root='experiments/runs/fm_cfm_GP6001103_full80_failure_observation_v2',
              output_root='F:/aicoding/IIA_Data/experiments/cfm_GP6001103_full80_failure_observation_v2',
              citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
              diagnostic_boundary='One full80 original500-grid exactsolver observation from saved GP600/375000 model after actual numerical scoring failure at14. No retraining or relaxed tolerances; every failure remains explicit and prevents full original performance claim.')
pins=[queuepath,guardpath,job['model_config'],job['data_path'],receipt,progress,stderr,
      'scripts/flow_matching/audit_cfm_gp600_seed1103_full80_failure_v2.py','tests/test_cfm_gp600_seed1103_full80_failure_v2.py',
      'scripts/flow_matching/audit_cfm_gp600_seed1105_full80_failure_v1.py',
      'scripts/flow_matching/audit_cfm_sine600_full80_failure_observation_v1.py',
      'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py','scripts/flow_matching/audit_cfm_full80_observed_arrays_v2.py',
      'scripts/flow_matching/run_cfm_ts_original_job.py','scripts/flow_matching/register_cfm_ts_original.py',
      'src/iia_benchmark/models/fm_cfm_ts_original.py','src/iia_benchmark/models/fm_native_trajectory.py',config['environment_lock']]
config['source_receipts']=[dict(path=p,sha256=current.sha(root/p)) for p in pins]
cfgpath='configs/runtime/fm_cfm_GP6001103_full80_failure_observation.v2.json'
with (root/cfgpath).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
for source,name in [(root/receipt,'original_failed_source_receipt.json'),(root/progress,'original_failed_scoring_progress.json'),
                    (root/stderr,'original_failed_scoring.stderr.log')]:
    (folder/name).write_bytes(source.read_bytes())
(folder/'.gitattributes').write_text('*.log -diff -text\npreserved_pre_registration_validation_failure.xml -diff -text\n',encoding='utf-8')
print(json.dumps(dict(configuration=cfgpath,checkpoint_sha256=config['checkpoint_sha256'],original_failed_scoring_trajectories=14,required_original_trajectories=80,required_original_optimizer_updates=375000)))
