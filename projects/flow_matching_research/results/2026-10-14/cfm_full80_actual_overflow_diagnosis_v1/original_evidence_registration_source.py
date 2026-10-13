import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_sine600_full80_failure_observation_v1 as native

root=native.ROOT
relative='projects/flow_matching_research/results/2026-10-14/cfm_full80_actual_overflow_diagnosis_v1'
folder=root/relative
folder.mkdir(exist_ok=False)
runtimepath='configs/runtime/fm_cfm_sine600_full80_failure_observation.v1.json'
runtime=native.read(root/runtimepath)
output=Path(runtime['output_root'])
paths=[runtimepath,'scripts/flow_matching/audit_cfm_full80_observed_arrays_v1.py',
       'tests/test_cfm_full80_observed_arrays_v1.py',runtime['canonical_job']['data_path'],
       str(output/'full80_native_failure_observation.json'),str(output/'all_original_grid_attempts.npz'),
       runtime['state_root']+'/receipt.json']+[str(output/('trajectory_%02d.json'%i)) for i in range(80)]
config=dict(native_configuration=runtimepath,output_report=relative+'/independent_every_grid_and_failure_audit.json',
            citation=runtime['citation'],source_receipts=[dict(path=p,sha256=native.sha(root/p)) for p in paths])
cfgpath='configs/reproducibility/fm_cfm_full80_observed_arrays.2026-10-14_v1.json'
with (root/cfgpath).open('x',encoding='utf-8') as stream:json.dump(config,stream,indent=2)
for source,name in [(output/'full80_native_failure_observation.json','full80_native_failure_observation.json'),
                    (root/runtime['state_root']/'receipt.json','native_guard_receipt.json'),
                    (root/runtime['state_root']/'status.json','native_terminal_state.json'),
                    (output/'trajectory_22.json','failed_original_trajectory103.json'),
                    (root/runtime['state_root']/'worker.stdout.log','native_worker.stdout.log'),
                    (root/runtime['state_root']/'worker.stderr.log','native_worker.stderr.log')]:
    (folder/name).write_bytes(source.read_bytes())
individual=folder/'individual_trajectory_receipts'
individual.mkdir()
for position in range(80):
    (individual/('trajectory_%02d.json'%position)).write_bytes((output/('trajectory_%02d.json'%position)).read_bytes())
(folder/'.gitattributes').write_text('*.log -diff -text\n',encoding='utf-8')
print(json.dumps(dict(configuration=cfgpath,original_trajectory_receipts=80,full_array_bytes=(output/'all_original_grid_attempts.npz').stat().st_size)))
