from copy import deepcopy
import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_conditional_GP6001104_full80_failure_v1 as audit

root,read,sha=audit.ROOT,audit.read,audit.sha
old=read(root/'configs/runtime/fm_cfm_GP6001105_full80_failure_observation.v1.json')
queue=read(root/old['original_queue']);job=next(j for j in queue['jobs'] if j['id']==audit.IDENTITY)
config=deepcopy(old)
config.update(canonical_job=job,checkpoint_path=str(Path(job['artifact_directory'])/'model.pt'),
    original_failed_receipt='experiments/runs/fm_cfm_ts_original_v1/receipts/'+audit.IDENTITY+'.json',
    original_failed_progress=job['output_directory']+'/progress.json',
    original_failed_stderr='experiments/runs/fm_cfm_ts_original_v1/logs/'+audit.IDENTITY+'.stderr.log',
    state_root='experiments/runs/fm_cfm_conditional_GP6001104_full80_failure_v1',
    output_root='F:/aicoding/IIA_Data/experiments/cfm_conditional_GP6001104_full80_failure_v1',
    diagnostic_boundary='Actual complete600/375000 conditional GP weights; source failed underflow after58/80 scored trajectories. One full80 original500-grid/solver observation, preserve all failures. No retraining, precision/guard/tolerance changes, shortened grid or successful-subset aggregate.')
config['checkpoint_sha256']=sha(config['checkpoint_path'])
files=[queue['source_receipts'][0]['path'],old['original_queue'],old['guard_reference_configuration'],
    job['model_config'],job['data_path'],config['original_failed_receipt'],config['original_failed_progress'],config['original_failed_stderr'],
    'scripts/flow_matching/audit_cfm_conditional_GP6001104_full80_failure_v1.py',
    'tests/test_cfm_conditional_GP6001104_full80_failure_v1.py']+[p['path'] for p in old['source_receipts'] if p['path'].startswith(('scripts/','src/','configs/reproducibility/'))]
config['source_receipts']=[dict(path=p,sha256=sha(root/p)) for p in dict.fromkeys(files)]
audit.verify(config)
path='configs/runtime/fm_cfm_conditional_GP6001104_full80_failure_observation.v1.json'
with (root/path).open('x') as stream:json.dump(config,stream,indent=2);stream.write('\n')
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP6001104_original_failure_observation_v1'
folder.mkdir(exist_ok=False)
for name,key in [('original_failed_receipt.json','original_failed_receipt'),('original_failed_progress.json','original_failed_progress'),
    ('original_failed_stderr.log','original_failed_stderr')]:
    (folder/name).write_bytes((root/config[key]).read_bytes())
(folder/'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.log -diff -text\n')
(folder/'README.md').write_text('Original Appendix conditional GP600/Sine seed1104 completed600 epochs/375000 updates and saved model.pt. Scoring failed with original Dopri5 dt underflow after58/80 trajectories, without resource abort. Preserve every input and raw failed attempt. Register one saved-model observation across all80x500 original held-out coordinates; original dtype, tolerances, maximum steps, native30GiB commit/16GiB physical/16GiB emergency/2GiB RSS/4GiB private guards remain. No retraining or repeated numeric attempts until pass. A complete finite five-seed conditional GP600 mean cannot currently be certified; never drop this seed or average only successful trajectories. All285 CFM and45/681/273 research scope remains.\n')
(folder/'full_five_finite_budget_comparison_not_yet_defined.json').write_text(json.dumps(dict(
    planned_full_cohort='configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_pairs.2026-10-14_v1.json',
    planned_full_cohort_sha256=sha(root/'configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_pairs.2026-10-14_v1.json'),
    required_seeds=[1103,1104,1105,1106,1107],actual_failed_seed=1104,original_failed_receipt_sha256=sha(root/config['original_failed_receipt']),
    full_GP600_five_seed_mean_MSE=None,failed_seed_excluded=False,all_original_experiments_complete=False),indent=2)+'\n')
print(json.dumps(dict(config=path,saved_checkpoint_sha256=config['checkpoint_sha256'],additional_training_seed_slots=0)))
