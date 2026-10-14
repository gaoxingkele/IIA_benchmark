from copy import deepcopy
import json
from pathlib import Path
from scripts.flow_matching import run_cfm_saved_full_score_recovery_v1 as recovery

root, read, sha = recovery.ROOT, recovery.cold.original.read, recovery.cold.streaming_sha
primarypath = 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
primary = read(root/primarypath)
original = next(r['job'] for r in primary['routes'] if r['original_canonical_id'] == recovery.IDENTITY)
state = 'experiments/runs/fm_cfm_saved_score_recovery_v1'
job = dict(original, id=recovery.IDENTITY+'__saved_score_recovery_v1', recovery_original_job_id=recovery.IDENTITY,
    output_directory=state+'/jobs/'+recovery.IDENTITY,
    artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_saved_score_recovery_v1/'+recovery.IDENTITY)
checkpoint = str(Path(original['artifact_directory'])/'model.pt')
sourcebase = 'experiments/runs/fm_cfm_ts_original_v1'
config = {k:deepcopy(primary[k]) for k in ['python','cpu_threads','settings','resource_lock','environment_lock','original_scope']}
config.update(primary_runtime=primarypath, original_queue=primary['original_queue'], original_job=original, job=job,
    checkpoint_path=checkpoint, checkpoint_sha256=sha(checkpoint), state_root=state,
    family_lock='experiments/runs/fm_cfm_ts_original_cpu.lock',
    failed_receipt=sourcebase+'/receipts/'+recovery.IDENTITY+'.json',
    failed_progress=original['output_directory']+'/progress.json',
    failed_stderr=sourcebase+'/logs/'+recovery.IDENTITY+'.stderr.log',
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    boundary='Preserve original full400/62400 trained weights. Recover all20x500 scoring after WinError5 metadata-only interruption; no optimizer, target preprocessing, new seed or shortened solver.')
files = [primarypath, primary['original_queue'], original['data_path'], original['model_config'],
    config['failed_receipt'], config['failed_progress'], config['failed_stderr'],
    'scripts/flow_matching/run_cfm_saved_full_score_recovery_v1.py',
    'scripts/flow_matching/run_cfm_saved_score_full_replay_v1.py',
    'scripts/flow_matching/run_cfm_metadata_safe_original_job_v1.py',
    'scripts/flow_matching/queued_family_lock_v1.py', 'tests/test_cfm_saved_full_score_recovery_v1.py']
config['source_receipts'] = [dict(path=p,sha256=sha(root/p)) for p in files]
recovery.verify(config)
configpath = 'configs/runtime/fm_cfm_saved_score_recovery.v1.json'
runtimepath = 'configs/runtime/fm_cfm_saved_score_full_replay.v1.json'
runtime = deepcopy(primary)
runtime.update(previous_full285_configuration=primarypath, producer_configuration=configpath,
    selected_canonical_ids=[recovery.IDENTITY], state_root='experiments/runs/fm_cfm_saved_score_full_replay_v1',
    child_config_root='configs/reproducibility/fm_cfm_saved_score_full_replay_children_v1',
    child_state_root='experiments/runs/fm_cfm_saved_score_full_replay_children_v1',
    artifact_root='F:/aicoding/IIA_Data/experiments/cfm_saved_score_full_replay_v1')
runtime['routes'] = [dict(r,job=job,result_path=job['output_directory']+'/result.json')
    if r['original_canonical_id']==recovery.IDENTITY else r for r in primary['routes']]
runtime['source_receipts'] += config['source_receipts']+[dict(path=configpath,sha256='pending')]
for path, value in [(configpath,config), (runtimepath,runtime)]:
    if path==runtimepath: value['source_receipts'][-1]['sha256']=sha(root/configpath)
    with (root/path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2);stream.write('\n')
folder = root/'projects/flow_matching_research/results/2026-10-14/cfm_saved_GP4001106_score_recovery_v1'
folder.mkdir(exist_ok=False)
for name,path in [('preserved_original_failed_receipt.json',config['failed_receipt']),
    ('preserved_original_failed_progress.json',config['failed_progress']),
    ('preserved_original_failed_stderr.log',config['failed_stderr'])]:
    (folder/name).write_bytes((root/path).read_bytes())
(folder/'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n*.log -diff -text\n')
print(json.dumps(dict(config=configpath,replay=runtimepath,checkpoint_sha256=config['checkpoint_sha256'])))
