from pathlib import Path
from copy import deepcopy
import json
from scripts.flow_matching import run_cfm_metadata_permission_full_retry_v1 as producer
from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as replay
from scripts.flow_matching.run_cfm_streaming_idle_handoff_v3 import streaming_sha

root=producer.ROOT;relative='projects/flow_matching_research/results/2026-10-14/cfm_unsaved_permission_full400_recovery_v1';folder=root/relative
assert not folder.exists()
original_path='configs/experiments/fm_cfm_ts_original_queue.v1.json';canonical=producer.original.read(root/original_path);queue=deepcopy(canonical)
selected=[];evidence=[]
for j in queue['jobs']:
    if j['id'] not in ['cfm_ts_original__main_text__bb_paper_literal_e400__sine__seed'+str(s) for s in (1104,1106)]:continue
    old=j['id'];selected.append(old+'__metadata_permission_full_retry_v1')
    j.update(id=selected[-1],recovery_original_job_id=old,
        output_directory='experiments/runs/fm_cfm_metadata_permission_full_retry_v1/jobs/'+old,
        artifact_directory='F:/aicoding/IIA_Data/experiments/cfm_metadata_permission_full_retry_v1/'+old)
    evidence.append(dict(original_canonical_id=old,receipt=canonical['state_root']+'/receipts/'+old+'.json',
        progress=next(k['output_directory'] for k in canonical['jobs'] if k['id']==old)+'/progress.json',
        stderr=canonical['state_root']+'/logs/'+old+'.stderr.log',partial_epoch_credit=0,saved_checkpoint_exists=False))
queuepath='configs/experiments/fm_cfm_metadata_permission_full_retry_queue.v1.json'
queue.update(state_root='experiments/runs/fm_cfm_metadata_permission_full_retry_v1',recovery_original_queue=original_path,
    recovery_original_queue_sha256=streaming_sha(root/original_path),recovery_job_ids=selected,
    recovery_evidence=evidence,recovery_selection='Both unsaved Windows metadata failures; original full400 budgets, no metric-based selection')
newpins=['scripts/flow_matching/run_cfm_metadata_safe_original_job_v1.py','scripts/flow_matching/run_cfm_metadata_permission_full_retry_v1.py',
    'scripts/flow_matching/run_cfm_metadata_permission_replay_queue_v1.py','tests/test_cfm_metadata_permission_full_retry_v1.py']
pins=[dict(path=p,sha256=streaming_sha(root/p)) for p in newpins]
for row in evidence:
    for key in ['receipt','progress','stderr']:pins.append(dict(path=row[key],sha256=streaming_sha(root/row[key])))
queue['source_receipts']+=pins
with (root/queuepath).open('x',encoding='utf-8') as f:json.dump(queue,f,indent=2)
runtime=dict(controllers={'original':dict(queue=original_path,queue_sha256=streaming_sha(root/original_path)),
    'recovery':dict(queue=queuepath,queue_sha256=streaming_sha(root/queuepath),selected_job_ids=selected)},
    family_lock='experiments/runs/fm_cfm_ts_original_cpu.lock',source_receipts=pins+canonical['source_receipts'],
    original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    boundary='Original full285 slots and all numeric algorithms, data, seeds, epochs, updates and guards preserved. Exactly two unsaved permission failures restart their full400 budgets in fresh paths; only metadata writes are retried.')
runtimepath='configs/runtime/fm_cfm_metadata_permission_full_retry.v1.json'
with (root/runtimepath).open('x',encoding='utf-8') as f:json.dump(runtime,f,indent=2)
producer.verify_runtime(runtime)
previouspath='configs/runtime/fm_cfm_completion_replay_queue.v4.json';previous=producer.original.read(root/previouspath);review=deepcopy(previous)
review.update(routes=replay.expected_routes(previous,queue),previous_full285_configuration=previouspath,producer_configuration=runtimepath,
    selected_canonical_ids=[j['recovery_original_job_id'] for j in queue['jobs'] if j['id'] in selected],
    state_root='experiments/runs/fm_cfm_metadata_permission_full_replay_v1',
    child_state_root='experiments/runs/fm_cfm_metadata_permission_full_replay_v1/jobs',
    child_config_root='configs/reproducibility/fm_cfm_metadata_permission_full_replay_queue_v1',
    artifact_root='F:/aicoding/IIA_Data/experiments/cfm_metadata_permission_full_replay_v1')
review['source_receipts']+=pins+[dict(path=runtimepath,sha256=streaming_sha(root/runtimepath)),dict(path=queuepath,sha256=streaming_sha(root/queuepath))]
reviewpath='configs/runtime/fm_cfm_metadata_permission_full_replay.v1.json'
with (root/reviewpath).open('x',encoding='utf-8') as f:json.dump(review,f,indent=2)
folder.mkdir()
(folder/'.gitattributes').write_text('*.log -diff -text\n*.py -text\n',encoding='utf-8')
for row in evidence:
    target=folder/row['original_canonical_id'];target.mkdir()
    for key in ['receipt','progress','stderr']:(target/('original_'+key+('.log' if key=='stderr' else '.json'))).write_bytes((root/row[key]).read_bytes())
(folder/'registration_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(producer_configuration=runtimepath,review_configuration=reviewpath,selected_full_retries=len(selected),full_original_slots=len(queue['jobs']))))
