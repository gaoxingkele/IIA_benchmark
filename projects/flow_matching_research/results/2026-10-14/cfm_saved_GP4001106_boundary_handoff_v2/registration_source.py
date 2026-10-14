from copy import deepcopy
import json
from pathlib import Path
import psutil
from scripts.flow_matching import run_cfm_saved_score_queued_handoff_v2 as new

root,read,sha=new.ROOT,new.previous.cold.original.read,new.previous.cold.streaming_sha
oldpath='configs/runtime/fm_cfm_saved_score_recovery.v1.json'
old=read(root/oldpath)
statepath=old['state_root']+'/status.json';state=read(root/statepath)
p=psutil.Process(state['pid'])
assert p.pid==3252 and p.create_time()==1791940286.600826
descendants=[dict(pid=c.pid,name=c.name(),exe=c.exe()) for c in p.children(recursive=True)]
assert new.idle_allowed('original',state,(root/old['job']['output_directory']).exists(),descendants)
config=deepcopy(old)
queuepath='configs/experiments/fm_cfm_ts_original_queue.v1.json'
config.update(previous_scoring_configuration=oldpath,original_training_queue=queuepath,
    family_coordination_settings=read(root/queuepath)['resource_policy'],
    previous_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline()),previous_state_path=statepath,
    state_root='experiments/runs/fm_cfm_saved_score_queued_handoff_v2',
    handoff_state_root='experiments/runs/fm_cfm_saved_score_boundary_handoff_v2',
    complete_review_runtime='configs/runtime/fm_cfm_saved_score_full_replay.v1.json',maximum_review_window_seconds=900)
files=[oldpath,queuepath,config['complete_review_runtime'],
    'scripts/flow_matching/run_cfm_saved_score_queued_handoff_v2.py',
    'scripts/flow_matching/reserve_cfm_complete_review_window_v1.py',
    'scripts/flow_matching/run_timegan_corrected_observer_dispatch_v2.py',
    'tests/test_cfm_saved_score_queued_handoff_v2.py']
config['source_receipts'] += [dict(path=f,sha256=sha(root/f)) for f in files]
new.verify(config)
path='configs/runtime/fm_cfm_saved_score_queued_handoff.v2.json'
with (root/path).open('x') as stream:json.dump(config,stream,indent=2);stream.write('\n')
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_saved_GP4001106_boundary_handoff_v2'
folder.mkdir(exist_ok=True)
failed_validation=folder/'validation.xml'
if failed_validation.exists():
    failed_validation.rename(folder/'preserved_pre_registration_state_race_validation_failure.xml')
(folder/'verified_actual_old_idle_controller.json').write_text(json.dumps(dict(state=state,pid=p.pid,
    create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private),indent=2)+'\n')
(folder/'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
print(json.dumps(dict(config=path,original_scoring_commit_bytes=config['settings']['minimum_commit_headroom_bytes'],
    metadata_only_family_queue_commit_bytes=config['family_coordination_settings']['minimum_commit_headroom_bytes'])))
