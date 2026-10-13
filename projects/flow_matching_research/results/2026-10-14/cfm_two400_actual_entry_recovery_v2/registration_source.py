from pathlib import Path
import copy
import json
from datetime import datetime, timezone
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.run_cfm_uncovered_recovery_completion_v1 import expected_routes

root = cold.ROOT
read, sha = cold.original.read, cold.streaming_sha
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_two400_actual_entry_recovery_v2'
folder = root / relative
folder.mkdir(exist_ok=True)
assert not (root / 'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json').exists()
old_runtime_path = root / 'configs/runtime/fm_cfm_metadata_permission_full_retry.v1.json'
old_runtime = read(old_runtime_path)
old_queue = read(root / old_runtime['controllers']['recovery']['queue'])
old_base = root / old_queue['state_root']
evidence = []
for source in [old_base / 'status.json', *sorted((old_base / 'receipts').glob('*.json')),
               *sorted((old_base / 'logs').glob('*.log'))]:
    name = source.name
    if 'seed1104' in name or 'seed1106' in name:
        seed = '1104' if 'seed1104' in name else '1106'
        name = 'seed' + seed + ('.receipt.json' if source.parent.name == 'receipts' else
                               '.stderr.log' if name.endswith('.stderr.log') else '.stdout.log')
    target = folder / 'preserved_actual_entry_failure' / source.parent.name / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): assert target.read_bytes() == source.read_bytes()
    else: target.write_bytes(source.read_bytes())
    evidence.append({'path': source.relative_to(root).as_posix(), 'sha256': sha(source)})

queue = copy.deepcopy(old_queue)
queue['state_root'] = 'experiments/runs/fm_cfm_metadata_entry_full_retry_v2'
chosen = set(old_runtime['controllers']['recovery']['selected_job_ids'])
newids = []
for job in queue['jobs']:
    if job['id'] not in chosen: continue
    job['id'] = job['recovery_original_job_id'] + '__metadata_entry_full_retry_v2'
    job['output_directory'] = queue['state_root'] + '/jobs/' + job['id']
    job['artifact_directory'] = 'F:/aicoding/IIA_Data/experiments/cfm_metadata_entry_full_retry_v2/' + job['id']
    newids.append(job['id'])
queue['recovery_job_ids'] = newids
queuepath = root / 'configs/experiments/fm_cfm_metadata_entry_full_retry_queue.v2.json'
def write(path, data):
    if path.exists():
        assert path.read_text(encoding='utf-8') == json.dumps(data, indent=2)
        return
    with path.open('x', encoding='utf-8') as stream: json.dump(data, stream, indent=2)
write(queuepath, queue)
new_sources = [root / 'scripts/flow_matching/run_cfm_metadata_safe_original_job_v2.py',
               root / 'scripts/flow_matching/run_cfm_metadata_entry_full_retry_v2.py',
               root / 'scripts/flow_matching/run_cfm_metadata_entry_full_replay_v2.py',
               root / 'tests/test_cfm_metadata_entry_full_retry_v2.py',
               root / 'scripts/flow_matching/queued_family_lock_v1.py',
               root / 'scripts/flow_matching/run_cfm_metadata_queued_family_handoff_v2.py']
newpins = [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in new_sources]
runtime = copy.deepcopy(old_runtime)
runtime['controllers']['recovery'] = dict(queue=queuepath.relative_to(root).as_posix(),
                                         queue_sha256=sha(queuepath), selected_job_ids=newids)
runtime['source_receipts'].extend(newpins)
runtimepath = root / 'configs/runtime/fm_cfm_metadata_entry_full_retry.v2.json'
write(runtimepath, runtime)
control = dict(runtime=runtimepath.relative_to(root).as_posix(),
    failed_runtime=old_runtime_path.relative_to(root).as_posix(),
    failed_state=(old_base / 'status.json').relative_to(root).as_posix(),
    failed_parent={'pid': 3848, 'create_time': 1791933520.3622932},
    original_scope={'papers': 45, 'method_baseline_ablation_records': 681, 'original_data_task_records': 273},
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    boundary='Proven zero-training KeyError in inline CLI dispatch only. Preserve failed evidence and every full285 slot; use exact400/6400 and20x500 grids with unchanged family lock and resource guards.',
    source_receipts=evidence + newpins + [{'path': runtimepath.relative_to(root).as_posix(), 'sha256': sha(runtimepath)}])
write(root / 'configs/runtime/fm_cfm_metadata_entry_controller.v2.json', control)
primary = read(root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json')
replay = copy.deepcopy(read(root / 'configs/runtime/fm_cfm_metadata_permission_full_replay.v1.json'))
replay['producer_configuration'] = runtimepath.relative_to(root).as_posix()
replay['routes'] = expected_routes(primary, queue)
for key in ['state_root', 'child_state_root', 'child_config_root', 'artifact_root']:
    replay[key] = replay[key].replace('metadata_permission_full_replay', 'metadata_entry_full_replay').replace('_v1', '_v2')
replay['source_receipts'].extend(newpins + [{'path': runtimepath.relative_to(root).as_posix(), 'sha256': sha(runtimepath)},
                                         {'path': queuepath.relative_to(root).as_posix(), 'sha256': sha(queuepath)}])
replay['boundary'] = control['boundary'] + ' Only two corrected execution routes change; native plus independent review guards/tolerance and other283 routes retained.'
write(root / 'configs/runtime/fm_cfm_metadata_entry_full_replay.v2.json', replay)
(folder / 'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.log -diff -text\n', encoding='utf-8')
print(json.dumps({'new_ids': newids, 'full_scope_slots': len(queue['jobs']), 'original_resource_policy': queue['resource_policy'],
                  'new_replay_paths': {key: replay[key] for key in ['state_root', 'child_state_root', 'child_config_root', 'artifact_root']}}, indent=2))
