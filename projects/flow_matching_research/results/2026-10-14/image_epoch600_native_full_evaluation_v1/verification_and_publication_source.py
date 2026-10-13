from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import run_flow_mismatching_full_image_evaluation_queue_v1 as image
from scripts.flow_matching import run_cfm_supplement_streaming_handoff_v2 as supplement

root = image.ROOT
folder = root / 'projects/flow_matching_research/results/2026-10-14/image_epoch600_native_full_evaluation_v1'
folder.mkdir(exist_ok=False)
read = image.read
write = lambda p, v: p.write_text(json.dumps(v, indent=2)+'\n', encoding='utf-8')
runtime = read(root / 'configs/runtime/fm_flow_mismatching_full_image_evaluation_queue.v1.json')
queue = read(root / runtime['queue'])
image.validate_queue(queue)
image.verify_receipts(runtime['source_receipts'])
image.verify_receipts(queue['source_receipts'])
case = next(c for c in queue['cases'] if c['id'] == 'table9_epoch_checkpoints__row05__mvtec__seed1103')
assert case['checkpoint_epoch'] == 600 and image.complete(case)
locationpath = root / case['output_directory'] / 'fully_verified_score_result.json'
location = read(locationpath)
reportpath = Path(location['report'])
report = read(reportpath)
terminalpath = reportpath.parent / 'terminal.json'
terminal = read(terminalpath)
candidatepath = Path(report['native_score_candidate'])
candidate = read(candidatepath)
assert terminal['returncode'] == 0 and terminal['resource_usage']['resource_guard_aborted'] is False
assert report['complete_original_test_images'] == 1725 and report['complete_original_categories'] == 15
assert report['max_absolute_heatmap_replay_error'] == report['max_absolute_image_score_replay_error'] == 0
assert report['original_author_equivalence_certified'] is False and report['strict_TAB_result'] is False
for source, name in [(locationpath,'fully_verified_score_location.json'), (reportpath,'independent_full_native_image_evaluation.json'),
                     (candidatepath,'native_score_candidate.json'), (terminalpath,'native_guarded_terminal.json')]:
    (folder / name).write_bytes(source.read_bytes())
evidence = dict(captured_utc=datetime.now(timezone.utc).isoformat(), citation='https://arxiv.org/abs/2605.23070',
                model='Flow Mismatching', dataset='MVTec-AD', case_id=case['id'], seed=1103, checkpoint_epoch=600,
                original_required_case_seed_evaluations=325, original_full_training_epochs=2000,
                complete_original_test_images=1725, complete_original_categories=15,
                metrics_percent=report['category_macro_percent'], paper_comparison=report['original_paper_comparison'],
                full_native_checkpoint_and_every_model_heatmap_score_and_probe_seed_replayed=True,
                max_absolute_native_heatmap_difference=0., max_absolute_native_score_difference=0.,
                independent_full_metrics_passed=True, report_sha256=image.sha(reportpath),
                checkpoint_sha256=report['full_original_checkpoint_sha256'],
                metric_aggregation='Unweighted macro mean of15 categories; units percentage; gaps percentage points.',
                uncertainty_boundary='One seed and correlated checkpoints do not establish independent-seed uncertainty.',
                F1_max_boundary='Full test-label oracle threshold sweep; not strict TAB calibration.',
                local_unreported_settings=report['local_unreported_settings'],
                final2000_epoch_completion_proved=False, causal_paper_gap_reason_certified=False,
                original_author_equivalence_certified=False, strict_TAB_result=False,
                all_original_experiments_complete=False)
write(folder / 'actual_epoch600_metrics_and_paper_comparison.json', evidence)
(folder / 'verification_and_publication_source.py').write_bytes(Path(__file__).read_bytes())
handofffolder = root / 'projects/flow_matching_research/results/2026-10-14/cfm_supplement_streaming_handoff_v2'
identity = read(handofffolder / 'actual_idle_handoff_and_new_owner.json')
p = psutil.Process(identity['pid'])
assert p.create_time() == identity['create_time'] and p.cmdline() == identity['command']
config = read(root / identity['configuration'])
original = read(root / config['original_runtime'])
state = read(root / original['state_root'] / 'status.json')
assert state['pid'] == p.pid and len(state['verified_original_ids']) >= 128 and not state['failed_preserved']
for pin in config['source_receipts']:
    assert supplement.cold.streaming_sha(root / pin['path']) == pin['sha256']
live_workers = []
for process in psutil.process_iter(['pid','name']):
    if 'python' not in (process.info['name'] or '').lower():
        continue
    try:
        command = process.cmdline()
        if ('scripts.flow_matching.run_cfm_ts_original_job' in command
                or any('run_saits.py' in s or 'run_anomaly.py' in s for s in command)):
            cpu = process.cpu_times()
            live_workers.append(dict(pid=process.pid, create_time=process.create_time(), command=command,
                                     cpu_seconds=cpu.user+cpu.system, private_bytes=process.memory_info().private))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
write(handofffolder / 'actual_live_handoff_and_training_handles.json', dict(
    checked_at_utc=datetime.now(timezone.utc).isoformat(), live_owner=dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),private_bytes=p.memory_info().private),
    original_controller_state=state, native_training_handles=live_workers,
    current_full80_failure_observation_state=read(root/'experiments/runs/fm_cfm_sine600_full80_failure_observation_v1/status.json'),
    training_interrupted=False, original_numeric_children_changed=False, all_original_experiments_complete=False))
for suffix in ['validation.xml']:
    tree = ET.parse(handofffolder / suffix).getroot()
    assert all(int(t.get('failures',0)) == int(t.get('errors',0)) == 0 for t in tree.iter('testsuite'))
paths = ['scripts/flow_matching/run_cfm_supplement_streaming_handoff_v2.py',
         'tests/test_cfm_supplement_streaming_handoff_v2.py',
         'configs/runtime/fm_cfm_supplement_streaming_handoff.v2.json']
paths += [p.relative_to(root).as_posix() for directory in [folder,handofffolder]
          for p in directory.iterdir() if p.is_file() and p.suffix in {'.py','.json','.xml'}]
parent = root.parent / 'flow-matching-research'
head = lambda cwd: subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'], text=True).strip()
write(root / 'tmp/cfm_supplement_and_image600_manifest_20261014.json',dict(previous_root=head(root),previous_parent=head(parent),paths=sorted(paths)))
print(json.dumps(dict(image600_metrics=evidence['metrics_percent'], paper_comparison=evidence['paper_comparison'], live_training_handles=[(r['pid'],r['cpu_seconds']) for r in live_workers], supplement_private_bytes=p.memory_info().private, exact_paths=len(paths))))
