from datetime import datetime, timezone
import json
import math
from pathlib import Path
import psutil
from scripts.flow_matching.audit_pi_training_log_units_v1 import ROOT, sha, audit

root = ROOT
relative = 'projects/flow_matching_research/results/2026-10-14/pi_actual_training_log_units_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
queuepath = root / 'configs/experiments/fm_pi_causal42_full_incident_recovery_queue.v5.json'
queue = json.loads(queuepath.read_text())
identity = 'pi_author__causal_axis_repaired__psm__full__seed42__full_incident_recovery_v5'
case = next(c for c in queue['jobs'] if c['job']['id'] == identity)
job = case['job']
worker = psutil.Process(22432)
assert abs(worker.create_time() - 1791927002.4746633) < .001
command = worker.cmdline()
assert identity in command and 'scripts.flow_matching.run_resource_recovery_job' in command
logroot = root / 'experiments/runs/fm_pi_causal42_full_incident_recovery_v5/logs' / identity
for name in ['stdout.log', 'stderr.log']:
    (folder / ('original_' + name)).write_bytes((logroot / name).read_bytes())
sourcepath = root / job['runtime'] / 'main.py'
sources = [queuepath, sourcepath, folder / 'original_stdout.log', folder / 'original_stderr.log',
           root / 'scripts/flow_matching/audit_pi_training_log_units_v1.py',
           root / 'tests/test_pi_training_log_units_v1.py']
author = job['author_config']
expected_windows = job['data_audit']['expected_shapes']['train'][0] - author['data']['win_size'] + 1
steps = math.ceil(expected_windows / author['data']['batch_size'])
assert steps == 518 and author['training']['n_epochs'] == 3
config = dict(author_training_source=sourcepath.relative_to(root).as_posix(),
    frozen_stdout=(folder / 'original_stdout.log').relative_to(root).as_posix(),
    frozen_stderr=(folder / 'original_stderr.log').relative_to(root).as_posix(),
    canonical_id=case['original_id'], execution_id=identity, required_epochs=3, steps_per_epoch=steps,
    full_training_windows=expected_windows, window=author['data']['win_size'], batch_size=author['data']['batch_size'],
    scientific_worker=dict(pid=worker.pid, create_time=worker.create_time(), command=command),
    original_scope={'papers': 45, 'method_baseline_ablation_records': 681, 'original_data_task_records': 273},
    citation=job.get('citation', 'Original pinned Pi historical mirror training source; corresponding journal equivalence remains unverified.'),
    output_report=relative + '/actual_source_and_live_worker_timing_unit_audit.json',
    boundary='Read original logs only. Constant counter inflates interior100-update intervals; epoch starts and validation excluded. No completed model or benchmark claim; no frozen source edits or worker restarts.',
    source_receipts=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in sources])
configpath = root / 'configs/reproducibility/fm_pi_actual_training_log_units.2026-10-14_v1.json'
with configpath.open('x', encoding='utf-8') as stream: json.dump(config, stream, indent=2)
report = audit(config)
with (root / config['output_report']).open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
assert worker.create_time() == config['scientific_worker']['create_time'] and worker.cmdline() == command
assert sha(sourcepath) == config['source_receipts'][1]['sha256']
(folder / 'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n*.log -diff -text\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ['completed_epoch_logs', 'completed_optimizer_updates_lower_bound', 'latest_valid_interior_interval']}, indent=2))
