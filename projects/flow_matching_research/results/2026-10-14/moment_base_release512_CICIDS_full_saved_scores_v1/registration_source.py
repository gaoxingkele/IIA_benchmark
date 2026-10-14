from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from scripts.flow_matching.prepare_tsad_execution import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = 'projects/flow_matching_research/results/2026-10-14/moment_base_release512_CICIDS_full_saved_scores_v1'
CONFIG = 'configs/reproducibility/fm_moment_base_release512_CICIDS_full_saved_score_audit.2026-10-14_v1.json'
JOB = 'fm_moment_pretrained_base__release512__cicids__seed42'
queue_path = 'configs/experiments/fm_moment_pretrained_queue.v1.json'
queue = json.loads((ROOT / queue_path).read_text(encoding='utf-8'))
job = next(item for item in queue['jobs'] if item['id'] == JOB)
output = ROOT / BASE
output.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, output / 'registration_source.py')
shutil.copyfile(ROOT / 'tmp/moment_CICIDS_saved_score_audit_validation_20261014_v1.xml', output / 'validation.xml')
settings_source = 'configs/reproducibility/fm_saits_full_checkpoint_replay.2026-10-11_v1.json'
settings = json.loads((ROOT / settings_source).read_text(encoding='utf-8'))
receipt = 'experiments/runs/fm_bounded_remaining_cpu_v3/moment_cpu/receipts/' + JOB + '.json'
range_config = 'configs/experiments/fm_moment_pretrained_tab_range.sparse.v2.json'
range_evaluation = 'experiments/runs/fm_moment_pretrained_tab_range_sparse_v2/jobs/' + JOB + '/evaluation.json'
range_resource = 'experiments/runs/fm_moment_pretrained_tab_range_sparse_v2/logs/' + JOB + '/resource_usage.json'
pins = [queue_path, settings_source, receipt, range_config, range_evaluation,
    str(Path(range_evaluation).with_name('evaluation.sha256')).replace('\\', '/'), range_resource,
    job['output_directory'] + '/result.json', job['output_directory'] + '/model.pt', job['output_directory'] + '/scores.npz',
    'scripts/flow_matching/audit_moment_completed_saved_scores_v1.py', 'tests/test_fm_moment_completed_saved_score_identity_v1.py',
    'scripts/flow_matching/verify_tranad_precision_and_smd_completion_v1.py',
    'scripts/flow_matching/publish_full_range_increment_v25.py',
    BASE + '/registration_source.py', BASE + '/validation.xml']
config = dict(queue=queue_path, job_id=JOB, python=queue['python'], terminal_receipt=receipt,
    range_config=range_config, range_evaluation=range_evaluation, range_resource_receipt=range_resource,
    range_controller_state='experiments/runs/fm_moment_pretrained_tab_range_sparse_v2/status.json',
    settings=settings['settings'], resource_lock=settings['resource_lock'],
    state_root='experiments/runs/fm_moment_CICIDS_full_saved_score_audit_v1', output_directory=BASE + '/verification',
    source_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in pins],
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    boundary='Independent full saved train/validation/test arrays and exact original producer-environment primary/core metrics, event metrics and grouped uncertainty; full original range scope/source/macros rechecked. No new inference/fitting, no changed context/threshold-grid/buffer-budget, no historical-paper or fullTAB-training-harness equivalence claim.')
with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(config, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
command = ['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe', '-X', 'utf8', '-u', '-m',
    'scripts.flow_matching.audit_moment_completed_saved_scores_v1', '--config', CONFIG]
with (output / 'controller.stdout.log').open('xb') as out, (output / 'controller.stderr.log').open('xb') as err:
    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
        creationflags=subprocess.CREATE_NO_WINDOW)
process = psutil.Process(child.pid)
ct = process.create_time()
time.sleep(.5)
assert process.is_running() and process.create_time() == ct and process.cmdline() == command
with (output / 'actual_launch_identity.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(), pid=process.pid, create_time=ct,
        actual_command=process.cmdline(), actual_live=True, configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG),
        scientific_training_workers_modified_or_stopped=False), stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(pid=process.pid, create_time=ct, actual_live=True, configuration=CONFIG)))
