import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import psutil
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import sha, read

ROOT = Path(__file__).resolve().parents[1]
BASE = 'projects/flow_matching_research/results/2026-10-14/local_SB_SWaT_full_checkpoint_scores_v1'
CONFIG = 'configs/reproducibility/fm_local_SB_SWaT_full_checkpoint_audit.2026-10-14_v1.json'
JOB = 'fm_objective_sb__swat__seed1103__stable_numerics_v2'
QUEUE = 'configs/experiments/fm_tsad_entropic_repair_queue.v1.json'
queue = read(ROOT / QUEUE)
job = next(item for item in queue['jobs'] if item['id'] == JOB)
base = ROOT / BASE
base.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, base / 'registration_source.py')
shutil.copyfile(ROOT / 'tmp/local_SB_full_checkpoint_validation_20261014_v1.xml', base / 'validation.xml')
settings_source = 'configs/reproducibility/fm_saits_full_checkpoint_replay.2026-10-11_v1.json'
settings = read(ROOT / settings_source)
receipt = 'experiments/runs/fm_bounded_remaining_cpu_v3/entropic_cpu/receipts/' + JOB + '.json'
range_config = 'configs/experiments/fm_tab_range_execution.sparse.v2.json'
range_evaluation = 'experiments/runs/fm_tsad_tab_range_sparse_v2/jobs/' + JOB + '/evaluation.json'
pins = [QUEUE, settings_source, receipt, range_config, range_evaluation,
    str(Path(range_evaluation).with_name('evaluation.sha256')).replace('\\', '/'),
    job['output_directory'] + '/result.json', job['output_directory'] + '/model.pt', job['output_directory'] + '/scores.npz',
    'scripts/flow_matching/audit_local_SB_full_checkpoint_scores_v1.py',
    'tests/test_fm_local_SB_full_budget_identity_v1.py',
    'scripts/flow_matching/verify_tranad_precision_and_smd_completion_v1.py',
    'scripts/flow_matching/publish_full_range_increment_v25.py',
    BASE + '/registration_source.py', BASE + '/validation.xml']
config = dict(queue=QUEUE, job_id=JOB, python=queue['python'], terminal_receipt=receipt,
    range_config=range_config, range_evaluation=range_evaluation,
    settings=settings['settings'], resource_lock=settings['resource_lock'],
    state_root='experiments/runs/fm_local_SB_SWaT_full_checkpoint_audit_v1', output_directory=BASE + '/verification',
    source_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in pins],
    original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
    boundary='Cold full CPU checkpoint replay of all original train/validation/test timestamp scores and exact original core/event/bootstrap metrics. Original20 epochs,4 Monte Carlo draws and5 scoring times unchanged. Completed range source/scope/macros audited without rerunning integrals. Local objective adapter; historical author TSAD and full TAB training equivalence remain unverified.')
with (ROOT / CONFIG).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(config, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
command = [queue['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_local_SB_full_checkpoint_scores_v1', '--config', CONFIG]
with (base / 'controller.stdout.log').open('xb') as out, (base / 'controller.stderr.log').open('xb') as err:
    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
        creationflags=subprocess.CREATE_NO_WINDOW)
process = psutil.Process(child.pid)
identity = dict(captured_utc=datetime.now(timezone.utc).isoformat(), pid=child.pid,
    create_time=process.create_time(), actual_command=process.cmdline(), configuration=CONFIG,
    configuration_sha256=sha(ROOT / CONFIG), scientific_training_workers_modified_or_stopped=False)
with (base / 'actual_launch_identity.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(identity, stream, indent=2)
    stream.write('\n')
print(json.dumps(identity))
