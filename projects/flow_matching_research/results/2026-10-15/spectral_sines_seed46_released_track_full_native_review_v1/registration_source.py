import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
from scripts.flow_matching.prepare_spectral_original_data import read, sha
from scripts.flow_matching.audit_spectral_completed_cohorts_v1 import registration

BASE = 'projects/flow_matching_research/results/2026-10-15/spectral_sines_seed46_released_track_full_native_review_v1'
CONFIG = 'configs/reproducibility/fm_spectral_sines_seed46_released_track_full_native_review.2026-10-15_v1.json'


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
args = parser.parse_args()
if not args.launch:
    (ROOT / BASE).mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, ROOT / BASE / 'registration_source.py')
    queue_path = 'configs/experiments/fm_spectral_original_resource_admission_queue.v3.json'
    queue = read(ROOT / queue_path)
    cohorts = {'sines': [46]}
    audits = []
    pins = []
    source_paths = ['scripts/flow_matching/audit_spectral_completed_generation_v1.py',
        'scripts/flow_matching/audit_spectral_completed_generation_v2.py',
        'scripts/flow_matching/audit_spectral_completed_cohorts_v1.py',
        'tests/test_fm_spectral_completed_cohorts_v1.py', 'tests/test_fm_spectral_completed_generation_v1.py',
        'tests/test_fm_spectral_sines_data_track_selection.py',
        'scripts/flow_matching/run_spectral_original_job.py', 'scripts/flow_matching/prepare_spectral_original_data.py',
        'src/iia_benchmark/models/fm_spectral_author.py', 'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py', 'scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/heavy_resources.py', BASE + '/registration_source.py']
    for job in queue['jobs']:
        if job['model'] != 'diffusion_ts' or job['data_track'] != 'released_author_data' or job['dataset'] not in cohorts or job['seed'] not in cohorts[job['dataset']]:
            continue
        result_path = job['output_directory'] + '/result.json'
        result = read(ROOT / result_path)
        receipt_path = queue['state_root'] + '/receipts/' + job['id'] + '.json'
        terminal = read(ROOT / receipt_path)
        assert terminal['exitcode'] == 0 and terminal['full_result_validated'] and not terminal['resource_guard_aborted']
        paths = [queue_path, queue['protocol'], queue['environment_lock'], job['model_config'], job['author_yaml'],
            result_path, receipt_path, job['frozen_samples'], job['reference_path'], *source_paths,
            *[artifact['path'] for artifact in result['artifacts']]]
        audit_config = f"configs/reproducibility/fm_spectral_sines46_review_20261015_v1/{job['id']}.json"
        save(audit_config, dict(queue=queue_path, job_id=job['id'], terminal_receipt=receipt_path,
            output_directory=BASE + '/models/' + job['id'],
            source_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in dict.fromkeys(paths)],
            boundary='Full unchanged original native training/sampling/four evaluators; read-only CPU artifact checks, no new training or reinference.'))
        audits.append(dict(id=job['id'], configuration=audit_config))
        pins.extend(paths + [audit_config])
    assert len(audits) == 1
    settings = {key: queue['settings'][key] for key in ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes']}
    previous = read(ROOT / 'configs/reproducibility/fm_spectral_stocks_completed_native_audits.2026-10-11_v2.json')
    settings.update({key: previous['settings'][key] for key in ['maximum_worker_rss_bytes', 'maximum_worker_private_bytes']})
    pins += ['configs/reproducibility/fm_spectral_stocks_completed_native_audits.2026-10-11_v2.json']
    config = dict(original_queue=queue_path, original_required_seeds=[42, 43, 44, 45, 46],
        completed_cohort_seeds=cohorts, audits=audits, python=str(ROOT / queue['python']), settings=settings,
        resource_lock=previous['resource_lock'], state_root='experiments/runs/fm_spectral_sines46_native_review_v1',
        source_receipts=[dict(path=path, sha256=sha(ROOT / path)) for path in dict.fromkeys(pins)],
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273),
        boundary='The actual terminal-success Sines seed46 model require full frozen-v1 native tensor/array/statistic checks. Original full70 generation queue and all5 training-seed obligations remain. This original seed46 audit and all other registered scopes remain required. Shared readonlyCPU lock and unchanged12GiBphysical/20GiBcommit/12GiBemergency admission; existing3GiBRSS/6GiBprivate limits. No GPU use, new fitting, sampling, learned evaluator retraining or author/TAB claim.')
    registration(config)
    save(CONFIG, config)
    save(BASE + '/registration.json', dict(configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG),
        audits=audits, completed_cohort_seeds=cohorts, independently_reviewed_models=0,
        full_native_reviews_still_required=True, all_original_experiments_complete=False))
    print(json.dumps(dict(registered_full_native_reviews=len(audits), configuration=CONFIG)))
else:
    config = read(ROOT / CONFIG)
    registration(config)
    validation = ROOT / BASE / 'validation.xml'
    import xml.etree.ElementTree as ET
    suites = list(ET.parse(validation).getroot().iter('testsuite'))
    assert sum(int(suite.get('tests', 0)) for suite in suites) >= 21
    assert not any(int(suite.get('failures', 0)) or int(suite.get('errors', 0)) for suite in suites)
    command = [str(Path('C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe')), '-X', 'utf8', '-u', '-m',
        'scripts.flow_matching.audit_spectral_completed_cohorts_v1', '--config', CONFIG]
    with (ROOT / BASE / 'controller.stdout.log').open('xb') as out, (ROOT / BASE / 'controller.stderr.log').open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
            creationflags=subprocess.CREATE_NO_WINDOW)
    handle = psutil.Process(child.pid)
    ct = handle.create_time()
    time.sleep(.5)
    assert handle.is_running() and handle.create_time() == ct and handle.cmdline() == command
    save(BASE + '/actual_launch_identity.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        pid=child.pid, create_time=ct, actual_command=handle.cmdline(), actual_live=True,
        configuration_sha256=sha(ROOT / CONFIG), experiment_workers_modified_or_stopped=False,
        independently_reviewed_models=0, all_original_experiments_complete=False))
    print(json.dumps(dict(pid=child.pid, create_time=ct, actual_live=True, full_native_reviews=1)))
