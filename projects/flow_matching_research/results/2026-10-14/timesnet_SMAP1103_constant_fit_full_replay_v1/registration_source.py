"""Freeze the complete local TimesNet audit and launch its guarded point stage."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as audit
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

root=audit.ROOT
base='projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1'
folder=root/base;folder.mkdir(parents=True,exist_ok=False)
queue_path='configs/experiments/fm_tsad_constant_fit_strict_baselines_queue.v2.json'
queue=audit.read(root/queue_path);job=next(j for j in queue['jobs'] if j['id']==audit.CANONICAL)
result=audit.read(root/job['output_directory']/'result.json')
guard_template='configs/reproducibility/fm_saits_full_checkpoint_replay.2026-10-11_v1.json'
template=audit.read(root/guard_template)
terminal='experiments/runs/fm_tsad_constant_fit_cpu_controls_v2/strict_baselines/receipts/'+audit.CANONICAL+'.json'
pins={queue_path,guard_template,terminal,job['output_directory']+'/result.json',
    job['output_directory']+'/model.pt',job['output_directory']+'/scores.npz',job['model_config'],
    job['dataset_manifest_path'],'scripts/flow_matching/audit_timesnet_constant_fit_complete_v1.py',
    'tests/test_fm_timesnet_constant_fit_complete_v1.py','scripts/flow_matching/verify_tranad_precision_and_smd_completion_v1.py',
    'configs/experiments/fm_constant_fit_tab_range.sparse.v2.json',
    'projects/flow_matching_research/results/2026-10-11/constant_fit_preprocessing_v2/full_recomparison_registration.json'}
pins.update(pin['path'] for pin in job['frozen_files'])
manifest=audit.read(root/job['dataset_manifest_path']);dataset=next(d for d in manifest['datasets'] if d['id']==job['dataset'])
pins.add(dataset['input_npz']);pins.update(pin['path'] for pin in dataset['source_files'])
shutil.copyfile(__file__,folder/'registration_source.py')
shutil.copyfile(root/'tmp/TimesNet_SMAP1103_constant_fit_complete_validation_20261014_v1.xml',folder/'validation.xml')
pins.update((base+'/registration_source.py',base+'/validation.xml'))
configs={}
for mode in ('point','native'):
    config=dict(queue=queue_path,job_id=audit.CANONICAL,mode=mode,
        python=queue['python'],cpu_threads=queue['evaluation']['cpu_threads'],
        guard_template=guard_template,settings=template['settings'],resource_lock=template['resource_lock'],
        terminal_receipt=terminal,original_scope=audit.SCOPE,
        state_root='experiments/runs/fm_timesnet_SMAP1103_constant_fit_full_replay_v1/'+mode,
        output_directory=base+'/point' if mode=='point' else
            'F:/aicoding/IIA_Data/experiments/timesnet_SMAP1103_constant_fit_full_replay_v1/native',
        source_receipts=[dict(path=p,sha256=audit.sha(root/p)) for p in sorted(pins)],
        boundary='Original full35-slot baseline queue and291corrected data controls retained. Complete51SMAPentities/all train-validation-test timestamps; same28,132,633parameter weights,float32,CPU2,batch128,configured3epochs. Independent point/core/event/bootstrap score verification and separate complete native checkpoint inference, bitwise scores required. Original source receipt confirms retained registered budget but no per-epoch/update counter exists. Full TAB range/five-seed/author-harness equivalence remain separate incomplete requirements.')
    path='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.'+mode+'.2026-10-14_v1.json'
    exclusive_json(root/path,config);audit.binding(config);configs[mode]=path
exclusive_json(folder/'registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    canonical_id=audit.CANONICAL,configurations=configs,original_scope=audit.SCOPE,
    original_full_baseline_queue_jobs=35,original_corrected_data_control_jobs=291,
    original_completed_guarded_training=True,source_guard_ended_without_abort=True,
    saved_score_audit_passed=False,full_native_checkpoint_reinference_passed=False,
    full_TAB_range_metrics_completed=False,validation_tests_passed=37,
    original_required_cohort_seeds=[1103,1104,1105,1106,1107],additional_training_seed_slots=0,
    all_original_experiments_complete=False))
bootstrap="import sys,runpy;from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader;install_cached_loader(lambda snapshot:None);sys.argv=['full-timesnet-score-audit','--config',"+repr(configs['point'])+"];runpy.run_module('scripts.flow_matching.audit_timesnet_constant_fit_complete_v1',run_name='__main__')"
command=[queue['python'],'-X','utf8','-u','-c',bootstrap]
with (folder/'point.controller.stdout.log').open('xb') as out,(folder/'point.controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
        creationflags=subprocess.CREATE_NO_WINDOW)
p=psutil.Process(child.pid)
exclusive_json(folder/'actual_point_controller_launch.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid,create_time=p.create_time(),actual_command=p.cmdline(),configuration=configs['point'],
    original_numeric_workers_changed=False))
print(json.dumps(dict(point_controller_pid=p.pid,create_time=p.create_time(),configurations=configs)))
