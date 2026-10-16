from pathlib import Path
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import psutil
from scripts.flow_matching import publish_native_condition_catalog_v55 as p
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write,capped_wait

ROOT=p.ROOT
c=p.read(ROOT/p.CONFIG)
base=ROOT/c['state_root']

def controller():
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades()
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('Publication metadata lane already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        configuration=p.CONFIG,configuration_sha256=p.sha(ROOT/p.CONFIG),
        original_scope=c['original_scope'],all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {},**kw,captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json',state)
    try:
        for item in c['source_receipts']:
            assert p.sha(ROOT/item['path'])==item['sha256']
        assert not (ROOT/c['output_directory']).exists()
        update(state='waiting_shared_readonly_metric_slot_no_model_execution')
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
            command=[sys.executable,'-B','-X','utf8','-u','-m',p.MODULE,'--config',p.CONFIG]
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                    creationflags=subprocess.CREATE_NO_WINDOW)
                handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                write(base/'worker.launch.json',handle)
                update(state='reading_full_proofs_and_publishing_condition_catalog',active_pid=child.pid,exact_child_handle=handle)
                code,usage=capped_wait(child,c,api,lambda v:update(v))
            report=ROOT/c['output_directory']/'publication_validation.json'
            passed=code==0 and not usage['resource_guard_aborted'] and report.exists()
            if passed:
                r=p.read(report)
                passed=(r['metric_rows']==10991 and r['CrossAD_verified_conditions']==11
                    and r['spectral_verified_conditions']==6 and r['new_numeric_cells']==189
                    and r['old8626_rows_and_capture_times_unchanged'] and r['new_training_updates']==0)
            write(base/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
                report_sha256=p.sha(report) if report.exists() else None,model_training_or_inference_executed=False))
            update(state='full_catalog_published_no_new_experiments' if passed else 'publication_failure_preserved',
                active_pid=None,publication_passed=passed)
    finally:lane.close()

if '--controller' in sys.argv:
    controller()
else:
    tests=ROOT/'tmp/native_condition_catalog_v55_tests.xml'
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==12
    assert not any(int(s.get(k,0)) for s in suites for k in ['failures','errors','skipped'])
    assert not base.exists()
    base.mkdir()
    command=[sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--controller']
    with (base/'controller.stdout.log').open('xb') as out,(base/'controller.stderr.log').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    actor=psutil.Process(process.pid)
    write(base/'launch.json',dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
        validation_path=str(tests),validation_sha256=p.sha(tests),configuration_sha256=p.sha(ROOT/p.CONFIG),
        numerical_experiment_workers_modified=False))
    print(actor.pid,actor.create_time())
