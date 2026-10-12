"""Admit complete PSM replay only after its exact live diagnostic succeeds."""
import argparse
from datetime import datetime,timezone
import sys
import time
import psutil
from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as full
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write


def diagnostic_ready(config):
    path=full.ROOT/config['diagnostic_terminal']
    if path.exists():
        receipt=full.read(path)
        if receipt['exit_code']!=0 or not receipt['passed'] or receipt['resource_guard_aborted'] or receipt['diagnostic_checks']!=16:
            raise ValueError('Preserve the failed original numerical attempt; never silently retry')
        return True
    expected=config['diagnostic_owner'];actor=psutil.Process(expected['pid'])
    if actor.create_time()!=expected['create_time'] or actor.cmdline()!=expected['command']:
        raise ValueError('The exact numerical diagnostic owner disappeared or changed')
    return False


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--dispatch-config',required=True)
    args=parser.parse_args();config=full.read(args.dispatch_config)
    for pin in config['source_receipts']:
        if full.digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen complete PSM admission source changed')
    base=full.ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    owner=resource_api()['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('The full PSM admission already has a live owner')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
        required_original_jobs=139,all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while not diagnostic_ready(config):
            update(dict(state='waiting_exact_guarded_native_attention_diagnostics',actual_live_diagnostic_owner=config['diagnostic_owner']))
            time.sleep(config['poll_seconds'])
        from scripts.flow_matching.register_crossad_PSM_attention_completion_v3 import register,CONFIG
        from scripts.flow_matching.run_crossad_PSM_attention_completion_v2 import main as completion
        register()
        # Pure registration/source tests do not load an ML backend. Keep the
        # exact tests and every original numerical gate before native dispatch.
        import subprocess
        result=full.ROOT/config['output_directory']
        result.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-X','utf8','-m','pytest','tests/test_fm_crossad_PSM_attention_registration_v3.py',
            '-q','-p','no:cacheprovider','--junitxml='+str(result/'actual_full_registration_validation.xml')]
        with (base/'registration.stdout.log').open('xb') as out,(base/'registration.stderr.log').open('xb') as err:
            checked=subprocess.run(command,cwd=full.ROOT,stdout=out,stderr=err)
        if checked.returncode:raise RuntimeError('Preserve the failed full-source registration validation')
        update(dict(state='actual_diagnostics_and_full_registration_passed_dispatching_same_PSM'))
        sys.argv=['run_crossad_PSM_attention_completion_v2','--config',CONFIG]
        completion()
        native=full.read(full.read(CONFIG)['native_configuration'])
        terminal=full.read(native['state_root']+'/receipt.json')
        update(dict(state='native_terminal_observed',native_terminal=terminal,all_original_experiments_complete=False))
    except Exception as error:
        update(dict(state='failure_preserved',exception_type=type(error).__name__,exception=str(error)))
        raise
    finally:owner.close()


if __name__=='__main__':main()
