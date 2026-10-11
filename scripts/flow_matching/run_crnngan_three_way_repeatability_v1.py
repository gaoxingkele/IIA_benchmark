"""Run the full native three-way reader after the owned original diagnostic exits."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import psutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def owned_process(binding):
    try:
        process = psutil.Process(binding['pid'])
        if abs(process.create_time() - binding['create_time']) > 0.001:
            raise ValueError('Diagnostic PID was reused; do not observe another process')
        if binding['command_token'] not in process.cmdline():
            raise ValueError('Owned diagnostic command changed')
        return dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(),
                    cpu_seconds=sum(process.cpu_times()[:2]))
    except psutil.NoSuchProcess:
        return None


def dependency_ready(config, observation):
    directory = ROOT / config['dependency_output_directory']
    guard_path = directory / 'guard_receipt.json'
    if not guard_path.exists():
        if observation is None:
            raise RuntimeError('Original diagnostic controller is missing without a terminal guard receipt')
        return False
    guard = read(guard_path)
    if guard['exit_code'] != 0 or not guard['passed'] or guard['resource_guard_aborted']:
        raise RuntimeError('Original four-epoch diagnostic terminated unsuccessfully; retain all attempts')
    from scripts.flow_matching.audit_crnngan_three_way_repeatability_v1 import require_complete_original_diagnostic
    require_complete_original_diagnostic(read(directory / 'original_four_epoch_repeatability.json'), guard)
    # Wait until its resource slot is released; the actual model is never interrupted.
    return observation is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    for receipt in config['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen native three-way execution input changed')
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api()
    output = ROOT / config['controller_output_directory']
    output.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](output / 'controller.lock')
    if owner is None:
        raise RuntimeError('Three-way native reader already has a live controller')
    if (output / 'guard_receipt.json').exists() or (ROOT / config['output_path']).exists():
        owner.close()
        raise FileExistsError('Preserve all completed native three-way results')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
                 configuration_sha256=sha(ROOT / args.config), benchmark_metric_rows_added=0)

    def update(values):
        state.update(values)
        state['captured_utc'] = datetime.now(timezone.utc).isoformat()
        write(output / 'controller_status.json', state)

    try:
        while True:
            observed = owned_process(config['dependency_controller'])
            if dependency_ready(config, observed):
                break
            update(dict(state='waiting_owned_full_original_diagnostic', dependency_observation=observed))
            time.sleep(10)
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.audit_crnngan_three_way_repeatability_v1', '--config', args.config]
            environment = dict(os.environ, **config['native_execution_environment'])
            with (output / 'native_reader.stdout.log').open('xb') as out, (output / 'native_reader.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='reading_all_three_complete_native_attempts', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api,
                    lambda values: update(dict(values, state='reading_all_three_complete_native_attempts')))
            passed = code == 0 and (ROOT / config['output_path']).exists()
            write(output / 'guard_receipt.json', dict(exit_code=code, passed=passed, **usage))
            update(dict(state='completed_full_native_diagnostic' if passed else 'failed_preserved', active_pid=None))
    except BaseException as error:
        update(dict(state='failed_preserved', error=repr(error)))
        raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
