"""Hand off idle controllers with verified Windows console helpers only.

Keep the frozen V1 handoff and runtime sources unchanged. A conhost child is
not a model worker; any other descendant or active job still blocks handoff.
"""
import json
import os
from pathlib import Path
import sys

import psutil
from scripts.flow_matching import transition_light_controllers as original


def console_helper(process):
    expected = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/conhost.exe'
    try:
        return (process.name().lower() == 'conhost.exe'
                and Path(process.exe()).resolve() == expected.resolve()
                and not process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def idle_observation(binding):
    path = original.ROOT / binding['status_path']
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding='utf-8'))
    try:
        process = psutil.Process(state['pid'])
        command = process.cmdline()
        if not any(binding['old_worker_token'] in a for a in command):
            return None
        # Match the full queue binding, not just a process name or reusable PID.
        argument = binding['queue_argument']
        if argument not in command:
            return None
        actual = Path(command[command.index(argument)+1])
        actual = actual if actual.is_absolute() else original.ROOT / actual
        if actual.resolve() != (original.ROOT / binding['queue_path']).resolve():
            return None
        if any(state.get(k) for k in ('active_pid', 'active_job', 'active_case')):
            return None
        if state.get('state') not in ('waiting_for_memory', 'waiting_for_heavy_lock',
                                     'waiting_for_remaining_native_preflight',
                                     'waiting_training_results'):
            return None
        if any(not console_helper(p) for p in process.children(recursive=True)):
            return None
        return process, state, command, process.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
        return None


def main():
    original.idle_observation = idle_observation
    original.main()  # Preserve suspend/recheck/archive/terminate/launch sequence.
    path = original.ROOT / sys.argv[sys.argv.index('--report')+1]
    report = json.loads(path.read_text(encoding='utf-8'))
    report.update(handoff_source=Path(__file__).relative_to(original.ROOT).as_posix(),
                  handoff_source_sha256=original.digest(Path(__file__)),
                  console_helper_policy='Only exact SystemRoot/System32/conhost.exe without descendants may accompany an otherwise idle, queue-bound controller. Any model or unknown child blocks handoff.')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
