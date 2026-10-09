"""Hand off only a specifically observed, idle, childless TSAD GPU scheduler."""
import json
from pathlib import Path
import psutil
from scripts.flow_matching import transition_light_controllers as original


def idle_observation(binding):
    path = original.ROOT / binding['status_path']
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
        if (state.get('status') != 'waiting_existing_imputation'
                or state.get('pid') != binding['expected_old_pid']
                or any(state.get(key) for key in ['active_pid', 'active_job', 'active_case'])):
            return None
        process = psutil.Process(state['pid'])
        if process.create_time() != binding['expected_old_create_time'] or process.children(recursive=True):
            return None
        command = process.cmdline()
        source = (original.ROOT / binding['source_path']).resolve()
        if not any(Path(arg).resolve() == source for arg in command if not arg.startswith('-')):
            return None
        argument = binding['queue_argument']
        if argument not in command:
            return None
        actual = Path(command[command.index(argument) + 1])
        actual = actual if actual.is_absolute() else original.ROOT / actual
        if actual.resolve() != (original.ROOT / binding['queue_path']).resolve():
            return None
        return process, state, command, process.create_time()
    except (OSError, KeyError, IndexError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def main():
    original.idle_observation = idle_observation
    original.main()


if __name__ == '__main__':
    main()
