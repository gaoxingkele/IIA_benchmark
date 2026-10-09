"""Allow only verified childless Windows console helpers at the idle boundary."""
import sys
from pathlib import Path
import psutil

from scripts.flow_matching import transition_cfm_ts_low_memory_v2 as original
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


def idle_observation(runtime):
    old=runtime['predecessor'];state=original.read(original.ROOT/old['status_path'])
    try:
        process=psutil.Process(old['pid'])
        if process.create_time()!=old['create_time'] or process.cmdline()!=old['command']:
            return None
        if state['pid']!=process.pid or state.get('active_pid') or state.get('active_job'):
            return None
        if state.get('state') not in ('waiting_for_memory','between_jobs'):
            return None
        if any(not console_helper(child) for child in process.children(recursive=True)):
            return None
        return process,state
    except (psutil.NoSuchProcess,psutil.AccessDenied):return None


if __name__=='__main__':
    original.idle_observation=idle_observation
    original.main()
    runtime=original.read(original.ROOT/sys.argv[sys.argv.index('--runtime-config')+1])
    path=original.ROOT/runtime['handoff_root']/'transition.json'
    value=original.read(path)
    value.update(handoff_source=Path(__file__).relative_to(original.ROOT).as_posix(),
        handoff_source_sha256=original.sha(Path(__file__)),
        console_helper_policy='Only exact SystemRoot/System32/conhost.exe with no descendants is allowed; any model or unknown child blocks handoff.')
    original.write(path,value)
