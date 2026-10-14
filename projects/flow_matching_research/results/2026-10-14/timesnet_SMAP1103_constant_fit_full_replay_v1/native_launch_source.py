"""Launch complete native inference in the existing serial readonly resource lane."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

import psutil

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as audit
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

root=audit.ROOT
base=root/'projects/flow_matching_research/results/2026-10-14/timesnet_SMAP1103_constant_fit_full_replay_v1'
path='configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.native.2026-10-14_v1.json'
config=audit.read(root/path);audit.binding(config)
point=audit.read(base/'actual_point_controller_launch.json')
p=psutil.Process(point['pid']);assert p.create_time()==point['create_time']
assert p.cmdline()==point['actual_command']
bootstrap="import sys,runpy;from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader;install_cached_loader(lambda snapshot:None);sys.argv=['full-timesnet-native-replay','--config',"+repr(path)+"];runpy.run_module('scripts.flow_matching.audit_timesnet_constant_fit_complete_v1',run_name='__main__')"
with (base/'native.controller.stdout.log').open('xb') as out,(base/'native.controller.stderr.log').open('xb') as err:
    child=subprocess.Popen([config['python'],'-X','utf8','-u','-c',bootstrap],cwd=root,
        stdin=subprocess.DEVNULL,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
        creationflags=subprocess.CREATE_NO_WINDOW)
p=psutil.Process(child.pid)
exclusive_json(base/'actual_native_controller_launch.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    pid=p.pid,create_time=p.create_time(),actual_command=p.cmdline(),configuration=path,
    configuration_sha256=audit.sha(root/path),same_serial_readonly_lock_as_point=True,
    full_saved_score_review_independently_repeated_by_native_worker=True,original_numeric_workers_changed=False))
exclusive_json(base/'native_launch_source_binding.json',dict(path=str(Path(__file__).relative_to(root)).replace('\\','/'),
    sha256=audit.sha(Path(__file__))))
print(json.dumps(dict(native_controller_pid=p.pid,create_time=p.create_time(),configuration=path)))
