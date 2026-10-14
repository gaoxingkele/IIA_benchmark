import ast
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

root=Path(__file__).resolve().parents[1]
folder=root/'projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP400_GP600_all_five_budget_pairs_v1'
host='C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe'
command=[host,'-X','utf8','-u','-m','scripts.flow_matching.audit_cfm_gp_conditional_five_seed_budgets_v1',
    '--config','configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_pairs.2026-10-14_v1.json']
env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(root/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
probe=subprocess.run(command+['--probe-only'],cwd=root,env=env,capture_output=True,check=True)
report=ast.literal_eval(probe.stdout.decode())
assert report['required_complete_models']==10 and report['original_CFM_scope']==285 and not report['model_training_executed']
(folder/'actual_full_ten_model_registration_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
with (folder/'controller.stdout.log').open('xb') as out,(folder/'controller.stderr.log').open('xb') as err:
    child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
p=psutil.Process(child.pid)
with (folder/'actual_readonly_analysis_controller_launch.json').open('x') as stream:
    json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(),pid=p.pid,create_time=p.create_time(),
        command=p.cmdline(),additional_training_seed_slots=0,additional_model_inference=0),stream,indent=2)
print(json.dumps(dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline())))
