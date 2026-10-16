"""Stop only identified research processes at the user's explicit request."""
from pathlib import Path
from datetime import datetime,timezone
import json,time
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'projects/flow_matching_research/results/2026-10-17/user_requested_stop_v1'
OUT.mkdir(parents=True,exist_ok=False)
def selected(p):
    if 'python' not in p.name().lower():return False
    args=p.cmdline()
    prefix=str(ROOT/'scripts/flow_matching').replace('\\','/').lower()+'/'
    return any(a.startswith('scripts.flow_matching.') or a.replace('\\','/').lower().startswith(prefix) for a in args)
def record(p):
    return dict(pid=p.pid,create_time=p.create_time(),command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]))
def same(r):
    try:
        p=psutil.Process(r['pid'])
        return p if p.create_time()==r['create_time'] else None
    except psutil.NoSuchProcess:return None
records={};errors=[];suspended=[];terminated=[]
for p in psutil.process_iter():
    try:
        if selected(p):records[p.pid]=record(p)
    except (psutil.Error,OSError):pass
def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
save('identified_before_stop.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),processes=list(records.values())))
# Suspend identified orchestrators before terminating children so they cannot respawn workers.
for r in list(records.values()):
    try:
        p=same(r)
        if p:p.suspend();suspended.append(r['pid'])
    except psutil.NoSuchProcess:pass
    except psutil.Error as e:errors.append(dict(pid=r['pid'],phase='suspend',error=repr(e)))
for r in list(records.values()):
    try:
        p=same(r)
        if not p:continue
        for ch in p.children(recursive=True):
            if ch.pid in records or ch.pid==psutil.Process().pid:continue
            # Child ancestry is captured from an exact, suspended experiment owner.
            item=record(ch);item['identified_experiment_ancestor']=p.pid
            records[ch.pid]=item
            try:ch.suspend();suspended.append(ch.pid)
            except psutil.NoSuchProcess:pass
    except psutil.Error as e:errors.append(dict(pid=r['pid'],phase='descendants',error=repr(e)))
save('all_identified_processes.json',list(records.values()))
pending=[]
for r in reversed(list(records.values())):
    try:
        p=same(r)
        if p:p.terminate();pending.append(p);terminated.append(p.pid)
    except psutil.NoSuchProcess:pass
    except psutil.Error as e:errors.append(dict(pid=r['pid'],phase='terminate',error=repr(e)))
_,alive=psutil.wait_procs(pending,timeout=5)
for p in alive:
    try:
        r=records[p.pid]
        if same(r):p.kill()
    except psutil.Error as e:errors.append(dict(pid=p.pid,phase='kill',error=repr(e)))
psutil.wait_procs(alive,timeout=5)
checks=[]
for index in range(2):
    remaining=[]
    for p in psutil.process_iter():
        try:
            if selected(p):remaining.append(record(p))
        except (psutil.Error,OSError):pass
    checks.append(dict(captured_utc=datetime.now(timezone.utc).isoformat(),remaining=remaining))
    if index==0:time.sleep(2)
report=dict(user_requested_stop=True,goal_status='paused',identified_process_count=len(records),
    terminated_process_count=len(terminated),suspended_process_count=len(suspended),errors=errors,
    verification_observations=checks,all_identified_experiment_processes_stopped=not any(x['remaining'] for x in checks),
    existing_raw_data_results_and_checkpoints_deleted=False,automatic_resume_authorized=False)
save('process_stop_receipt.json',report)
print(json.dumps(dict(packet=str(OUT),**{k:v for k,v in report.items() if k not in ['verification_observations']}),ensure_ascii=False))
