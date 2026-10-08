"""Register exact-budget fresh retries for failures during the verified memory incident."""
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha,write_json


def main():
    settings={'output_root':'experiments/runs/fm_resource_recovery_20261009_v2',
        'minimum_available_memory_bytes':24*1024**3,'minimum_commit_headroom_bytes':16*1024**3,
        'emergency_commit_headroom_bytes':8*1024**3,
        'prerequisite_report':'docs/reports/fm_crossad_complete_preflight_2026-10-09.json'}
    left,right=datetime(2026,10,9,4,15),datetime(2026,10,9,4,23)
    cases=[]
    queues=[('industrial','fm_industrial_tsad_cpu_queue.v1'),('strict','fm_reflow_queue.v1'),
        ('strict','fm_strict_baseline_queue.v1'),('pi','fm_pi_transformer_author_queue.v1'),('maelnet','fm_maelnet_author_queue.v1')]
    for kind,name in queues:
        path=ROOT/'configs/experiments'/(name+'.json'); queue=json.loads(path.read_text())
        base=ROOT/queue.get('state_root',queue.get('settings',{}).get('output_root',''))
        failed={r['id']:r for p in base.glob('*status.json') for r in json.loads(p.read_text()).get('records',[]) if r.get('returncode',0)}
        for original in queue['jobs']:
            output=ROOT/original['output_directory']; failure=output/'failure.json'
            logs=list((base/'logs').glob(original['id']+'*')) if kind in ('strict','industrial') else list(output.glob('*.log'))
            evidence=[p for p in logs+[failure] if p.exists() and left<=datetime.fromtimestamp(p.stat().st_mtime)<=right]
            if not evidence or (output/'result.json').exists() or (not failure.exists() and original['id'] not in failed):continue
            job=copy.deepcopy(original); job['id']+='__resource_recovery_v2'
            job['output_directory']=settings['output_root']+'/jobs/'+job['id']
            cases.append({'kind':kind,'original_queue':path.relative_to(ROOT).as_posix(),'original_queue_sha256':sha(path),
                'original_id':original['id'],'original_job':original,'job':job,
                'settings':queue.get('settings',{}),'evaluation':queue.get('evaluation'),
                'python':queue.get('settings',{}).get('python',queue.get('python')),
                'failure_record':json.loads(failure.read_text()) if failure.exists() else failed[original['id']],
                'evidence':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'modified_local':datetime.fromtimestamp(p.stat().st_mtime).isoformat()} for p in evidence]})
    assert len(cases)==11,len(cases)
    common=['scripts/flow_matching/heavy_resources.py','scripts/flow_matching/run_resource_recovery.py',
        'scripts/flow_matching/run_resource_recovery_job.py','scripts/flow_matching/register_resource_recovery.py']
    queue={'schema_version':2,'settings':settings,'jobs':cases,'frozen_files':[{'path':p,'sha256':sha(ROOT/p)} for p in common],
        'superseded_unlaunched_queue':'configs/experiments/fm_resource_recovery_queue.v1.json',
        'boundary':'Fresh exact-budget recovery attempts after four Windows resource-exhaustion events. Original failures/partial artifacts preserved. Recovery is not a new independent seed; same-seed results must be bound to originals before aggregation.'}
    destination=ROOT/'configs/experiments/fm_resource_recovery_queue.v2.json'
    if destination.exists() and json.loads(destination.read_text())!=queue:raise ValueError('Recovery registry already frozen')
    write_json(destination,queue)
    events=[]
    for raw in json.loads((ROOT/'tmp/windows_oom_events_20261009.json').read_text(encoding='utf-8-sig')):
        tree=ET.fromstring(raw['Xml']); ns={'s':'http://schemas.microsoft.com/win/2004/08/events/event','r':'http://www.microsoft.com/Windows/Resource/Exhaustion/Detector/Events'}
        info=tree.find('.//r:SystemInfo',ns); processes=tree.find('.//r:ProcessInfo',ns)
        events.append({'time_local':raw['Time'],'event_id':2004,
            'system_info':{p.tag.split('}')[-1]:int(p.text) for p in info},
            'top_processes':[{p.tag.split('}')[-1]:p.text for p in entry} for entry in processes if entry.find('r:Name',ns).text]})
    write_json(ROOT/'docs/reports/fm_resource_recovery_registration_2026-10-09_v2.json',
        {'registered_jobs':len(cases),'queue_path':destination.relative_to(ROOT).as_posix(),'queue_sha256':sha(destination),
         'events':events,'jobs':[{'kind':c['kind'],'original_id':c['original_id'],'recovery_id':c['job']['id'],'evidence':c['evidence']} for c in cases],
         'boundary':queue['boundary'],'all_recoveries_complete':False})
    print(json.dumps({'registered_jobs':len(cases),'queue_sha256':sha(destination)}))


if __name__=='__main__':main()
