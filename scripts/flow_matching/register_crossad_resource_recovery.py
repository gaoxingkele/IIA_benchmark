"""Exact original CrossAD release pipelines with preserved failed attempts."""
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest


def build(root):
    parent_path='configs/experiments/fm_crossad_complete_queue.v7.json'
    parent=json.loads((root/parent_path).read_text(encoding='utf-8'))
    cases=[]
    for original in parent['jobs']:
        if original['mode']!='release_checkpoint':continue
        failure=root/original['output_directory']/'failure.json'
        if not failure.exists() or (failure.parent/'result.json').exists():continue
        data=json.loads(failure.read_text(encoding='utf-8'))
        if not data['resources']['resource_guard_aborted']:
            raise ValueError('CrossAD source failure needs a repair before recovery')
        identity=original['id']+'__resource_recovery_v1'
        job=dict(original,id=identity,output_directory='experiments/runs/fm_crossad_resource_recovery_v1/jobs/'+identity)
        cases.append({'original_queue':parent_path,'original_queue_sha256':digest(root/parent_path),
                      'original_id':original['id'],'original_job':original,'job':job,'kind':'crossad',
                      'settings':parent['settings'],'evaluation':parent.get('evaluation'),
                      'original_failure_path':failure.relative_to(root).as_posix(),
                      'original_failure_sha256':digest(failure)})
    if not cases:raise ValueError('No failed release pipelines to recover')
    native_path='configs/experiments/fm_crossad_resource_recovery_native.v1.json'
    native={'settings':parent['settings'],'jobs':[c['job'] for c in cases]}
    resources=dict(parent['settings'])
    resources['minimum_commit_headroom_bytes']=max(resources['minimum_commit_headroom_bytes'],24*1024**3)
    config={'jobs':cases,'settings':resources,'native_queue':native_path,
            'runtime_config':'configs/runtime/fm_light_controllers.v1.json',
            'output_root':'experiments/runs/fm_crossad_resource_recovery_v1',
            'boundary':'Retry only preserved resource-aborted release pipelines. Original author source, batch, weights, seed, all metrics and native-tail control unchanged. Fresh identity/output; same original slot, no added repeat. Start reservation raised; emergency guard unchanged.'}
    return config,native


def main():
    config,native=build(ROOT)
    native_path=ROOT/config['native_queue']
    def freeze(path,value):
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Already frozen: '+str(path))
        else:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    freeze(native_path,native)
    sources={r['path']:r['sha256'] for case in config['jobs'] for r in case['job']['frozen_files']}
    for p in (config['native_queue'],config['runtime_config'],
              'scripts/flow_matching/register_crossad_resource_recovery.py',
              'scripts/flow_matching/run_crossad_resource_recovery.py'):
        sources[p]=digest(ROOT/p)
    for case in config['jobs']:
        sources[case['original_queue']]=case['original_queue_sha256']
        sources[case['original_failure_path']]=case['original_failure_sha256']
    config['source_receipts']=[{'path':p,'sha256':h} for p,h in sorted(sources.items())]
    path=ROOT/'configs/experiments/fm_crossad_resource_recovery.v1.json'
    freeze(path,config)
    print(json.dumps({'jobs':len(config['jobs']),'config_sha256':digest(path),'source_files':len(sources)}))


if __name__=='__main__':main()
