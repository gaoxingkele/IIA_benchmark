"""Freeze a metric-only mirror of all strict/industrial incident retries."""
import copy
import json
from pathlib import Path
from scripts.flow_matching.run_light_controller import ROOT,digest


def main():
    recovery_path='configs/experiments/fm_resource_recovery_queue.v3.json'
    recovery=json.loads((ROOT/recovery_path).read_text(encoding='utf-8'))
    jobs=[c['job'] for c in recovery['jobs'] if c['kind'] in ('strict','industrial')]
    queue_path='configs/experiments/fm_recovery_tab_mirror.v1.json'
    mirror={'jobs':jobs,'recovery_queue':recovery_path,'recovery_queue_sha256':digest(ROOT/recovery_path),
            'boundary':'Metric-only mirror: exact recovery job dictionaries, no additional training or independent seeds.'}
    path=ROOT/queue_path
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==mirror
    else:path.write_text(json.dumps(mirror,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    parent='configs/experiments/fm_tab_range_execution.sparse.v2.json'
    config=copy.deepcopy(json.loads((ROOT/parent).read_text(encoding='utf-8')))
    config.update(id='fm_recovery_tab_ranges_v1',output_root='experiments/runs/fm_recovery_tab_ranges_v1',queue_paths=[queue_path],
                  parent_config=parent,parent_config_sha256=digest(ROOT/parent),worker_script='run_recovery_tab_ranges.py',
                  light_runtime_config='configs/runtime/fm_light_controllers.v1.json')
    sources={r['path']:r['sha256'] for r in config['source_receipts']}
    for p in (queue_path,recovery_path,parent,'configs/runtime/fm_light_controllers.v1.json',
              'scripts/flow_matching/run_recovery_tab_ranges.py','scripts/flow_matching/register_recovery_tab_ranges.py'):
        sources[p]=digest(ROOT/p)
    runtime=json.loads((ROOT/config['light_runtime_config']).read_text(encoding='utf-8'))
    sources.update({r['path']:r['sha256'] for r in runtime['source_receipts']})
    config['source_receipts']=[{'path':p,'sha256':h} for p,h in sorted(sources.items())]
    config['boundary']+=' Recovery attempts substitute original failed seed slots; range evaluation itself never adds repeats.'
    target=ROOT/'configs/experiments/fm_recovery_tab_ranges.v1.json'
    if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==config
    else:target.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'metric_only_attempts':len(jobs),'config_sha256':digest(target)}))


if __name__=='__main__':main()
