"""Register only failed native-batch checks, preserving every original failure."""
import json
from pathlib import Path
from scripts.flow_matching.run_light_controller import ROOT, digest


def main():
    parent_path = 'configs/experiments/fm_crossad_complete_queue.v4.json'
    parent = json.loads((ROOT/parent_path).read_text(encoding='utf-8'))
    cases = []
    base = ROOT/'experiments/runs/fm_crossad_complete_preflight_v4'
    for failure in sorted(base.glob('*/failure.json')):
        name = failure.parent.name
        report_path = 'docs/reports/fm_crossad_complete_preflight_v4_'+name+'_2026-10-09.json'
        if (ROOT/report_path).exists(): continue
        data = json.loads(failure.read_text(encoding='utf-8'))
        if not data['resource_usage']['resource_guard_aborted']:
            raise ValueError('A source error needs a code repair, not an unchanged resource retry')
        dataset, separator, row = name.partition('_row')
        cases.append({'name':name,'dataset':dataset,'row':int(row) if separator else None,
                      'original_failure_path':failure.relative_to(ROOT).as_posix(),'original_failure_sha256':digest(failure),
                      'canonical_proof_path':report_path})
    sources = {r['path']:r['sha256'] for j in parent['jobs'] for r in j['frozen_files']}
    runtime = json.loads((ROOT/'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    sources.update({r['path']:r['sha256'] for r in runtime['source_receipts']})
    for p in ('scripts/flow_matching/register_crossad_preflight_resource_retry.py',
              'scripts/flow_matching/run_crossad_preflight_resource_retry.py',
              'scripts/flow_matching/preflight_crossad_author.py'):
        sources[p]=digest(ROOT/p)
    config = {'parent_queue':parent_path,'parent_queue_sha256':digest(ROOT/parent_path),
              'runtime_config':'configs/runtime/fm_light_controllers.v1.json',
              'runtime_config_sha256':digest(ROOT/'configs/runtime/fm_light_controllers.v1.json'),
              'output_root':'experiments/runs/fm_crossad_preflight_resource_retry_v1',
              'settings':parent['settings'],'cases':cases,
              'source_receipts':[{'path':p,'sha256':h} for p,h in sorted(sources.items())],
              'boundary':'Only repeat previously resource-aborted full native batch checks. Same original queue, batch, architecture, backward, raw inputs and guards. Original failures and logs retained. Preflight is not benchmark performance.'}
    path = ROOT/'configs/experiments/fm_crossad_preflight_resource_retry.v1.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==config, 'Retry config already frozen'
    else:
        path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':[c['name'] for c in cases],'source_files':len(sources),'config_sha256':digest(path)}))


if __name__=='__main__':main()
