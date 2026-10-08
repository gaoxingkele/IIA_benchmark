"""Register uncovered native resource failures with peak-based start headroom."""
import json
import math
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest


def build(root):
    predecessor_path = 'configs/experiments/fm_crossad_preflight_resource_retry.v1.json'
    predecessor = json.loads((root/predecessor_path).read_text(encoding='utf-8'))
    covered = {c['name'] for c in predecessor['cases']}
    cases = []
    peak = 0
    for failure in sorted((root/'experiments/runs/fm_crossad_complete_preflight_v4').glob('*/failure.json')):
        name = failure.parent.name
        proof = f'docs/reports/fm_crossad_complete_preflight_v4_{name}_2026-10-09.json'
        if name in covered or (root/proof).exists():
            continue
        data = json.loads(failure.read_text(encoding='utf-8'))
        resources = data['resource_usage']
        if not resources['resource_guard_aborted']:
            raise ValueError('Uncovered failure requires a source repair, not a resource retry')
        dataset, separator, row = name.partition('_row')
        cases.append({'name':name,'dataset':dataset,'row':int(row) if separator else None,
                      'original_failure_path':failure.relative_to(root).as_posix(),
                      'original_failure_sha256':digest(failure),'canonical_proof_path':proof})
        peak = max(peak,resources['peak_tree_private_bytes'])
    if not cases:
        raise ValueError('No uncovered resource failures to register')
    result = {k:v for k,v in predecessor.items() if k not in ('cases','settings','output_root','boundary')}
    settings = dict(predecessor['settings'])
    reserve = settings['emergency_commit_headroom_bytes']
    headroom = 2*1024**3
    floor = math.ceil((peak+reserve+headroom)/1024**3)*1024**3
    settings['minimum_commit_headroom_bytes'] = max(settings['minimum_commit_headroom_bytes'],floor)
    result.update(settings=settings,cases=cases,output_root='experiments/runs/fm_crossad_preflight_resource_retry_v2',
                  predecessor_config=predecessor_path,predecessor_config_sha256=digest(root/predecessor_path),
                  resource_reservation={'observed_failed_peak_private_bytes':peak,
                      'emergency_reserve_bytes':reserve,'additional_start_margin_bytes':headroom,
                      'minimum_start_commit_headroom_bytes':settings['minimum_commit_headroom_bytes']},
                  boundary='Uncovered resource-aborted native checks only; predecessor cases remain with their live controller. Original queue, batch, model, backward, inputs and emergency guard unchanged. Only start headroom raised above observed peak plus reserve. Preflight is not benchmark performance.')
    own = 'scripts/flow_matching/register_crossad_native_retry_v2.py'
    result['source_receipts'] = list(predecessor['source_receipts']) + [{'path':own,'sha256':digest(root/own)},
        {'path':predecessor_path,'sha256':digest(root/predecessor_path)}]
    return result


def main():
    config=build(ROOT)
    path=ROOT/'configs/experiments/fm_crossad_preflight_resource_retry.v2.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != config:
            raise ValueError('V2 retry already frozen; use a successor')
    else:
        path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':[c['name'] for c in config['cases']],
                      'minimum_start_commit_headroom_bytes':config['settings']['minimum_commit_headroom_bytes'],
                      'config_sha256':digest(path)}))


if __name__ == '__main__':main()
