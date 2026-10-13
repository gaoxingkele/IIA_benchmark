from pathlib import Path
import json
from scripts.flow_matching.audit_cfm_gp_five_seed_targets_v1 import ROOT, read, sha, identities

root = ROOT
relative = 'projects/flow_matching_research/results/2026-10-14/cfm_GP400_all_five_target_pairs_v1'
folder = root / relative
folder.mkdir(exist_ok=False)
runtimepath = root / 'configs/runtime/fm_cfm_completion_replay_queue.v4.json'
runtime = read(runtimepath)
routes = {r['original_canonical_id']: r['job'] for r in runtime['routes']}
pins = [runtimepath, root / 'scripts/flow_matching/audit_cfm_gp_five_seed_targets_v1.py',
        root / 'tests/test_cfm_gp_five_seed_targets_v1.py',
        root / 'scripts/flow_matching/run_cfm_streaming_idle_handoff_v3.py',
        root / 'scripts/flow_matching/run_cfm_completion_replay_queue_v1.py']
for pair in identities():
    for identity in pair:
        job = routes[identity]
        pins += [root / job['data_path'], root / job['model_config']]
pins = list(dict.fromkeys(pins))
config = dict(primary_runtime=runtimepath.relative_to(root).as_posix(), paired_canonical_ids=identities(),
    state_root='experiments/runs/fm_cfm_gp400_five_target_pairs_v1', poll_seconds=10,
    output_report=relative + '/all_five_full_original_target_pair_comparison.json',
    frozen_complete_snapshot=relative + '/frozen_all_ten_complete_model_snapshot.json',
    bootstrap_seed=20261014, bootstrap_replicates=10000,
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    original_scope={'papers': 45, 'method_baseline_ablation_records': 681, 'original_data_task_records': 273},
    boundary='Require every original paired seed1103..1107, ten400/250000 full native plus independent models, every80x500 grid, same pair data and truth. Joint data/training seed outer groups and paired whole trajectories inner groups. Never fit/infer/tune/drop failed or missing seeds; not author-equivalence or TAB.',
    source_receipts=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in pins])
path = root / 'configs/reproducibility/fm_cfm_GP400_all_five_original_target_pairs.2026-10-14_v1.json'
with path.open('x', encoding='utf-8') as stream: json.dump(config, stream, indent=2)
(folder / 'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder / '.gitattributes').write_text('*.py -text\n', encoding='utf-8')
print(json.dumps(dict(config=path.relative_to(root).as_posix(), required_full_models=10, full_CFM_scope=285)))
