import json
from pathlib import Path
from scripts.flow_matching import audit_cfm_gp_conditional_five_seed_budgets_v1 as audit

root,read,sha=audit.ROOT,audit.read,audit.sha
runtimepath='configs/runtime/fm_cfm_completion_replay_queue.v4.json'
runtime=read(root/runtimepath)
relative='projects/flow_matching_research/results/2026-10-14/cfm_conditional_GP400_GP600_all_five_budget_pairs_v1'
folder=root/relative;folder.mkdir(exist_ok=False)
config=dict(primary_runtime=runtimepath,paired_canonical_ids=audit.identities(),
    state_root='experiments/runs/fm_cfm_conditional_GP_five_budget_pairs_v1',poll_seconds=10,
    output_report=relative+'/all_five_complete_original_budget_pair_comparison.json',
    frozen_complete_snapshot=relative+'/frozen_all_ten_complete_model_snapshot.json',
    bootstrap_seed=20261014,bootstrap_replicates=10000,original_scope=runtime['original_scope'],
    citation='https://openreview.net/forum?id=Hqn4Aj7xrQ',
    boundary='All five original paired data/training seeds1103..1107; unchanged conditional GP target, data, full80x500 grids and solver;400/250000 versus600/375000 only. Require all ten guarded native-plus-independent model certificates. Descriptive paired seed/whole-trajectory bootstrap; no failed-seed exclusion, tuning, fitting, inference, author equivalence or TAB.')
files=[runtimepath,'scripts/flow_matching/audit_cfm_gp_conditional_five_seed_budgets_v1.py',
    'tests/test_cfm_gp_conditional_five_seed_budgets_v1.py',
    'scripts/flow_matching/audit_cfm_gp_five_seed_targets_v1.py',
    'scripts/flow_matching/run_cfm_streaming_idle_handoff_v3.py',
    'scripts/flow_matching/run_cfm_completion_replay_queue_v1.py']
for pair in audit.identities():
    for identity in pair:
        job=next(r['job'] for r in runtime['routes'] if r['original_canonical_id']==identity)
        files.extend([job['data_path'],job['model_config']])
config['source_receipts']=[dict(path=p,sha256=sha(root/p)) for p in dict.fromkeys(files)]
audit.verify_config(config)
path='configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_pairs.2026-10-14_v1.json'
with (root/path).open('x') as stream:json.dump(config,stream,indent=2);stream.write('\n')
(folder/'registration_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
snapshot=read(root/runtime['state_root']/'verified_full_model_records.json')
_,missing=audit.select_complete_pairs(snapshot['records'],audit.identities())
(folder/'initial_actual_missing_full_certificates.json').write_text(json.dumps(dict(
    required_complete_models=10,missing_complete_model_ids=missing,existing_certified_models=10-len(missing),
    source_models_restarted=0,additional_training_seed_slots=0,full_400_vs_600_cohort_comparison_complete=False),indent=2)+'\n')
(folder/'README.md').write_text('Registered all five original Appendix-D conditional GP400/GP600 Sine pairs before the full600 cohort finishes. Only the400/600 epoch budgets (250000/375000 updates) differ; data/split, target, seed, architecture and full80x500 grids/solver must match. Wait for all ten actual full guarded native and independent model proofs. Recompute all400000 coordinates and compare paired joint seed groups with whole trajectories inside each group (10000 hierarchical paired bootstrap replicates, seed20261014). Preserve every seed, including any failure as unresolved; never report a successful-subset five-seed mean. Report whether all first400 epoch losses match without claiming intermediate weight/optimizer equivalence. Initially5/10 models had full certificates; all five600-epoch models still require closure. No additional model fitting/inference; not author equivalence, strict TAB or completion of the full45/681/273 research scope.\n')
print(json.dumps(dict(config=path,existing_certified_models=10-len(missing),missing=missing)))
