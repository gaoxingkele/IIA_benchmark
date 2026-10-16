"""Publish actual complete source outcomes without promoting partial paper rows."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import psutil
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as source
from scripts.flow_matching import summarize_otcfm_EB100_full_paper_rows_v1 as rows
from scripts.flow_matching.otcfm_EB100_result_integrity_v1 import inspect
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as grasp
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding


def dump(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def copy(path, target):
    with target.open('xb') as stream:
        stream.write(path.read_bytes())
    assert sha(path) == sha(target)


def xml_proof(path, tests):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites) == tests
    assert all(int(s.attrib.get(k, '0')) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))


def handle(pid, created):
    try:
        p = psutil.Process(pid)
        same = p.is_running() and p.create_time() == created
        return dict(pid=pid, expected_create_time=created, exact_process_live=same,
                    actual_create_time=p.create_time(), command=p.cmdline() if same else None,
                    cpu_seconds=sum(p.cpu_times()[:2]) if same else None)
    except psutil.NoSuchProcess:
        return dict(pid=pid, expected_create_time=created, exact_process_live=False)


root_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
parent_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent/'flow-matching-research', text=True).strip()
assert root_head == '6137c52aed6322008e845d940c9eaed4f01e45c1'
assert parent_head == '2db3b540a081716c0cc5b585f5809b808e3d8e72'
q = read(ROOT/source.QUEUE)
source.verify(q)
c = read(ROOT/rows.CONFIG)
rows.verify(c)
jobs = [j for j in q['jobs'] if j['protocol'] == 'released_yaml' and j['method'] == 'cfm'
        and j['leaveout_timepoint'] == 1 and j['seed'] in range(42, 46)]
assert len(jobs) == 4
validated = [(j, *inspect(q, j)) for j in jobs]
preflight_path = ROOT/q['state_root']/'native18_preflight.json'
preflight_guard_path = ROOT/q['state_root']/'preflight/guard_receipt.json'
preflight, pg = read(preflight_path), read(preflight_guard_path)
assert preflight['status'] == 'passed' and preflight['model_training_updates'] == 0
assert pg['passed'] and pg['exit_code'] == 0 and not pg['resource_guard_aborted']
assert pg['native18_report_sha256'] == sha(preflight_path)
assert preflight['full90_source_queue_sha256'] == sha(ROOT/source.QUEUE)
assert len(preflight['records']) == len({r['id'] for r in preflight['records']}) == 18
assert all(r['model_optimizer_updates'] == 0 and r['full_data_cells'] == 16819
           and r['parameters'] == 9093 and r['native_training_gradients_finite']
           and r['native_validation_loss_finite'] and not r['benchmark_performance_claim']
           for r in preflight['records'])

folder = ROOT/'projects/flow_matching_research/results/2026-10-16/EB100_four_sources_full_rows_and_GRASP48_v1'
assert not folder.exists()
folder.mkdir(parents=True)
copy(preflight_path, folder/'actual18_full_data_preflight.json')
copy(preflight_guard_path, folder/'actual18_preflight_guard.json')
summaries, proofs = [], []
with (folder/'all44_source_metrics_four_models.csv').open('x', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['id', 'seed', 'leaveout_timepoint', 'metric', 'value'])
    writer.writeheader()
    for j, r, provenance in validated:
        assert r['actual_completed_epochs'] == 1000 and r['optimizer_updates'] == 3000
        assert r['parameter_count'] == 9093 and len(r['test_metrics']) == 44
        key = f"seed{j['seed']}"
        copy(Path(provenance['result_path']), folder/(key+'_source_result.json'))
        copy(Path(provenance['guard_path']), folder/(key+'_source_guard.json'))
        proofs.append(dict(id=j['id'], **provenance))
        summaries.append(dict(id=j['id'], seed=j['seed'], leaveout_timepoint=1,
            actual_completed_epochs=r['actual_completed_epochs'], actual_optimizer_updates=r['optimizer_updates'],
            selected_checkpoint_optimizer_updates=r['posttest_selected_checkpoint_optimizer_updates'],
            heldout_W1=r['primary_heldout_W1'], full_native_scalar_count=44,
            training_seconds=r['training_seconds'], source_guard_passed=True,
            native_checkpoint_replay_complete=False, historical_paper_equivalence_certified=False))
        for metric, value in sorted(r['test_metrics'].items()):
            writer.writerow(dict(id=j['id'], seed=j['seed'], leaveout_timepoint=1, metric=metric, value=value))
dump(folder/'actual_four_complete_source_integrity.json', dict(passed=True, records=proofs,
    source_conditions_complete=4, full90_conditions_required=90,
    all_source_checkpoints_and_artifacts_rehashed=True, checkpoint_reinference_executed_here=False,
    full_paper_row_aggregation_executed=False, all_original_experiments_complete=False))

asset_dir = Path('F:/aicoding/IIA_Data/flow_matching/TrajectoryNet_EB100_official_v1')
acq = read(asset_dir/'verified_acquisition.json')
audit = read(asset_dir/'native_input_audit.json')
payload = (asset_dir/'eb_velocity_v5.npz').read_bytes()
assert len(payload) == acq['bytes'] == 28142608
assert hashlib.sha256(payload).hexdigest() == acq['sha256']
assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest() == acq['git_blob_sha1']
assert audit['records'][0]['cells'] == 16819 and audit['records'][0]['PC_dimensions'] == 100
copy(asset_dir/'verified_acquisition.json', folder/'official_asset_acquisition.json')
copy(asset_dir/'native_input_audit.json', folder/'full_input_audit.json')

window = ROOT/'experiments/runs/fm_CITE13176_GEO_nonpreemptive_data_window_v1'
wr = read(window/'window_receipt.json')
assert wr['all_metadata_locks_released'] and not wr['numerical_children_signalled'] and not wr['original_guards_changed']
assert wr['last_state']['state'] == 'release_no_original_headroom_after_grace'
assert not handle(16544, 1792116022.6208596)['exact_process_live']
copy(window/'window_receipt.json', folder/'bounded_window_terminal_receipt.json')
copy(window/'status.json', folder/'bounded_window_terminal_state.json')

watched = ['fm_CITE13176_GEO_raw_backed_QC_pca5_v2',
           'fm_otcfm_EB100_code_referenced_native_full90_v1',
           'fm_otcfm_EB100_full90_checkpoint_replay_v1',
           'fm_otcfm_EB100_all90_full_paper_rows_v1',
           'fm_otcfm_CITE13176_GEO_native60_deferred_v1']
live = []
for name in watched:
    base = ROOT/'experiments/runs'/name
    launch, state = read(base/'launch.json'), read(base/'status.json')
    actual = handle(launch['pid'], launch['create_time'])
    assert actual['exact_process_live']
    copy(base/'launch.json', folder/(name+'_launch.json'))
    copy(base/'status.json', folder/(name+'_state.json'))
    live.append(dict(run=name, handle=actual, observed_state=state.get('state'),
                     completed_ids=state.get('completed_ids'), verified_ids=state.get('verified_ids'),
                     completed_rows=state.get('completed_rows')))
dump(folder/'actual_live_handle_snapshot.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), records=live))
replay_c = read(ROOT/c['native_replay_configuration'])
groups = []
for g in c['groups']:
    ready = [i for i in g['required_case_ids']
             if rows.availability(q, replay_c, next(j for j in q['jobs'] if j['id'] == i)) is True]
    groups.append(dict(**g, currently_native_complete_ids=ready,
                       complete_paper_row_ready=len(ready)==15))
assert all(not g['complete_paper_row_ready'] for g in groups)
dump(folder/'complete_six_rows_required_not_yet_closed.json', dict(groups=groups,
    required_rows=6, required_distinct_models=90, local_uncertainty_unit='five seed means,each averaging three independent leaveout trainings',
    author_reported_SD_aggregation_certified=False, paper_scalar_targets_only_not_reproduction_claims=rows.TARGETS,
    further_Table4_baselines=['DSB','TrajectoryNet','RegCNF'], all_original_experiments_complete=False))

old_q_path = 'configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json'
from scripts.flow_matching.otcfm_EB_CITE_result_integrity_v1 import inspect as old_inspect
oq = read(ROOT/old_q_path)
oj = next(j for j in oq['jobs'] if j['protocol']=='released_yaml' and j['method']=='cfm'
          and j['dataset']=='released_author_EB' and j['leaveout_timepoint']==1 and j['seed']==42)
old, oldproof = old_inspect(oq, oj)
dump(folder/'separate_EB_input_tracks_not_causal_ablation.json', dict(
    old_input=dict(data=oj['data'], source=oldproof, epochs=old['actual_completed_epochs'],
        actual_updates=old['optimizer_updates'], selected_updates=old['posttest_selected_checkpoint_optimizer_updates'],
        validation_compatibility=oj['validation_compatibility'], heldout_W1=old['primary_heldout_W1']),
    code_referenced_input=dict(data=jobs[0]['data'], source=proofs[0], epochs=1000,
        actual_updates=3000, selected_updates=summaries[0]['selected_checkpoint_optimizer_updates'],
        validation_compatibility=jobs[0]['validation_compatibility'], heldout_W1=summaries[0]['heldout_W1']),
    causal_data_only_ablation=False, matched_optimizer_budget=False, matched_population_or_PC_artifact=False,
    single_seed_single_leaveout_not_full_paper_row=True, historical_equivalence_certified=False))

new_files = [rows.MODULE.replace('.','/')+'.py', rows.TEST, rows.CONFIG,
    'scripts/flow_matching/reserve_CITE_full_preprocessing_window_v1.py',
    'tests/test_fm_CITE_full_preprocessing_window_v1.py',
    'configs/runtime/fm_CITE13176_GEO_nonpreemptive_data_window.v1.json']
snapshots = []
for index, path in enumerate(new_files + [source.QUEUE, 'configs/acquisition/fm_TrajectoryNet_EB100_official.v1.json']):
    p = ROOT/path
    target = folder/f'source{index:02d}{p.suffix}'
    copy(p, target)
    snapshots.append(dict(path=path, sha256=sha(p), snapshot=target.name))
dump(folder/'source_manifest.json', snapshots)
for filename, tests, target in [
    ('EB100_full90_complete_paper_row_contract_20261016_v1_tests.xml',27,'validation_27_checks.xml'),
    ('CITE_full_preprocessing_nonpreemptive_window_v1_tests.xml',11,'validation_11_checks.xml')]:
    path = Path('F:/aicoding/IIA_Data/experiments')/filename
    xml_proof(path, tests)
    copy(path, folder/target)
copy(Path(__file__), folder/'publication_source.py')
dump(folder/'iteration_summary.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    complete_source_models=summaries, required_EB100_models=90, complete_paper_rows=0,
    native_replay_counts_snapshot=[dict(id=g['id'], count=len(g['currently_native_complete_ids'])) for g in groups],
    CITE_window='bounded attempt released;original full preprocessing still waits for original32GiB admission',
    all420_singlecell_conditions_defined_not_all_materialized_or_completed=True,
    original_scope=q['original_scope'], all_original_experiments_complete=False,
    preceding_goal_turn='hook repair verification;no numerical experiment completion claimed from that maintenance',
    current_goal_turn='progress:four complete full1000epoch EB100 sources verified and GNN seed1107 full replay closed'))
(folder/'.gitattributes').write_text('*.py -text\n*.json -text\n*.xml -text\n', encoding='utf-8')
(folder/'README.md').write_text(
    'EB100 original-code data track: four complete source trainings and complete-row verification\n\n'
    'The original16819-cell/100-PC TrajectoryNet Git asset passed full payload/Git-blob/SHA256 checks. '
    'All18 real-data method/recipe/leaveout preflights passed with zero optimizer updates. '
    'Four released-YAML I-CFM leaveout1 trainings,seeds42..45,each completed1000epochs/3000updates/9093parameters. '
    'All44 source metrics and every source checkpoint/artifact hash were checked. '
    'These are complete source executions;full native checkpoint replay and all15conditions per paper row are still required.\n\n'
    'Heldout W1 by seed42..45: '+', '.join(str(s['heldout_W1']) for s in summaries)+'. '
    'Paper Table4 reports I-CFM0.872+/-0.087,OT-CFM0.790+/-0.068,SB-CFM1.221+/-0.380 across heldout-time experiments. '
    'A single trained leaveout or four incomplete seeds cannot establish a reproduced paper row. '
    'The frozen collector covers all90conditions in six method/recipe rows,requires every source/native guard and full LP proof, '
    'and reports five seed-level means with three independently trained leaveouts per seed. '
    'The paper SD aggregation is not explicitly certified;both seed-level and15-model spreads stay separate. '
    'DSB,TrajectoryNet and RegCNF Table4 baselines remain separately required.\n\n'
    'The other EB artifact has18203cells/50PCs. Its seed42 W1=1.0388561487197876 and this code-referenced input W1=0.8616111874580383 '
    'are separate input tracks,not a causal data-only or matched-compute ablation:populations/PCs/update counts/validation batches/checkpoint choices differ. '
    'Source whitening uses all times and validation includes leftout-timepoint observations;this is not globally inductive validation.\n\n'
    'One finite scheduling window naturally acquired both idle family locks,retained original32GiB admission and all active workers, '
    'then released both after124.48seconds without admitting preprocessing. '
    'CITE29394cell x58676RNA full preprocessing remains live and waiting;the deferred60source/native pipeline remains live. '
    'No resource gate,tolerance,batch,epoch,data population or existing numerical process was changed. '
    'Validation27+11 contract checks passed;fixtures are not benchmark results. '
    'Preserve45papers/681method records/273data tasks/702scalar targets and every additional actual experiment.\n', encoding='utf-8')

addendum = ROOT/'projects/flow_matching_research/materials/TrajectoryNet_EB100_official_addendum.v1.json'
dump(addendum, dict(acquisition_configuration='configs/acquisition/fm_TrajectoryNet_EB100_official.v1.json',
    evidence=folder.relative_to(ROOT).as_posix(), source=acq, input_audit=audit['records'][0],
    original_code_reference='EBData(version=5)', source_code_referenced_public_asset_ready=True,
    full_native_source_input_ready=True, full90_training_complete=False, full90_native_replay_complete=False,
    historical_OT_CFM_recipe_identity_certified=False, full_paper_row_complete=False,
    raw_gene_barcode_identity_metadata_available=False, raw_data_uploaded_to_git=False,
    original_scope=q['original_scope'], all_original_experiments_complete=False))

# Independently close the one new actual GRASP model against the immutable47-ID closure.
runtime = read(ROOT/'configs/runtime/fm_grasp_completion_replay_queue.v1.json')
originals, executions, _, _ = grasp.verify(runtime)
identity = 'grasp_protocol_v3__gnn__swan__swan_sf__seed1107'
reference = grasp.completed_pipeline(runtime, identity)
pipelinepath = ROOT/reference['completed_pipeline_configuration']
pipeline = read(pipelinepath)
producer = binding(pipeline)
_, native, _ = grasp.current_native(pipeline['producer_configuration'])
record = native['records'][0]
assert len(native['records']) == 1 and record['canonical_id'] == identity
resultpath = ROOT/record['result_path']
result = read(resultpath)
grasp.check_complete_result(originals[identity], executions[identity], result)
assert sha(resultpath)==record['result_sha256'] and sha(record['checkpoint_path'])==record['checkpoint_sha256']
assert record['epochs_completed']==1500 and record['optimizer_updates']==93000
assert record['parameter_count']==result['parameter_count']==54706
for artifact in result['artifacts']:
    assert sha(ROOT/artifact['path'])==artifact['sha256']
for recipe in record['recipes']:
    assert recipe['recipe']==dict(source_samples=5,flow_evaluations=10)
    assert recipe['complete_validation_points']==3935 and recipe['complete_test_points']==72000
    for tag in ('original','replay'):
        assert sha(recipe[tag+'_score_path'])==recipe[tag+'_score_sha256']
independent_path=ROOT/pipeline['review_output_directory']/'independent_full_replay_verification.json'
independent=read(independent_path)
assert independent['passed'] and independent['actual_completed_recipe_count']==len(record['recipes'])==1
assert independent['all_original_forward_counts_and_recipe_budgets_verified']
assert independent['all_metrics_independently_recomputed_from_full_replay']
priorpath=ROOT/'projects/flow_matching_research/results/2026-10-16/grasp_tau8_seed1106_full47_model_increment_v1/actual_full47_model_increment_closure.json'
prior=read(priorpath)
assert len(set(prior['current_complete_original_ids']))==47
for name,digest in prior['current_source_proof_hashes'].items():
    assert sha(priorpath.parent/(name+'.json'))==digest
statepath=ROOT/runtime['state_root']/'status.json'
state=read(statepath)
assert set(state['verified_original_ids'])==set(prior['current_complete_original_ids'])|{identity}
gf=ROOT/'projects/flow_matching_research/results/2026-10-16/grasp_gnn_seed1107_full48_model_increment_v1'
gf.mkdir(parents=True,exist_ok=False)
sources=dict(original_result=resultpath,native_configuration=ROOT/pipeline['producer_configuration'],
    native_report=ROOT/producer['output_directory']/'checkpoint_replay_audit.json',
    native_guarded_receipt=ROOT/producer['state_root']/'receipt.json',
    completed_pipeline_configuration=pipelinepath,completed_pipeline_proof=ROOT/reference['completed_full_replay_proof'],
    independent_configuration=ROOT/pipeline['review_configuration'],independent_full_recipe_proof=independent_path,
    independent_guarded_receipt=ROOT/pipeline['state_root']/'receipt.json',
    original_training_resource_usage=ROOT/result['job']['output_directory']/'resource_usage.json',
    frozen_current_full9120_completion_state=statepath)
for name,path in sources.items():copy(path,gf/(name+'.json'))
cohort=[i for i in state['verified_original_ids'] if i.startswith('grasp_protocol_v3__gnn__swan__swan_sf__seed')]
assert sorted(int(i.rsplit('seed',1)[1]) for i in cohort)==list(range(1103,1108))
dump(gf/'actual_full48_model_increment_closure.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
    canonical_id=identity,complete_original_training_and_full_native_replay_and_independent_recipe_review=True,
    epochs=1500,optimizer_updates=93000,parameter_count=54706,recipes=record['recipes'],
    actual_current_unique_complete_models=48,current_complete_original_ids=state['verified_original_ids'],
    prior47_model_closure=priorpath.relative_to(ROOT).as_posix(),prior47_closure_sha256=sha(priorpath),
    current_source_proof_hashes={name:sha(path) for name,path in sources.items()},
    required_original_entity_models=9120,required_original_training_cohorts=480,required_original_training_seeds=10,
    complete_GNN_Swan_seeds=list(range(1103,1108)),pending_GNN_Swan_seeds=list(range(1108,1113)),
    full_ten_seed_GNN_cohort_complete=False,additional_model_inference=0,additional_optimizer_updates=0,
    inherited_native_boundary_metadata_discrepancy='Frozen prose names tau1/seed1104;actual canonical/profile/checkpoint/source/independent bindings prove GNN/seed1107. Preserve frozen artifacts.',
    author_equivalence_certified=False,strict_TAB_result=False,original_scope=runtime['original_scope'],
    all_original_experiments_complete=False))
copy(Path(__file__),gf/'closure_and_publication_source.py')
(gf/'.gitattributes').write_text('*.py -text\n*.json -text\n',encoding='utf-8')
(gf/'README.md').write_text('GRASP GNN / Swan / swan_sf / seed1107\n\n'
    'Full1500epochs/93000updates/54706parameters completed the original5source x10flow recipe,3935validation/72000test points. '
    'Native checkpoint restoration/reinference and independent full-array/forward-budget/metric checks passed. '
    'All48completed IDs equal immutable47-ID closure plus this model;all source/checkpoint/array/proof hashes were rechecked.\n\n'
    'Independent replay metrics: '+json.dumps(record['recipes'][0]['replayed_metrics'])+'\n\n'
    'GNN has5/10original Swan seeds1103..1107,with1108..1112 pending. '
    'Original score rtol0.0002/atol0.001,metric bounds,model/recipe budgets and resource guards remain. '
    'Frozen template prose incorrectly names tau1/seed1104;structured canonical/profile/checkpoint bindings prove GNN/seed1107. '
    'No bitwise replay,full10seed aggregate,strictTAB or original-author equivalence claim. '
    'Validation-only99% threshold/noPA F1=0.44885894704516227;test-label oracle BestF1=0.4633485558851054 remains diagnostic. '
    'Keep all9120entity models/480cohorts/10seeds and original45/681/273/702scope.\n',encoding='utf-8')

manifest=dict(previous_root=root_head,previous_parent=parent_head,
    paths=sorted(new_files+[addendum.relative_to(ROOT).as_posix()]+
        [p.relative_to(ROOT).as_posix() for base in [folder,gf] for p in base.rglob('*') if p.is_file()]))
dump(ROOT/'tmp/EB100_four_sources_GRASP48_publication_manifest_20261016_v1.json',manifest)
print(json.dumps(dict(packet=folder.relative_to(ROOT).as_posix(),new_GRASP_closed=identity,
    complete_GRASP_models=48,complete_EB100_sources=4,paper_rows_closed=0,files=len(manifest['paths']))))
