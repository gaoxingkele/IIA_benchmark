from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import psutil
from scripts.flow_matching import run_grasp_completion_replay_queue_v1 as audit
from scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2 import binding

root,read,sha=audit.ROOT,audit.read,audit.sha
runtimepath='configs/runtime/fm_grasp_completion_replay_queue.v1.json'
runtime=read(root/runtimepath)
originals,executions,template,preserved=audit.verify(runtime)
statepath=root/runtime['state_root']/'status.json';raw=statepath.read_bytes();state=json.loads(raw)
owner=psutil.Process(state['pid']);assert owner.create_time()==state['create_time']
identities=state['verified_original_ids'];assert len(identities)>=33 and len(set(identities))==len(identities)
folder=root/'projects/flow_matching_research/results/2026-10-14/grasp_actual_complete_model_metric_catalog_v1'
folder.mkdir(exist_ok=True)
initial=folder/'frozen_actual_full9120_replay_controller_status.json'
if initial.exists():initial.rename(folder/'preserved_pre_catalog_legacy_record_resolution_status.json')
(folder/'frozen_actual_full9120_replay_controller_status.json').write_bytes(raw)
rows=[];groups={};proofs=[]
for identity in identities:
    if identity in preserved:
        reference=preserved[identity]
        nativepath=reference['native_configuration'];proofpath=reference['independent_proof']
        audit.check_preserved(reference,originals[identity],executions[identity])
    else:
        reference=audit.completed_pipeline(runtime,identity)
        saved=read(root/runtime['state_root']/'completed'/(identity+'.json'))
        assert reference==saved
        pipeline=read(root/reference['completed_pipeline_configuration']);producer=binding(pipeline)
        nativepath=pipeline['producer_configuration'];proofpath=reference['completed_full_replay_proof']
    native,report,receipt=audit.current_native(nativepath)
    record=next((r for r in report['records'] if r['canonical_id']==identity),None)
    if record is None:
        assert identity in preserved and reference['kind']=='legacy_union'
        prior=next(v for v in native['completed_full_replays_preserved'] if v['canonical_id']==identity)
        assert sha(root/prior['path'])==prior['sha256']
        saved_record=read(root/prior['path'])
        original=read(root/reference['result_path'])
        assert saved_record['canonical_id']==identity and all(r['full_recipe_reinference_passed'] for r in saved_record['completed_recipes'])
        for artifact in original['artifacts']:assert sha(root/artifact['path'])==artifact['sha256']
        checkpoint=next(a for a in original['artifacts'] if Path(a['path']).name=='checkpoint.pt')
        record=dict(canonical_id=identity,result_path=reference['result_path'],result_sha256=reference['result_sha256'],
            checkpoint_path=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],
            epochs_completed=original['epochs_completed'],optimizer_updates=original['optimizer_updates'],
            parameter_count=original['parameter_count'],recipes=saved_record['completed_recipes'])
        reference=dict(reference,actual_preserved_complete_per_model_record=prior,
            native_container_report_does_not_directly_cover_this_preserved_model=True)
    resultpath=root/record['result_path'];result=read(resultpath)
    audit.check_complete_result(originals[identity],executions[identity],result)
    assert sha(resultpath)==record['result_sha256'] and sha(record['checkpoint_path'])==record['checkpoint_sha256']
    job=originals[identity]
    assert record['epochs_completed']==1500 and record['optimizer_updates']==job['expected_optimizer_updates']
    proof_path=root/proofpath;assert proof_path.is_file()
    proofs.append(dict(canonical_id=identity,native_configuration=nativepath,native_configuration_sha256=sha(root/nativepath),
        independent_proof=proofpath,independent_proof_sha256=sha(proof_path),reference=reference))
    for recipe in record['recipes']:
        for key in ['original','replay']:
            assert sha(recipe[key+'_score_path'])==recipe[key+'_score_sha256']
        metrics=recipe['replayed_metrics']
        row=dict(canonical_id=identity,profile=job['profile'],dataset=job['dataset'],entity=job['entity'],
            seed=job['seed'],epochs=1500,optimizer_updates=record['optimizer_updates'],parameter_count=record['parameter_count'],
            complete_validation_points=recipe['complete_validation_points'],complete_test_points=recipe['complete_test_points'],
            recipe=recipe['recipe'],score_agreement=recipe['score_agreement'],replayed_metrics=metrics,
            current_score_array_and_checkpoint_and_result_hashes_verified=True,
            original_score_path=recipe['original_score_path'],original_score_sha256=recipe['original_score_sha256'],
            replay_score_path=recipe['replay_score_path'],replay_score_sha256=recipe['replay_score_sha256'],
            native_configuration=nativepath,independent_proof=proofpath,
            author_equivalence_certified=False,strict_TAB_result=False)
        rows.append(row)
        key=json.dumps([job['profile'],job['dataset'],job['entity'],recipe['recipe']],sort_keys=True)
        groups.setdefault(key,[]).append(row)
    if identity.endswith('seed1105'):
        case=folder/identity;case.mkdir()
        sources=dict(original_result=resultpath,native_configuration=root/nativepath,
            native_report=root/native['output_directory']/'checkpoint_replay_audit.json',
            native_guarded_receipt=root/native['state_root']/'receipt.json',independent_proof=proof_path)
        for name,path in sources.items():(case/(name+'.json')).write_bytes(path.read_bytes())
cohorts=[]
for key,records in groups.items():
    seeds=[r['seed'] for r in records];assert len(set(seeds))==len(seeds)
    values={}
    for protocol,metrics in [('paper_retrospective',['AP','ROC','Best_F1']),
        ('validation_only_control',['precision','recall','f1'])]:
        for metric in metrics:
            measurements=[r['replayed_metrics'][protocol][metric] for r in records]
            assert all(isinstance(v,(float,int)) for v in measurements)
            values[protocol+'/'+metric]=dict(mean=statistics.mean(measurements),
                sample_std=statistics.stdev(measurements) if len(measurements)>1 else None,
                per_complete_seed=dict(zip(map(str,seeds),measurements)))
    profile,dataset,entity,recipe=json.loads(key)
    cohorts.append(dict(profile=profile,dataset=dataset,entity=entity,recipe=recipe,
        complete_registered_seeds=sorted(seeds),required_registered_seeds=list(range(1103,1113)),
        complete_original_model_count=len(seeds),required_original_repeats=10,
        full_ten_seed_cohort_complete=len(seeds)==10,descriptive_interim_metrics=values,
        uncertainty_boundary='Variation across registered model/scoring sampling seeds on the fixed prepared entity; not a completed ten-seed estimate or an isolated training-RNG-only effect.'))
report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),unique_complete_original_models=len(identities),
    required_original_entity_models=9120,required_original_cohorts=480,required_original_training_seeds=10,
    current_full_source_native_independent_and_actual_artifact_proofs_reverified=True,
    all_original_score_recipe_rows=rows,all_current_complete_model_proof_references=proofs,interim_entity_cohorts=cohorts,
    original_scope=runtime['original_scope'],additional_model_inference=0,additional_optimizer_updates=0,
    failed_or_missing_seed_exclusion_for_full_cohort_performance=False,
    author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
(folder/'actual_full_complete_model_and_protocol_metric_catalog.json').write_text(json.dumps(report,indent=2)+'\n')
(folder/'closure_source.py').write_bytes(Path(__file__).read_bytes())
(folder/'.gitattributes').write_text('*.py -text\n')
lines=['Current complete GRASP model catalog: every listed full1500-epoch model, registered update budget, complete validation/test point grid and original score recipe has successful guarded native checkpoint replay and independent review. All current result/checkpoint/original-score/replay-score hashes and proof chains were rechecked. Keep the full9120entity models,480cohorts,10registered seeds and45/681/273research scope.\n',
    '| Profile | Entity | Complete seeds / 10 | Paper oracle Best F1 mean | Validation-only F1 mean |',
    '| --- | --- | ---: | ---: | ---: |']
for cohort in sorted(cohorts,key=lambda c:c['profile']):
    v=cohort['descriptive_interim_metrics']
    lines.append('| %s | %s | %d / 10 | %.6f | %.6f |'%(cohort['profile'],cohort['entity'],cohort['complete_original_model_count'],
        v['paper_retrospective/Best_F1']['mean'],v['validation_only_control/f1']['mean']))
lines += ['', 'Means and sample standard deviations are descriptive interim summaries of the complete seeds shown. No incomplete cohort is claimed as a full ten-seed result. Oracle test-label threshold metrics and validation-only controls are separate; neither proves original-author equivalence or strict TAB. Native score agreement is within the unchanged preregistered tolerance and need not be bitwise identical. No synthetic smoke, additional fitting, inference or recipe selection is included.']
(folder/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(full_complete_models=len(identities),full_recipe_rows=len(rows),interim_cohorts=len(cohorts),
    complete_seeds_by_profile={c['profile']:c['complete_original_model_count'] for c in cohorts})))
