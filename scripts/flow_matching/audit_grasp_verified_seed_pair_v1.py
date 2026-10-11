"""Audit two actual original seed results, including explicit recovery aliases."""
from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT,read,sha,compare_full_scores,SCIENTIFIC_FIELDS
from scripts.flow_matching.audit_grasp_complete_replay_union_v1 import metrics

RECIPE=dict(source_samples=5,flow_evaluations=10)


def verify_seed_bindings(config):
    if set(config['completed_seed_bindings'])!={'1103','1104'}:
        raise ValueError('Exactly two distinct original training seeds1103/1104 are required')
    seen=[]
    for seed,binding in config['completed_seed_bindings'].items():
        expected=f"grasp_protocol_v3__{config['profile']}__swan__swan_sf__seed{seed}"
        if binding['canonical_id']!=expected:raise ValueError('A recovery alias cannot create or relabel an original seed')
        seen.append(binding['result_path'])
    if len(set(seen))!=2:raise ValueError('The original independent seeds must refer to distinct actual training results')


def verify_actual_execution(job,result,execution_queue):
    actual=next((j for j in execution_queue['jobs'] if j['id']==result['job']['id']),None)
    if actual!=result['job'] or any(result['job'][k]!=job[k] for k in SCIENTIFIC_FIELDS):
        raise ValueError('Every actual execution must match its registered job and original canonical scientific fields')
    return actual


def audit_pairs(config):
    import numpy as np
    verify_seed_bindings(config)
    queue=read(ROOT/config['grasp_queue']);jobs={j['id']:j for j in queue['jobs']}
    profile=config['profile'];results={profile:[]};records=[]
    for seed in (1103,1104):
        binding=config['completed_seed_bindings'][str(seed)];canonical=binding['canonical_id'];job=jobs[canonical]
        review_config=read(ROOT/binding['completed_review_configuration'])
        completed=read(ROOT/binding['completed_full_replay_proof'])
        registered=next(r for r in review_config['full_results'] if r['canonical_id']==canonical)
        if (registered['result_path']!=binding['result_path'] or registered['result_sha256']!=binding['result_sha256']
                or review_config['pending_full_recipe_count']!=0 or completed['actual_completed_recipe_count']!=1
                or not completed['full_original_model_restoration_and_reinference_completed']
                or not completed['all_saved_full_score_arrays_and_identities_reverified']
                or not completed['all_original_forward_counts_and_recipe_budgets_verified']
                or not completed['all_metrics_independently_recomputed_from_full_replay']
                or completed['additional_training_seed_slots']!=0):
            raise ValueError('Both actual original models need complete native replay and full independent review')
        guard=completed['terminal_resource_receipt']
        if guard['exit_code']!=0 or guard['resource_guard_aborted'] or not guard['full_replay_passed']:
            raise ValueError('A partial or resource-aborted replay cannot form a completed seed cohort')
        path=ROOT/binding['result_path'];result=read(path)
        if sha(path)!=binding['result_sha256']:raise ValueError('Frozen actual original result changed')
        verify_actual_execution(job,result,read(ROOT/binding['execution_queue']))
        if (result['status']!='completed' or result['diagnostic'] or result['epochs_completed']!=1500
                or result['optimizer_updates']!=93000 or job['expected_optimizer_updates']!=93000
                or result['parameter_count']!=69244 or result['scored_points']!=72000 or result['validation_points']!=3935
                or result['job']['seed']!=seed or job['score_recipes']!=[RECIPE] or len(result['metrics'])!=1
                or result['paper_equivalence_certified'] or result['strict_TAB_result']):
            raise ValueError('Complete original model, seed, training/data/scoring budgets must remain unchanged')
        if sha(ROOT/job['model_config'])!=job['model_config_sha256']:raise ValueError('Original model configuration changed')
        oracle=result['metrics'][0]['metrics']['paper_retrospective'];control=result['metrics'][0]['metrics']['validation_only_control']
        if (not oracle['test_labels_used_for_threshold_optimization'] or oracle['point_adjustment']
                or control['point_adjustment'] or control['quantile']!=.99 or control['threshold_comparator']!='>'
                or control['test_features_used_for_calibration'] or control['test_labels_used_for_calibration']):
            raise ValueError('Preserve separate original oracle and validation-only noPA protocols')
        recipe=next(r for r in completed['recipes'] if r['canonical_id']==canonical and r['recipe']==RECIPE)
        if (not recipe['full_recipe_reinference_passed'] or recipe['seed']!=seed
                or recipe['epochs_completed']!=1500 or recipe['optimizer_updates']!=93000 or recipe['parameter_count']!=69244
                or recipe['forward_evaluation_counts']!={'validation':{'count':800,'windows':194300},'test':{'count':14100,'windows':3597550}}):
            raise ValueError('Every actual full native original recipe and forward count must complete')
        for item in result['artifacts']:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Actual complete checkpoint/history/scores changed')
        data=result['data_audit']
        for item in data['raw_sources']+[dict(path=data['prepared_path'],sha256=data['prepared_sha256'])]:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Full original raw/prepared data changed')
        for tag in ('original','replay'):
            if sha(recipe[tag+'_score_path'])!=recipe[tag+'_score_sha256']:raise ValueError('Full actual score arrays changed')
        with np.load(ROOT/data['prepared_path'],allow_pickle=False) as prepared, \
                np.load(recipe['original_score_path'],allow_pickle=False) as original, \
                np.load(recipe['replay_score_path'],allow_pickle=False) as replay:
            for key,expected in [('labels',prepared['labels']),('test_row_ids',np.arange(72000)),
                    ('validation_raw_row_ids',np.arange(data['split_cut'],data['raw_train_points']))]:
                if not np.array_equal(original[key],expected) or not np.array_equal(replay[key],expected):
                    raise ValueError('Every full raw-label and validation/test row identity must match')
            for tag,points in [('validation',3935),('test',72000)]:
                if original[tag+'_scores'].shape!=(points,):raise ValueError('Complete original score-point coverage required')
                compare_full_scores(replay[tag+'_scores'],original[tag+'_scores'],config['score_tolerance'])
            saved_metrics=metrics(original['labels'],original['test_scores'],original['validation_scores'])
            replay_metrics=metrics(replay['labels'],replay['test_scores'],replay['validation_scores'])
            for field in ('paper_retrospective','validation_only_control'):
                for name,value in saved_metrics[field].items():
                    native=result['metrics'][0]['metrics'][field][name];replayed=replay_metrics[field][name]
                    tolerance=config['metric_tolerance']
                    bound=tolerance['threshold_atol']+tolerance['threshold_rtol']*abs(native) if name=='threshold' else tolerance['metric_atol']
                    if abs(value-native)>bound or abs(replayed-native)>bound:raise ValueError('Full independently recalculated native metrics differ')
            disagreements=int(np.count_nonzero((original['test_scores']>np.quantile(original['validation_scores'],.99))
                !=(replay['test_scores']>np.quantile(replay['validation_scores'],.99))))
            if disagreements!=recipe['prediction_disagreements'] or disagreements/72000>config['metric_tolerance']['maximum_prediction_disagreement_rate']:
                raise ValueError('Full original thresholded native predictions differ')
        results[profile].append(result)
        records.append(dict(canonical_id=canonical,actual_execution_id=result['job']['id'],seed=seed,
            actual_execution_queue=binding['execution_queue'],result_path=binding['result_path'],result_sha256=sha(path),
            all_full_raw_truth_and_score_row_identities_verified=True,
            original_metrics_independently_recomputed=saved_metrics,replay_metrics_independently_recomputed=replay_metrics,
            prediction_disagreements=disagreements,full_native_model_replay_verified=True,
            completed_original_training_seed_slots=1,recovery_alias_is_not_additional_training_seed=True))
    return results,records
