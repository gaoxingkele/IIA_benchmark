"""Advance the original GRASP Mean cohort after complete model and array replays."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime,timezone
import math
import statistics

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT,read,sha,compare_full_scores,SCIENTIFIC_FIELDS
from scripts.flow_matching.audit_grasp_complete_replay_union_v1 import metrics
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
from scripts.flow_matching.publish_pi_original_source_listing_v42 import interactive_table
from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

PROFILES=('mean',)
PROTOCOLS=[('paper_test_label_oracle_no_PA','paper_retrospective',('AP','ROC','Best_F1')),
           ('validation_only_1_percent_no_PA_control','validation_only_control',('precision','recall','f1'))]
RECIPE=dict(source_samples=5,flow_evaluations=10)


def check_delta(before,after):
    if len(before)!=len(after):
        raise ValueError('Every existing metric slot must be preserved')
    permitted={'n','mean','std','se','ci95_low','ci95_high','captured_utc','source','comparison','uncertainty_scope'}
    changed=[]
    for index,(old,new) in enumerate(zip(before,after)):
        if old==new:
            continue
        if (old['algorithm'] not in {'GRASP/'+p for p in PROFILES} or old['dataset']!='swan'
                or str(old['n'])!='1' or str(new['n'])!='2'
                or old['required_repeats']!='10' or new['required_repeats']!='10'
                or any(old[k]!=new[k] for k in old.keys()|new.keys() if k not in permitted)):
            raise ValueError('Only the six existing Mean one-of-ten metric rows may advance to two-of-ten')
        changed.append(index)
    if len(changed)!=6 or any(sum(after[i]['algorithm']=='GRASP/'+p for i in changed)!=6 for p in PROFILES):
        raise ValueError('Exactly all six unchanged protocols/metrics per original cohort must advance')
    return changed


def audit_pairs(config):
    import numpy as np
    queue=read(ROOT/config['grasp_queue'])
    jobs={j['id']:j for j in queue['jobs']}
    union=read(ROOT/config['preserved_thirteen_model_union'])
    if (union['full_original_models']!=13 or not union['all_complete_arrays_point_mapping_and_metrics_independently_verified']
            or not union['original_full_native_worker_exit_success_verified'] or not union['no_new_training_seed_slots']):
        raise ValueError('Preserved original full-model and full-array evidence is required')
    results,records={},[]
    for profile in PROFILES:
        completed=read(ROOT/config['completed_new_seed_proofs'][profile])
        if (not completed['full_original_model_restoration_and_reinference_completed']
                or not completed['all_saved_full_score_arrays_and_identities_reverified']
                or not completed['all_metrics_independently_recomputed_from_full_replay']
                or completed['additional_training_seed_slots']!=0):
            raise ValueError('New seed must have complete actual model replay and independently checked full arrays')
        guard=completed['terminal_resource_receipt']
        if guard['exit_code']!=0 or guard['resource_guard_aborted'] or not guard['full_replay_passed']:
            raise ValueError('Resource-aborted attempts are not full-model replay completion')
        results[profile]=[]
        for seed in (1103,1104):
            canonical=f'grasp_protocol_v3__{profile}__swan__swan_sf__seed{seed}'
            job=jobs[canonical]
            path=ROOT/job['output_directory']/'result.json'
            result=read(path)
            if (any(result['job'][k]!=job[k] for k in SCIENTIFIC_FIELDS)
                    or result['status']!='completed' or result['diagnostic']
                    or result['epochs_completed']!=1500 or result['optimizer_updates']!=93000
                    or job['expected_optimizer_updates']!=93000 or result['parameter_count']!=69244
                    or result['scored_points']!=72000 or result['validation_points']!=3935
                    or job['score_recipes']!=[RECIPE] or len(result['metrics'])!=1
                    or result['paper_equivalence_certified'] or result['strict_TAB_result']):
                raise ValueError('Original complete model, seed, data and full recipe budget must match')
            if sha(ROOT/job['model_config'])!=job['model_config_sha256']:
                raise ValueError('Original configured model changed')
            oracle=result['metrics'][0]['metrics']['paper_retrospective']
            validation=result['metrics'][0]['metrics']['validation_only_control']
            if (not oracle['test_labels_used_for_threshold_optimization'] or oracle['point_adjustment']
                    or validation['point_adjustment'] or validation['quantile']!=.99
                    or validation['threshold_comparator']!='>' or validation['test_features_used_for_calibration']
                    or validation['test_labels_used_for_calibration']):
                raise ValueError('Original retrospective oracle and validation-only noPA protocols must remain separate')
            if seed==1103:
                old=next(r for r in union['records'] if r['canonical_id']==canonical)
                if sha(path)!=old['original_complete_training_result_sha256']:
                    raise ValueError('Preserved original full training result changed')
                recipe=next(r for r in old['recipes'] if r['recipe']==RECIPE)
                if not recipe['passed']:
                    raise ValueError('Preserved complete full replay must pass')
            else:
                recipe=next(r for r in completed['recipes'] if r['canonical_id']==canonical and r['recipe']==RECIPE)
                if not recipe['full_recipe_reinference_passed']:
                    raise ValueError('New seed must complete its entire native recipe')
            for item in result['artifacts']:
                if sha(ROOT/item['path'])!=item['sha256']:
                    raise ValueError('Actual full checkpoint/history/scores changed')
            data=result['data_audit']
            for item in data['raw_sources']+[dict(path=data['prepared_path'],sha256=data['prepared_sha256'])]:
                if sha(ROOT/item['path'])!=item['sha256']:
                    raise ValueError('Original full raw/prepared dataset changed')
            for tag in ('original','replay'):
                if sha(recipe[tag+'_score_path'])!=recipe[tag+'_score_sha256']:
                    raise ValueError('Original/replayed complete scoring artifacts changed')
            with np.load(ROOT/data['prepared_path'],allow_pickle=False) as prepared, \
                    np.load(recipe['original_score_path'],allow_pickle=False) as original, \
                    np.load(recipe['replay_score_path'],allow_pickle=False) as replay:
                if not np.array_equal(original['labels'],prepared['labels']):
                    raise ValueError('Full original raw-label mapping differs')
                for key,expected in [('labels',prepared['labels']),('test_row_ids',np.arange(72000)),
                    ('validation_raw_row_ids',np.arange(data['split_cut'],data['raw_train_points']))]:
                    if not np.array_equal(original[key],expected) or not np.array_equal(replay[key],expected):
                        raise ValueError('Every original full validation/test row identity must match')
                for tag,points in [('validation',3935),('test',72000)]:
                    if original[tag+'_scores'].shape!=(points,):
                        raise ValueError('All original pointwise scores are required')
                    compare_full_scores(replay[tag+'_scores'],original[tag+'_scores'],config['score_tolerance'])
                saved_metrics=metrics(original['labels'],original['test_scores'],original['validation_scores'])
                replay_metrics=metrics(replay['labels'],replay['test_scores'],replay['validation_scores'])
                for field in ('paper_retrospective','validation_only_control'):
                    for name,value in saved_metrics[field].items():
                        native=result['metrics'][0]['metrics'][field][name]
                        replayed=replay_metrics[field][name]
                        tolerance=config['metric_tolerance']
                        bound=tolerance['threshold_atol']+tolerance['threshold_rtol']*abs(native) if name=='threshold' else tolerance['metric_atol']
                        if abs(value-native)>bound or abs(replayed-native)>bound:
                            raise ValueError('Independent complete original/replayed metrics differ')
                disagreements=int(np.count_nonzero(
                    (original['test_scores']>np.quantile(original['validation_scores'],.99)) !=
                    (replay['test_scores']>np.quantile(replay['validation_scores'],.99))))
                if disagreements!=recipe['prediction_disagreements'] or disagreements/72000>config['metric_tolerance']['maximum_prediction_disagreement_rate']:
                    raise ValueError('All actual native thresholded predictions must match')
            results[profile].append(result)
            records.append(dict(canonical_id=canonical,seed=seed,result_path=path.relative_to(ROOT).as_posix(),
                result_sha256=sha(path),all_full_raw_truth_and_score_row_identities_verified=True,
                original_metrics_independently_recomputed=saved_metrics,replay_metrics_independently_recomputed=replay_metrics,
                prediction_disagreements=disagreements,full_native_model_replay_verified=True))
    return results,records


def publish(config_path):
    config=read(ROOT/config_path)
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen two-seed publication source changed')
    results,records=audit_pairs(config)
    base=ROOT/config['base_publication']
    with (base/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);before=list(reader);headers=reader.fieldnames
    if len(before)!=8626 or len(headers)!=31 or sum(bool(r['mean']) for r in before)!=2128:
        raise ValueError('Keep the entire existing8626-row/31-field published scope')
    rows=deepcopy(before)
    captured=datetime.now(timezone.utc).isoformat()
    stats=[]
    for profile in PROFILES:
        for protocol,field,names in PROTOCOLS:
            for name in names:
                position=[i for i,r in enumerate(rows) if r['algorithm']=='GRASP/'+profile and r['dataset']=='swan'
                    and r['protocol']==protocol and r['metric']==name]
                if len(position)!=1:
                    raise ValueError('One previously published original metric slot is required')
                row=rows[position[0]]
                values=[r['metrics'][0]['metrics'][field][name] for r in results[profile]]
                if not math.isclose(float(row['mean']),values[0],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Preserve the exact previous original seed in the cohort')
                summary=independent_stats(values)
                if (not math.isclose(summary['mean'],statistics.mean(values),rel_tol=1e-12,abs_tol=1e-12)
                        or not math.isclose(summary['std'],statistics.stdev(values),rel_tol=1e-12,abs_tol=1e-12)):
                    raise ValueError('Independent original seed mean/sampleSTD disagrees')
                row.update({k:str(v) for k,v in summary.items()})
                row.update(source=config['cohort_summary_path'],captured_utc=captured,
                    comparison='two_of_ten_full_native_models_replayed_author_alignment_unverified',
                    uncertainty_scope='Two unique original independent training seeds1103/1104 of10; sampleSTD/SE/unclippedStudent-t95CI. Partial cohort; recipes, points and retries are not independent training seeds.')
                stats.append(dict(profile=profile,protocol=protocol,metric=name,original_seed_values=values,**summary))
    changed=check_delta(before,rows)
    target=ROOT/config['output_directory']
    target.mkdir(parents=True,exist_ok=False)
    exclusive_json(ROOT/config['cohort_summary_path'],dict(captured_utc=captured,full_native_original_model_replays=2,
        full_array_metric_reviews=records,original_training_seeds=[1103,1104],required_training_seeds=10,
        cohorts=1,metric_statistics=stats,score_tolerance=config['score_tolerance'],metric_tolerance=config['metric_tolerance'],
        original_scope=config['original_scope'],all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False))
    with (target/'all_algorithm_dataset_metrics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader();writer.writerows(rows)
    boundary=('Only6existing GRASP/Mean/SWAN metric rows advance from1/10to2/10original training seeds. '
        'All8,620other rows and capture times remain unchanged, including Pi original PA/pointwise results. '
        'Every original model completed1500epochs/93000updates and full native checkpoint replay. All3935validation/'
        '72000test points, raw-label mappings, recipes and metrics independently checked. Original frozen tolerances '
        'retained. SampleSTD,SE andunclippedStudent-t95CI describe only two seeds; no full10seed robustness or '
        'paper/TAB equivalence is certified. All45papers/681method-ablation/273original-data-task scope remains.')
    (target/'all_metrics.md').write_text('# Algorithm-dataset metrics v44\n\n'+boundary+'\n\n'
        +legacy.table(headers,[[r.get(k) or '-' for k in headers] for r in rows])+'\n',encoding='utf-8')
    import re
    page=interactive_table(rows,headers).replace('v42','v44')
    page=re.sub(r'<p>[^<]*</p>(?=<input)',
        '<p>GRASP Mean: six existing metrics now summarize 2/10 original seeds after complete native replay. '
        'Test-label oracle and validation-only protocols remain separate; paper/TAB alignment is uncertified.</p>',page)
    (target/'results.html').write_text(page,encoding='utf-8')
    (target/'README.rst').write_text('Algorithm-dataset metrics v44\n============================\n\n'+boundary
        +'\n\n8,626metric rows;2,128measured;6,498pending. The Mean cohort remains2/10seeds.\n',encoding='utf-8')
    exclusive_json(target/'publication_validation.json',dict(published_utc=captured,base_publication=config['base_publication'],
        aggregate_metric_rows=8626,fields=31,measured_metric_rows=2128,missing_metric_rows=6498,changed_metric_rows=6,
        changed_positions=changed,other8620rows_and_capture_times_unchanged=True,full_native_models_replayed_for_mean_cohort=2,
        partial_two_of_ten_seed_cohorts=1,additional_training_seed_slots=0,original_scope=config['original_scope'],
        all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False,
        source_receipts=config['source_receipts'],outputs={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
    print(dict(metric_rows=8626,changed_metrics=6,original_seeds_per_cohort=2,required_seeds_per_cohort=10))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    publish(parser.parse_args().config)
