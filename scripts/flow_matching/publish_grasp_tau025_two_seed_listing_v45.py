"""Advance the original GRASP tau=0.25 cohort after complete model and array replays."""
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

PROFILES=('tau_0p25',)
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
            raise ValueError('Only the six existing tau=0.25 one-of-ten metric rows may advance to two-of-ten')
        changed.append(index)
    if len(changed)!=6 or any(sum(after[i]['algorithm']=='GRASP/'+p for i in changed)!=6 for p in PROFILES):
        raise ValueError('Exactly all six unchanged protocols/metrics per original cohort must advance')
    return changed


from scripts.flow_matching.audit_grasp_verified_seed_pair_v1 import audit_pairs


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
    boundary=('Only6existing GRASP/tau=0.25/SWAN metric rows advance from1/10to2/10original training seeds. '
        'All8,620other rows and capture times remain unchanged, including Pi original PA/pointwise results. '
        'Every original model completed1500epochs/93000updates and full native checkpoint replay. All3935validation/'
        '72000test points, raw-label mappings, recipes and metrics independently checked. Original frozen tolerances '
        'retained. SampleSTD,SE andunclippedStudent-t95CI describe only two seeds; no full10seed robustness or '
        'paper/TAB equivalence is certified. All45papers/681method-ablation/273original-data-task scope remains.')
    (target/'all_metrics.md').write_text('# Algorithm-dataset metrics v45\n\n'+boundary+'\n\n'
        +legacy.table(headers,[[r.get(k) or '-' for k in headers] for r in rows])+'\n',encoding='utf-8')
    import re
    page=interactive_table(rows,headers).replace('v42','v45')
    page=re.sub(r'<p>[^<]*</p>(?=<input)',
        '<p>GRASP tau=0.25: six existing metrics now summarize 2/10 original seeds after complete native replay. '
        'Test-label oracle and validation-only protocols remain separate; paper/TAB alignment is uncertified.</p>',page)
    (target/'results.html').write_text(page,encoding='utf-8')
    (target/'README.rst').write_text('Algorithm-dataset metrics v45\n============================\n\n'+boundary
        +'\n\n8,626metric rows;2,128measured;6,498pending. The tau=0.25 cohort remains2/10seeds.\n',encoding='utf-8')
    exclusive_json(target/'publication_validation.json',dict(published_utc=captured,base_publication=config['base_publication'],
        aggregate_metric_rows=8626,fields=31,measured_metric_rows=2128,missing_metric_rows=6498,changed_metric_rows=6,
        changed_positions=changed,other8620rows_and_capture_times_unchanged=True,full_native_models_replayed_for_tau025_cohort=2,
        partial_two_of_ten_seed_cohorts=1,additional_training_seed_slots=0,original_scope=config['original_scope'],
        all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False,
        source_receipts=config['source_receipts'],outputs={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
    print(dict(metric_rows=8626,changed_metrics=6,original_seeds_per_cohort=2,required_seeds_per_cohort=10))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    publish(parser.parse_args().config)
