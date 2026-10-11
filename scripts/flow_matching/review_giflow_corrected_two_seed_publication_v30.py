"""Independently recheck both native replay arrays and the four-row publication."""
import argparse
import csv
from datetime import datetime,timezone
import math
import statistics

from scipy.stats import t
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write
from scripts.flow_matching.verify_giflow_corrected_full_replay_v1 import verify as verify_seed393
from scripts.flow_matching.verify_giflow_corrected_full_replay_v2 import verify as verify_seed394

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        return reader.fieldnames,list(reader)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=read(ROOT/args.config)
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen independent full-native publication review source changed')
    one=verify_seed393(read(ROOT/config['seed393_verification_config']))
    two=verify_seed394(read(ROOT/config['seed394_verification_config']))
    if (one['coverage']['full_masked_test_targets']!=268613 or two['coverage']['full_masked_test_targets']!=264837
            or one['coverage']['observed_forward_calls']!=280 or two['coverage']['observed_forward_calls']!=280
            or not one['full_model_replay_and_independent_saved_arrays_verified']
            or not two['full_model_replay_and_independent_saved_arrays_verified']):
        raise ValueError('All full native model replays and every original/reinferred batch required')
    base,target=ROOT/config['base_publication'],ROOT/config['publication']
    fields0,before=rows(base/'all_algorithm_dataset_metrics.csv')
    fields,after=rows(target/'all_algorithm_dataset_metrics.csv')
    if fields0!=fields or len(fields)!=31 or len(before)!=7283 or len(after)!=7283:
        raise ValueError('All complete metric rows and metadata fields required')
    changed=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
    if len(changed)!=4 or sum(bool(r['mean']) for r in after)!=1809:
        raise ValueError('Exactly four existing measured metric slots may update')
    required={'native_mean_batch_mae','native_mean_batch_mse','native_mean_batch_mape_percent','sqrt_native_mean_batch_mse'}
    if {after[i]['metric'] for i in changed}!=required:
        raise ValueError('Every original native metric required exactly once')
    identity=['task','algorithm','dataset','protocol','metric','epochs','pattern','missing_ratio','score_recipe']
    stats={}
    for i in changed:
        old,new=before[i],after[i]
        if (any(old[k]!=new[k] for k in identity) or new['algorithm']!='GiFlow/main'
                or new['dataset']!='air36' or new['protocol']!='giflow_native/reviewed_corrected'
                or new['n']!='2' or new['required_repeats']!='5' or new['author_equivalence_certified']!='False'
                or new['strict_TAB_result']!='False' or new['evaluator_repeats']!=''
                or old['n']!='1' or old['captured_utc']==new['captured_utc']):
            raise ValueError('Only the original two-of-five trained-seed cohort may change')
        key=new['metric']
        samples=[one['original_metrics'][key],two['original_metrics'][key]]
        if float(old['mean'])!=samples[0]:
            raise ValueError('First seed must preserve actual original native metric')
        mean=statistics.mean(samples)
        std=statistics.stdev(samples)
        se=std/math.sqrt(2.)
        delta=float(t.ppf(.975,1))*se
        values=dict(mean=mean,std=std,se=se,ci95_low=mean-delta,ci95_high=mean+delta)
        for name,value in values.items():
            if not math.isclose(float(new[name]),value,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Independent between-training-seed Student-t statistics differ')
        stats[key]=dict(original_seed_metrics=samples,n=2,**values)
    unchanged=[name for name in ['overview.md','strict_algorithm_dataset.md','original_generation_jobs.csv','original_energy_jobs.csv']]
    for name in unchanged:
        if (base/name).read_bytes()!=(target/name).read_bytes():
            raise ValueError('All other original track exports must remain byte-identical')
    receipt=read(target/'publication_validation.json')
    for name,digest in receipt['outputs'].items():
        if sha(target/name)!=digest:raise ValueError('Published full metric artifact changed')
    for pin in receipt['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Published evidence chain changed')
    if (receipt['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
            or receipt['all_original_experiments_complete'] or receipt['author_equivalence_certified']):
        raise ValueError('Original whole-study scope and pending work must remain explicit')
    text=(target/'all_metrics.md').read_text(encoding='utf-8')
    from scripts.flow_matching.publish_complete_text_metadata_v29 import complete_text
    expected_table=complete_text(fields,after).split('\n\n',2)[2]
    if not text.endswith(expected_table):
        raise ValueError('All31 CSV fields and every complete text row must match')
    output=ROOT/config['output_directory']
    if output.exists():raise FileExistsError('Preserve every independent complete publication audit')
    output.mkdir(parents=True)
    proof=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        full_original_native_models_replayed=2,original_independent_training_seeds=[393,394],required_training_seeds=5,
        all_original_and_reinferred_arrays_independently_rechecked=True,full_native_test_batches=28,
        actual_original_and_replayed_array_files_checked=56,full_test_windows_per_seed=1747,
        actual_seed_specific_masked_targets=[268613,264837],actual_existing_native_replay_forward_calls=560,
        full_original_binding_arrays_per_seed=[one['full_binding_array_count'],two['full_binding_array_count']],
        actual_no_new_native_model_inference_executed_by_this_review=True,metrics=stats,
        changed_metric_row_positions=changed,unchanged_other_metric_rows_and_capture_times=7279,
        registered_metric_rows=7283,measured_metric_rows=1809,missing_metric_rows=5474,
        full_complete_CSV_fields=31,full_complete_text_all_fields_verified=True,other_track_exports_byte_identical=unchanged,
        all_artifact_and_evidence_hashes_verified=True,additional_training_seed_slots=0,
        original_scope=receipt['original_scope'],author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False)
    write(output/'independent_publication_review.json',proof)
    write(output/'validation.json',dict(passed=True,source_receipts=config['source_receipts']+
        [dict(path=args.config,sha256=sha(ROOT/args.config))],
        outputs={'independent_publication_review.json':sha(output/'independent_publication_review.json')}))
    print(dict(full_native_models=2,all_original_and_replayed_arrays=56,changed_metric_rows=4,
        other7279rows_exact=True,original_scope_retained=True,passed=True))

if __name__=='__main__':
    main()
