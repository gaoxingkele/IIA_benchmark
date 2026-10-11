"""Verify complete GRASP replays against each actual original model's contract."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.audit_grasp_checkpoint_replay_resume_v2 import base, verify_reusable_recipes
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def check_terminal(report, receipt, config):
    expected = {r['canonical_id']:r for r in config['full_results']}
    ids = [r['canonical_id'] for r in report['records']]
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] or not receipt['full_replay_passed']
            or not report['full_checkpoint_contents_and_re_inference_passed']
            or report['full_job_count'] != config['full_job_count'] or len(ids)!=len(set(ids))
            or set(ids)!=set(expected) or len(ids)!=config['full_job_count']
            or report['additional_training_seed_slots']!=0 or report['cuda_context_initialized']):
        raise ValueError('Exactly the original complete native CPU replays must finish without abort or added seeds')
    for record in report['records']:
        registered = expected[record['canonical_id']]
        result = base.read(base.ROOT/registered['result_path'])
        if (result['status']!='completed' or result['epochs_completed']!=1500
                or record['epochs_completed']!=result['epochs_completed']
                or record['optimizer_updates']!=result['optimizer_updates']
                or record['optimizer_updates']!=result['job']['expected_optimizer_updates']
                or record['parameter_count']!=result['parameter_count']
                or not record['full_checkpoint_restoration_and_reinference_passed']
                or record['result_sha256']!=registered['result_sha256']
                or len(record['recipes'])!=len(result['metrics'])):
            raise ValueError('Each full original model, training budget and all recipe counts must match its actual result')


def verify(config, config_path):
    for source in config['source_receipts']:
        if base.sha(base.ROOT/source['path'])!=source['sha256']:
            raise ValueError('Frozen completed native replay evidence changed')
    producer = base.read(base.ROOT/config['completed_replay_configuration'])
    report_path = base.ROOT/producer['output_directory']/'checkpoint_replay_audit.json'
    receipt_path = base.ROOT/producer['state_root']/'receipt.json'
    report,receipt = base.read(report_path),base.read(receipt_path)
    check_terminal(report,receipt,config)
    if (report['configuration_sha256']!=base.sha(base.ROOT/config['completed_replay_configuration'])
            or report['score_tolerance']!=config['score_tolerance']
            or report['metric_tolerance']!=config['metric_tolerance']):
        raise ValueError('Keep the original full replay configuration and frozen tolerances')
    for record in report['records']:
        if (base.sha(record['checkpoint_path'])!=record['checkpoint_sha256']
                or base.sha(base.ROOT/record['result_path'])!=record['result_sha256']):
            raise ValueError('Completed actual full model checkpoint or native result changed')
    checked=verify_reusable_recipes(config)
    if len(checked)!=config['full_score_recipe_count'] or config['pending_full_recipe_count']!=0:
        raise ValueError('Every original complete full recipe must be independently rechecked')
    import numpy as np
    import importlib.util
    from scripts.flow_matching.grasp_protocol_runner import metric_records
    queue=base.read(base.ROOT/config['queue'])
    spec=importlib.util.spec_from_file_location('grasp_completed_review_original_metrics',base.ROOT/queue['native_metrics'])
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native=module.basic_metricor()
    records=[]
    for key,item in checked.items():
        registered=next(r for r in config['full_results'] if r['canonical_id']==key[0])
        result=base.read(base.ROOT/registered['result_path'])
        with np.load(item['replay_score_path'],allow_pickle=False) as actual,np.load(item['original_score_path'],allow_pickle=False) as saved:
            threshold=float(np.quantile(actual['validation_scores'],.99))
            original_threshold=float(np.quantile(saved['validation_scores'],.99))
            disagreements=int(np.count_nonzero((actual['test_scores']>threshold)!=(saved['test_scores']>original_threshold)))
            if (disagreements!=item['prediction_disagreements']
                    or disagreements/result['job']['test_points']>config['metric_tolerance']['maximum_prediction_disagreement_rate']):
                raise ValueError('Independently rechecked complete thresholded classifications differ')
            recalculated=metric_records(native,actual['labels'],actual['test_scores'],actual['validation_scores'])
            base.compare_metrics(recalculated,item['replayed_metrics'],config['metric_tolerance'])
        records.append(dict(canonical_id=key[0],profile=result['job']['profile'],seed=result['job']['seed'],
            epochs_completed=result['epochs_completed'],optimizer_updates=result['optimizer_updates'],
            parameter_count=result['parameter_count'],full_validation_points=result['job']['validation_points'],
            full_test_points=result['job']['test_points'],independently_recalculated_metrics=recalculated,**item))
    target=base.ROOT/config['verification_output_directory']
    target.mkdir(parents=True,exist_ok=False)
    proof=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        completed_native_replay_configuration=config['completed_replay_configuration'],
        completed_native_replay_report_sha256=base.sha(report_path),
        full_original_model_restoration_and_reinference_completed=True,
        all_saved_full_score_arrays_and_identities_reverified=True,
        all_original_forward_counts_and_recipe_budgets_verified=True,
        all_metrics_independently_recomputed_from_full_replay=True,
        actual_completed_recipe_count=len(checked),recipes=records,terminal_resource_receipt=receipt,
        additional_training_seed_slots=0,additional_model_inference_executed_by_this_review=False,
        original_scope=config['original_scope'],author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False,configuration=config_path,
        configuration_sha256=base.sha(base.ROOT/config_path),source_receipts=config['source_receipts'])
    exclusive_json(target/'independent_full_replay_verification.json',proof)
    print(json.dumps(dict(complete_native_replays_verified=config['full_job_count'],
        complete_recipes_rechecked=len(checked),prediction_disagreements=[r['prediction_disagreements'] for r in records])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args()
    verify(base.read(base.ROOT/args.config),args.config)
