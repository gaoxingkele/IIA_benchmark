"""Correct immutable receipt metadata after auditing every completed native batch."""
import argparse
import copy
import math

import numpy as np

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write
from scripts.flow_matching.audit_giflow_checkpoint_replay_resume_v3 import batch_coverage,compare_vectors


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    config=read(ROOT/args.config)
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Immutable completed native replay input changed')
    raw=read(ROOT/config['original_completed_proof']);result=read(ROOT/config['original_native_result'])
    original_batches=[r for r in result['artifacts'] if 'targets' in r]
    expected=config['expected_scope']
    for key,value in expected.items():
        if raw[key]!=value:raise ValueError('Full original native inference scope differs: '+key)
    if not all(raw[k] for k in ['full_checkpoint_contents_and_reinference_passed','full_data_masks_graph_scaler_and_splits_bitwise','actual_tested_weights_restored_bitwise','deterministic_model_forwards_consume_no_RNG']):
        raise ValueError('Full actual model and data restoration required')
    native=read(ROOT/config['original_replay_config']);prefix=native['preserved_prefix']['batch_count']
    corrected_counts=batch_coverage(raw['full_batch_receipts'],prefix,raw['this_attempt_forward_calls'])
    if len(original_batches)!=14 or len(raw['full_batch_receipts'])!=14:raise ValueError('Complete original native batch inventory required')
    array_checks=[];metrics=dict(mae=[],mse=[],mape_percent=[])
    total_windows=total_targets=0
    for index,(receipt,old) in enumerate(zip(raw['full_batch_receipts'],original_batches)):
        if receipt['batch']!=index or sha(receipt['path'])!=receipt['sha256'] or sha(old['path'])!=receipt['original_sha256']:
            raise ValueError('Full completed original batch arrays changed')
        with np.load(receipt['path'],allow_pickle=False) as replay,np.load(old['path'],allow_pickle=False) as original:
            for key in ['target','masked_flat_positions','batch_shape']:
                if not np.array_equal(replay[key],original[key]):raise ValueError('Original complete batch identity changed: '+key)
            agreement=compare_vectors(replay['prediction'],original['prediction'],native['prediction_tolerance'])
            target=replay['target'].astype(np.float64);pred=replay['prediction'].astype(np.float64)
            if not np.isfinite(target).all():raise ValueError('Full finite original targets required')
            delta=pred-target;nonzero=np.abs(target)>1e-6
            metrics['mae'].append(float(np.abs(delta).mean()));metrics['mse'].append(float((delta**2).mean()))
            metrics['mape_percent'].append(float(np.abs(delta[nonzero]/target[nonzero]).mean()*100))
            shape=replay['batch_shape'];windows=int(shape[0]);targets=len(target)
            if windows!=receipt['windows'] or targets!=receipt['targets']:raise ValueError('Complete per-batch scope changed')
            total_windows+=windows;total_targets+=targets
            array_checks.append(dict(batch=index,windows=windows,targets=targets,agreement=agreement,
                preserved_completed_batch=bool(receipt.get('preserved_completed_batch'))))
    if total_windows!=1747 or total_targets!=268613 or raw['this_attempt_window_evaluations']!=20*sum(r['windows'] for r in array_checks if not r['preserved_completed_batch']):
        raise ValueError('Every original window/target and all20 new sampler steps required')
    independent=dict(native_mean_batch_mae=float(np.mean(metrics['mae'])),native_mean_batch_mse=float(np.mean(metrics['mse'])),
        native_mean_batch_mape_percent=float(np.mean(metrics['mape_percent'])))
    independent['sqrt_native_mean_batch_mse']=math.sqrt(independent['native_mean_batch_mse'])
    for key,value in independent.items():
        if not math.isclose(value,raw['replayed_metrics'][key],**native['metric_tolerance']):raise ValueError('Independent complete unweighted native batch metric differs: '+key)
    corrected=copy.deepcopy(raw);corrected.update(corrected_counts)
    corrected.update(original_completed_proof=config['original_completed_proof'],original_completed_proof_sha256=sha(ROOT/config['original_completed_proof']),
        metadata_correction_only=True,additional_model_forwards_from_metadata_audit=0,
        replay_implementation_executed='scripts/flow_matching/audit_giflow_checkpoint_replay_resume_v2.py',
        future_corrected_receipt_implementation='scripts/flow_matching/audit_giflow_checkpoint_replay_resume_v3.py',
        correction_reason='V2 completed batch list aliased preserved prefix; appends changed len(preserved) in two receipt fields only. Actual per-batch flags and100 new forwards prove9 preserved and5 newly replayed native batches. Full numerical replay and original immutable receipt preserved.',
        independent_full_float64_unweighted_batch_metrics=independent,independent_full_array_checks=array_checks,
        all_original_experiments_complete=False,author_equivalence_certified=False)
    output=ROOT/config['output_directory']
    if output.exists():raise FileExistsError('Preserve original and corrected immutable receipts')
    output.mkdir(parents=True);write(output/'corrected_full_native_replay_audit.json',corrected)
    write(output/'validation.json',dict(passed=True,metadata_correction_only=True,source_receipts=config['source_receipts'],
        configuration=args.config,configuration_sha256=sha(ROOT/args.config),outputs={'corrected_full_native_replay_audit.json':sha(output/'corrected_full_native_replay_audit.json')}))
    print(dict(full_native_batches=14,preserved=prefix,newly_replayed=5,full_targets=total_targets,all_native_array_checks_passed=True))


if __name__=='__main__':main()
