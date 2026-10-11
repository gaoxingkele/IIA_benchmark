"""Verify complete native GiFlow replays with each seed's actual original masks.

Keep all native batch geometry, restored weights, Euler evaluations and frozen
CPU/GPU tolerances. Full target counts come from the immutable original result
and every original array, rather than another seed's masked-target count.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.audit_giflow_corrected_arrays_v1 import validate_selection
from scripts.flow_matching.run_giflow_serial_native_queue_v4 import complete
from scripts.flow_matching.giflow_native_protocol import verify as verify_queue
from scripts.flow_matching.verify_giflow_corrected_full_replay_v1 import verify_batch


def full_coverage(result, job):
    arguments=job['arguments']
    windows=result['data_binding']['windows']['test']
    batch=arguments['batch_size']
    original=[item for item in result['artifacts'] if 'targets' in item]
    targets=sum(item['targets'] for item in original)
    if (batch!=128 or arguments['window']!=24 or arguments['channel']!=1
            or windows!=1747 or result['data_binding']['nodes']!=36
            or result['full_test_batches']!=math.ceil(windows/batch)
            or len(original)!=result['full_test_batches']
            or targets!=result['test_targets_with_window_repetitions']
            or any(item['targets']<=0 for item in original)):
        raise ValueError('Full original Air36 native geometry, batches and seed-specific targets required')
    return dict(full_test_windows=windows,full_test_batches=len(original),full_masked_test_targets=targets,
        original_Euler_steps=20,observed_forward_calls=20*len(original),observed_window_evaluations=20*windows,
        original_batch_size=batch,actual_full_training_epochs=result['epochs_completed'],
        actual_optimizer_updates=result['optimizer_updates'],additional_training_seed_slots=0)


def check_terminal(proof,raw,receipt,result,job):
    if (receipt['exit_code']!=0 or receipt['resource_guard_aborted']
            or receipt['full_model_replay_passed'] is not True):
        raise ValueError('Actual complete successful model replay receipt required')
    if (proof['job_id']!=job['id'] or proof['seed']!=job['seed']
            or raw['job_id']!=job['id'] or raw['seed']!=job['seed'] or result['job']!=job):
        raise ValueError('Every actual original model/seed identity must match')
    for key,value in raw.items():
        if key!='boundary' and proof[key]!=value:
            raise ValueError('Reviewed clarification cannot alter actual full model replay evidence')
    required=full_coverage(result,job)
    for key,value in required.items():
        if proof.get(key)!=value:
            raise ValueError('Complete actual seed-specific native replay budget required: '+key)
    for key in ['full_checkpoint_contents_and_reinference_passed','full_data_masks_graph_scaler_and_splits_bitwise',
        'actual_tested_weights_restored_bitwise','tested_checkpoint_selected_on_validation',
        'source_test_block_unchanged','deterministic_model_forwards_consume_no_RNG',
        'original_training_seed_slots_unchanged','not_a_new_training_seed']:
        if proof.get(key) is not True:
            raise ValueError('Actual original full model restoration and evaluation proof required: '+key)
    if (proof['test_used_in_validation'] or proof['cuda_context_initialized'] or proof['author_equivalence_certified']
            or proof['selection']!=result['selection'] or proof['original_metrics']!=result['metrics']):
        raise ValueError('Frozen original reviewed model selection and native metric identity required')
    validate_selection(result)
    return required


def verify(config):
    for source in config['source_receipts']:
        if sha(ROOT/source['path'])!=source['sha256']:
            raise ValueError('Frozen complete native replay verification source differs: '+source['path'])
    replay_config=read(ROOT/config['replay_config'])
    for source in replay_config['source_receipts']:
        if sha(ROOT/source['path'])!=source['sha256']:
            raise ValueError('Actual full model/data/replay source changed: '+source['path'])
    queue=read(ROOT/replay_config['queue'])
    verify_queue(queue)
    job=next(j for j in queue['jobs'] if j['id']==replay_config['job_id'])
    if not complete(job):
        raise ValueError('Complete original native training and all full artifacts must verify')
    result=read(ROOT/job['output_directory']/'result.json')
    receipt=read(ROOT/config['worker_receipt'])
    raw=read(ROOT/config['raw_proof'])
    proof=read(ROOT/config['reviewed_proof'])
    if sha(ROOT/config['raw_proof'])!=proof['original_raw_replay_proof_sha256']:
        raise ValueError('Preserved original complete model replay proof changed')
    required=check_terminal(proof,raw,receipt,result,job)
    original=[r for r in result['artifacts'] if 'targets' in r]
    replayed=proof['full_batch_receipts']
    if len(replayed)!=required['full_test_batches']:
        raise ValueError('Every actually executed native full test batch required')
    batches=[]
    for index,(old,new) in enumerate(zip(original,replayed)):
        if new['batch']!=index or new['original_sha256']!=old['sha256']:
            raise ValueError('Actual original full batch identity/order differs')
        if sha(old['path'])!=old['sha256'] or sha(new['path'])!=new['sha256']:
            raise ValueError('Actual original or reinferred complete arrays changed')
        windows=min(job['arguments']['batch_size'],required['full_test_windows']-index*job['arguments']['batch_size'])
        with np.load(old['path'],allow_pickle=False) as saved,np.load(new['path'],allow_pickle=False) as actual:
            row=verify_batch(dict(actual),dict(saved),windows,old['targets'],replay_config['prediction_tolerance'])
        batches.append(dict(batch=index,original_path=old['path'],original_sha256=old['sha256'],
            replay_path=new['path'],replay_sha256=new['sha256'],**row))
    if (sum(r['targets'] for r in batches)!=required['full_masked_test_targets']
            or sum(r['windows'] for r in batches)!=required['full_test_windows']):
        raise ValueError('All actual seed-specific native windows and masked targets required')
    values={key:float(np.mean([b['independently_recomputed_metrics'][key] for b in batches]))
        for key in ['native_mean_batch_mae','native_mean_batch_mse','native_mean_batch_mape_percent']}
    values['sqrt_native_mean_batch_mse']=math.sqrt(values['native_mean_batch_mse'])
    for key,value in values.items():
        for expected in [proof['replayed_metrics'][key],proof['original_metrics'][key],result['metrics'][key]]:
            if not math.isclose(value,expected,**replay_config['metric_tolerance']):
                raise ValueError('Original full unweighted native batch-mean metric differs: '+key)
    old_binding=ROOT/job['output_directory']/'split_mask_graph_scaler.npz'
    new_binding=Path(replay_config['artifact_root'])/'split_mask_graph_scaler.npz'
    with np.load(old_binding,allow_pickle=False) as old,np.load(new_binding,allow_pickle=False) as new:
        if set(old.files)!=set(new.files) or any(not np.array_equal(old[k],new[k]) for k in old.files):
            raise ValueError('All original full masks/graph/scaler/grouped splits must be bitwise identical')
        arrays=len(old.files)
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),job_id=job['id'],seed=job['seed'],
        full_model_replay_and_independent_saved_arrays_verified=True,coverage=required,
        full_parameter_count=result['parameters'],full_batch_receipts=batches,
        independently_recomputed_metrics=values,original_metrics=result['metrics'],replayed_metrics=proof['replayed_metrics'],
        full_binding_array_count=arrays,original_binding_sha256=sha(old_binding),replay_binding_sha256=sha(new_binding),
        prediction_tolerance=replay_config['prediction_tolerance'],metric_tolerance=replay_config['metric_tolerance'],
        original_result_sha256=sha(ROOT/job['output_directory']/'result.json'),
        raw_full_native_replay_sha256=sha(ROOT/config['raw_proof']),reviewed_full_native_replay_sha256=sha(ROOT/config['reviewed_proof']),
        original_reports_preserved=True,old_fixed_seed_target_count_wording_not_reused=True,
        additional_training_seed_slots=0,training_seeds_completed_by_this_verification=1,training_seeds_required=5,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary=f"Actual original validation-selected full checkpoint and all original arrays verified. "
            f"All{required['full_test_windows']}overlapping windows/{required['full_masked_test_targets']}"
            f"actual seed-specific masked targets/{required['full_test_batches']}native batches/20Euler steps. "
            "Original unweighted batch means and MAPE cutoff retained; frozen CPU/GPU tolerances unchanged. "
            "Old raw/reviewed boundary text used another seed's target count and is preserved; this correction "
            "changes no arrays, metrics, forward budget or complete model restoration evidence.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=read(ROOT/args.config)
    target=ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve every independent complete replay audit')
    proof=verify(config)
    target.mkdir(parents=True)
    write(target/'independent_full_replay_verification.json',proof)
    write(target/'validation.json',dict(passed=True,source_receipts=config['source_receipts']+
        [dict(path=args.config,sha256=sha(ROOT/args.config))],
        outputs={'independent_full_replay_verification.json':sha(target/'independent_full_replay_verification.json')}))
    print(dict(full_model_replay_verified=True,coverage=proof['coverage'],additional_training_seeds=0))


if __name__=='__main__':
    main()
