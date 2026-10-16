"""Validate corrected full-training counters and every native timepoint metric."""
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.otcfm_singlecell_result_integrity_v2 import validate_record as legacy_validate
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as execution


DISTANCES = ('1-Wasserstein','2-Wasserstein','Mean_MSE','Mean_L2','Mean_L1','Median_MSE','Median_L2','Median_L1')


def full_metric_names(timepoints):
    if timepoints not in (4,5):raise ValueError('Registered complete four/five-timepoint cohorts required')
    return ({f'test/t{i}/{name}' for i in range(1,timepoints-1) for name in DISTANCES}
        | {f'test/{name}' for name in DISTANCES} | {f'test/t_out/{name}' for name in DISTANCES}
        | {'test/L1','test/L2','test/squared_L2','test/nfe'})


def validate_record(queue,job,result,guard):
    if (job.get("dataset") != "trajectorynet_EB100_code_referenced" or job["data"].get("cells") != 16819
            or job["data"].get("full_original_PC_dimensions") != 100
            or job["data"]["days"] != {"0":2381,"1":4163,"2":3278,"3":3665,"4":3332}):
        raise ValueError("Every complete original EB100 cell/PC/timepoint population required")
    if result['optimizer_updates']!=result['epoch_history'][-1]['optimizer_updates']:
        raise ValueError('New execution schema must retain actual complete training updates before checkpoint restoration')
    if set(result['test_metrics'])!=full_metric_names(len(job['data']['days'])):
        raise ValueError('Every native scalar/timepoint metric is required; do not truncate EB to Multiome or miscount36/44')
    selected=dict(result,optimizer_updates=result['posttest_selected_checkpoint_optimizer_updates'])
    legacy_validate(queue,job,selected,guard)
    for name in DISTANCES:
        if result['test_metrics'][f'test/t_out/{name}']!=result['test_metrics'][f'test/t{job["leaveout_timepoint"]}/{name}']:
            raise ValueError('Primary metrics must be the actual separately trained heldout timepoint, not an interior-time average')


def inspect(queue,job):
    base=Path(job['artifact_directory']);path=base/'result.json'
    guard_path=ROOT/queue['state_root']/'attempts'/base.name/'guard_receipt.json'
    result,guard=read(path),read(guard_path);validate_record(queue,job,result,guard)
    if guard['result_sha256']!=sha(path) or result['queue_sha256']!=sha(ROOT/execution.QUEUE):
        raise ValueError('Exact original EB100 full90 execution result and source queue required')
    if result['environment_lock_sha256']!=sha(ROOT/queue['environment_lock']):
        raise ValueError('Exact original author runtime lock required')
    if sha(ROOT/job['data']['path'])!=job['data']['sha256']:
        raise ValueError('Every original cell and complete PC input must remain immutable')
    for item in result['artifacts']:
        p=Path(item['path'])
        if not p.resolve().is_relative_to(base.resolve()) or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Every complete source checkpoint, training log and test artifact must match')
    if sha(result['best_checkpoint'])!=result['best_checkpoint_sha256']:
        raise ValueError('Original selected learned weights changed')
    return result,dict(result_path=str(path),result_sha256=sha(path),guard_path=str(guard_path),guard_sha256=sha(guard_path),
        actual_total_training_updates=result['optimizer_updates'],selected_checkpoint_updates=result['posttest_selected_checkpoint_optimizer_updates'],
        every_source_artifact_rehashed=len(result['artifacts']),full_scalar_metric_count=len(result['test_metrics']),
        native_reinference_executed_by_this_cold_audit=False,historical_paper_equivalence_certified=False)
