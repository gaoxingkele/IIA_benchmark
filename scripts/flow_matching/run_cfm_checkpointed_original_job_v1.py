"""Full original CFM reconstruction job with observation-only epoch checkpoints."""
import argparse
import ast
import inspect
from pathlib import Path

from scripts.flow_matching import run_cfm_ts_original_job as original
from scripts.flow_matching.cfm_complete_epoch_checkpoint_v1 import CompleteEpochLane, bind_fit, canonical_science


def verify_registration(config):
    for item in config['source_receipts']:
        if original.sha(original.ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen full checkpointed CFM registration changed')
    queue=original.read(original.ROOT/config['queue']);original.verify(queue)
    job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    canonical=config['canonical_job']
    if (len(queue['jobs'])!=285 or canonical_science(job)!=canonical_science(canonical)
            or job['recovery_original_job_id']!=canonical['id'] or job['seed']!=1105
            or job['method']!='node' or job['epochs']!=400 or job['expected_optimizer_updates']!=400
            or job['test_points_per_trajectory']!=500 or job['batch_size']!=258
            or config['checkpoint_interval']!=10 or config['resource_policy']!=queue['resource_policy']
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('Entire285 canonical scope, full400 NODE budget, same seed/data and all original guards required')
    if config['resume_full_epoch_checkpoint'] is not None and not config['resume_full_epoch_checkpoint_sha256']:
        raise ValueError('Any future resume requires an explicitly hashed complete native state')
    return queue,job


def worker_ast(source):
    tree=ast.parse(source);function=tree.body[0]
    locations=[i for i,n in enumerate(function.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and getattr(n.value.func.value,'id',None)=='model' and n.value.func.attr=='fit']
    if len(locations)!=1:
        raise ValueError('Exactly the complete original configured model fit is required')
    function.body.insert(locations[0],ast.parse('model.fit = _bind_complete_epoch_fit(model, job, artifacts)').body[0])
    return ast.fix_missing_locations(tree)


def run(config):
    queue,job=verify_registration(config)
    def bind(model,job,artifacts):
        lane=CompleteEpochLane(job,artifacts,interval=config['checkpoint_interval'],
            resume=config['resume_full_epoch_checkpoint'],resume_sha256=config['resume_full_epoch_checkpoint_sha256'])
        return bind_fit(model,lane)
    namespace=dict(original.__dict__,_bind_complete_epoch_fit=bind)
    exec(compile(worker_ast(inspect.getsource(original.run)),original.__file__,'exec'),namespace)
    namespace['run'](queue,job)
    ledger=original.read(Path(job['artifact_directory'])/'completed_full_epoch_checkpoints.json')
    if (config['resume_full_epoch_checkpoint'] is None
            and [r['completed_original_epochs'] for r in ledger]!=list(range(10,401,10))):
        raise ValueError('All40 complete original epoch checkpoints required for this full fresh400 retry')
    for record in ledger:
        if (not record['all_native_model_optimizer_rng_gradients_history_and_order_values_reread_bitwise']
                or not record['no_training_state_or_random_state_changed_by_save']
                or record['additional_optimizer_updates']!=0
                or original.sha(record['full_state_path'])!=record['full_state_sha256']):
            raise ValueError('Every original full-state observation must remain unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=original.read(original.ROOT/args.config)
    if args.probe_only:
        queue,job=verify_registration(config)
        print(dict(original_jobs=len(queue['jobs']),epochs=job['epochs'],seed=job['seed'],
            checkpoint_interval=config['checkpoint_interval'],author_equivalence_certified=False,
            torch_imported='torch' in __import__('sys').modules))
    else:
        run(config)
