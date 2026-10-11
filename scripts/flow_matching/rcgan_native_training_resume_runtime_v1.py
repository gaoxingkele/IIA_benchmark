"""Resume the interrupted original500-epoch experiment from verified full state.

Only the observer bookkeeping and source epoch-range entry point are extended.
The frozen v2 observer, original train/evaluator functions, source loop body,
data sizes, optimizer settings and stopping budgets remain unchanged. Prior
losses retain checkpoint-recalculation provenance; missing timings stay missing.
"""
import ast
from copy import deepcopy
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,write,sha
from scripts.flow_matching import rcgan_native_training_runtime_v2 as original


def resume_experiment_ast(path):
    tree=original.experiment_ast(path)
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    if (not isinstance(loop.iter,ast.Call) or not isinstance(loop.iter.func,ast.Name)
            or loop.iter.func.id!='range' or len(loop.iter.args)!=1
            or not isinstance(loop.iter.args[0],ast.Name) or loop.iter.args[0].id!='num_epochs'):
        raise ValueError('Only the unchanged original range(num_epochs) entry point may be extended')
    loop.iter.args.insert(0,ast.Name(id='_start_epoch',ctx=ast.Load()))
    tree.body.insert(tree.body.index(loop),ast.parse('_native_restore_before_loop(globals())').body[0])
    return ast.fix_missing_locations(tree)


def observer_ast():
    tree=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run']
    if len(tree.body)!=1:raise ValueError('Exactly the frozen original full observer is required')
    counts=dict(ledger=0,counts=0,sigma=0,namespace=0,result=0,journal=0)
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target=node.targets[0]
            if isinstance(target,ast.Tuple) and [getattr(n,'id',None) for n in target.elts]==['epochs','evaluators','checkpoints']:
                if not isinstance(node.value,ast.Tuple) or not all(isinstance(n,ast.List) and not n.elts for n in node.value.elts):
                    raise ValueError('The frozen empty observer ledgers changed')
                node.value=ast.parse('(_prior_epochs,_prior_evaluators,_prior_checkpoints)',mode='eval').body
                counts['ledger']+=1
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self':
                if target.attr=='counts':
                    node.value=ast.parse('_verified_counts.copy()',mode='eval').body;counts['counts']+=1
                if target.attr=='last_sigma_count':
                    node.value=ast.parse("_verified_counts['sigma']",mode='eval').body;counts['sigma']+=1
            if isinstance(target,ast.Name) and target.id=='namespace':
                if not isinstance(node.value,ast.Call):raise ValueError('Frozen source execution namespace changed')
                node.value.keywords.extend([ast.keyword(arg='_start_epoch',value=ast.Name(id='_start_epoch',ctx=ast.Load())),
                    ast.keyword(arg='_native_restore_before_loop',value=ast.Name(id='_restore',ctx=ast.Load()))])
                counts['namespace']+=1
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
            old=next((k for k in node.keywords if k.arg=='full_original500_epoch_loop_completed'),None)
            if old is not None:
                old.value=ast.parse('False',mode='eval').body
                node.keywords.extend(ast.parse("dict(full_original500_epoch_budget_completed_with_verified_resume=True,"
                    "verified_prior_full_epochs=238,resume_segment_full_epochs=262,"
                    "resume_proof_binding=_resume_binding,original_canonical_job=_canonical_job,"
                    "trace_time_is_resume_segment_elapsed=True)",mode='eval').body.keywords)
                counts['result']+=1
        if isinstance(node,ast.FunctionDef) and node.name=='epoch_complete':
            positions=[i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                and n.value.func.value.id=='epochs' and n.value.func.attr=='append']
            if len(positions)!=1:raise ValueError('Frozen per-epoch loss journal hook changed')
            node.body.insert(positions[0]+1,ast.parse("write(output/'completed_full_epochs.json',epochs)").body[0])
            counts['journal']+=1
    if any(value!=1 for value in counts.values()):raise ValueError('Every exact resume observer extension must occur once: '+str(counts))
    return ast.fix_missing_locations(tree)


def verify_result(config,queue):
    job=config['recovery_job']
    result=read(ROOT/job['output_directory']/'result.json')
    if (result['job']!=job or result['original_canonical_job']!=config['canonical_job']
            or result['full_original500_epoch_loop_completed']
            or not result['full_original500_epoch_budget_completed_with_verified_resume']
            or not result['full_training_completed'] or not result['full_original_evaluators_completed']
            or result['verified_prior_full_epochs']!=238 or result['resume_segment_full_epochs']!=262
            or len(result['epochs'])!=500 or len(result['evaluators'])!=500 or len(result['checkpoints'])!=500
            or any(result['optimizer_updates'][k]!=642000 for k in ('D','G'))
            or result['independent_training_seed_slots']!=1 or result['author_equivalence_certified'] or result['strict_TAB_result']):
        raise ValueError('All500 original epochs, including238 verified prior and262 new full epochs, are required')
    sigma=0
    for epoch,(evaluation,checkpoint,loss) in enumerate(zip(result['evaluators'],result['checkpoints'],result['epochs'])):
        sigma+=evaluation['sigma_optimizer_updates']
        if (evaluation['epoch']!=epoch or checkpoint['epoch']!=epoch or loss['epoch']!=epoch
                or evaluation['complete_generated_points']!=4984 or evaluation['full_fit_points']!=996
                or evaluation['full_test_points']!=3988 or not 1<=evaluation['sigma_optimizer_updates']<=2000
                or (evaluation['sigma_optimizer_updates']<2000 and evaluation['stopping_delta']>.001)
                or any(checkpoint['optimizer_updates'][k]!=(epoch+1)*1284 for k in ('D','G'))
                or checkpoint['optimizer_updates']['sigma']!=sigma):
            raise ValueError('Every full native epoch evaluator/checkpoint ledger must retain exact cumulative budgets')
        for item in checkpoint['artifacts']+[dict(path=evaluation['array_path'],sha256=evaluation['array_sha256'])]:
            if sha(item['path'])!=item['sha256']:raise ValueError('Every original/resumed checkpoint and full evaluator must remain unchanged')
    if result['optimizer_updates']['sigma']!=sigma:raise ValueError('Full sigma update ledger differs')
    return result


def run(config,queue,binding):
    from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1 import verify_registration
    state_config=read(ROOT/config['checkpoint_state_configuration'])
    checked_queue,canonical,prior_output,evaluations,chain=verify_registration(state_config)
    if checked_queue!=queue or canonical!=config['canonical_job']:raise ValueError('Preserve the complete original five-seed queue and seed identity')
    proof_path=ROOT/state_config['output_directory']/'checkpoint_resume_state_audit.json'
    terminal_path=ROOT/state_config['state_root']/'receipt.json'
    proof,terminal=read(proof_path),read(terminal_path)
    if (not proof['passed'] or not proof['all238_full_model_and_optimizer_checkpoints_restored_bitwise']
            or not proof['all231_persisted_native_trace_losses_reproduced_bitwise_float32']
            or not proof['all4984_latest_generator_samples_replayed_bitwise']
            or not proof['original_next_D_G_update_repeated_bitwise_in_all_model_and_optimizer_variables']
            or terminal['exit_code']!=0 or terminal['resource_guard_aborted'] or not terminal['checkpoint_resume_state_passed']
            or binding['checkpoint_state_proof_sha256']!=sha(proof_path)
            or binding['checkpoint_state_terminal_sha256']!=sha(terminal_path)
            or binding['configuration_sha256']!=sha(ROOT/config['configuration_path'])
            or binding['native_evaluator_proof_sha256']!=sha(ROOT/queue['native_evaluator_proof'])):
        raise ValueError('Actual successful native state/evaluator proofs must bind this exact full resume')
    prior_epochs=[];prior_checkpoints=[];sigma=0
    for epoch,record in enumerate(proof['epochs']):
        if record['epoch']!=epoch or not record['full_model_and_optimizer_restore_bitwise_verified']:
            raise ValueError('Every prior epoch must have a verified actual full checkpoint')
        sigma+=evaluations[epoch]['sigma_optimizer_updates']
        prior_epochs.append({k:record[k] for k in ['epoch','D_loss','G_loss','loss_provenance',
            'original_persisted_trace_loss_bitwise_verified','original_elapsed_time','missing_original_elapsed_time_preserved']})
        prior_checkpoints.append(dict(epoch=epoch,optimizer_updates=dict(D=(epoch+1)*1284,G=(epoch+1)*1284,sigma=sigma),
            artifacts=deepcopy(record['checkpoint_artifacts']),original_full_checkpoint_independently_restored=True))
    if len(prior_epochs)!=238:raise ValueError('Exactly238 completed original epochs required')
    job=config['recovery_job']
    if job['seed']!=canonical['seed'] or job['id']!=canonical['id']+'__state_recovery_v1' or job['output_directory']==canonical['output_directory']:
        raise ValueError('Preserve the original seed and failed output under a distinct recovery attempt')
    counts=read(prior_output/'progress.json')['optimizer_updates']

    def restore(namespace):
        import numpy as np
        import tensorflow as tf
        from scripts.flow_matching.audit_rcgan_random_order_chain_v1 import nested_tuple
        checkpoint=prior_output/'native_checkpoints/epoch_237/model'
        reader=tf.train.load_checkpoint(str(checkpoint))
        variables=tf.global_variables()
        if {v.op.name:v.shape.as_list() for v in variables}!=reader.get_variable_to_shape_map():
            raise ValueError('The unchanged original graph must include every restored optimizer slot')
        session=namespace['sess']
        session.native_saver=tf.train.Saver(var_list=variables,max_to_keep=None)
        session.native_saver.restore(session,str(checkpoint))
        if any(not np.array_equal(value,reader.get_tensor(v.op.name)) for v,value in zip(variables,session.run(variables))):
            raise ValueError('Every original model and optimizer variable must restore bitwise before continuing')
        with np.load(chain['derived_order_artifact']['path'],allow_pickle=False) as state:
            order=state['cumulative_train_order']
            initial=state['initial_split_permutation']
            if not np.array_equal(initial[:36000][order],state['cumulative_train_image_ids']):
                raise ValueError('Verified full cumulative training-image order differs')
        namespace['samples']['train']=namespace['samples']['train'][order]
        namespace['labels']['train']=namespace['labels']['train'][order]
        with np.load(str(checkpoint.parent/'numpy_random_state.npz'),allow_pickle=False) as state:
            np.random.set_state((str(state['algorithm'].item()),state['keys'],int(state['position']),int(state['has_gauss']),float(state['cached_gaussian'])))
        original.random.setstate(nested_tuple(read(checkpoint.parent/'python_random_state.json')))
        best_epoch,best_value=0,1000
        for evaluation in evaluations:
            if evaluation['epoch']>10 and evaluation['native_mmd2']<best_value:
                best_epoch,best_value=evaluation['epoch'],evaluation['native_mmd2']
        namespace['best_epoch']=best_epoch;namespace['best_mmd2_so_far']=best_value
        if session.counts!=counts or session.last_sigma_count!=counts['sigma']:
            raise ValueError('Original cumulative optimizer counters must start at the verified checkpoint')
        write(ROOT/job['output_directory']/'actual_full_resume_restoration.json',dict(
            original_checkpoint=str(checkpoint),every_model_optimizer_variable_bitwise_restored=True,
            full36000_training_image_order_and_both_random_states_restored=True,
            completed_original_epochs=238,next_original_epoch=238,remaining_full_epochs=262,
            original_best_epoch=best_epoch,original_best_mmd2=best_value,optimizer_updates=counts,
            missing_original_times_preserved=True,proof_binding=binding,benchmark_performance=False))

    namespace=dict(vars(original),experiment_ast=resume_experiment_ast,_prior_epochs=prior_epochs,
        _prior_evaluators=deepcopy(evaluations),_prior_checkpoints=prior_checkpoints,_verified_counts=counts,
        _start_epoch=238,_restore=restore,_resume_binding=binding,_canonical_job=canonical)
    exec(compile(observer_ast(),str(Path(__file__).resolve()),'exec'),namespace)
    bound=dict(queue,native_evaluator_proof_sha256=binding['native_evaluator_proof_sha256'])
    namespace['run'](bound,job)
    verify_result(config,queue)
