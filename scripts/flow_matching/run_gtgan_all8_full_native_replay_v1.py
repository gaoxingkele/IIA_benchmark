"""Replay all eight original GT-GAN commands, full generators and1200 trained critics.

Restore all native network checkpoints and original CPU/CUDA RNG state. Execute
the unchanged full native sampler/recovery and every imported trained TF graph
on its complete original test feed. No optimizer, fitting, additional seed,
shorter window set, generation chunk or relaxed numerical gate is permitted.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from scripts.flow_matching.gtgan_stock_observation_runtime_v2 import ROOT,read,sha,extended_budget,verify_author_environment,verify_extended_result
from scripts.flow_matching.gtgan_energy_metric_receipts_v1 import recompute_score
from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import exclusive
from scripts.flow_matching.timegan_native_common_v1 import write


def compiled_energy_budget():
    """Compile the unchanged full budget function without scheduling-only imports."""
    path=ROOT/'scripts/flow_matching/gtgan_energy_native_common_v1.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='full_budget']
    if len(tree.body)!=1:raise ValueError('One full unchanged Energy budget function required')
    namespace={}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace['full_budget']


energy_budget=compiled_energy_budget()


def queue_and_case(route):
    queue=read(ROOT/route['original_queue'])
    model=read(ROOT/queue['model_config'])
    case=next(c for c in model['cases']+model.get('release_evaluation_cases',[]) if c['id']==route['canonical_job']['case_id'])
    return queue,model,case


def budget(route):
    _,_,case=queue_and_case(route)
    if route['dataset']=='stock':
        return dict(extended_budget(case),native_release_checkpoint_evaluation=False)
    expected=energy_budget(case)
    return dict(expected,latent_noise_variables=case['parsed_original_arguments']['hidden_dim'],
        predictive_output_variables=1 if case['missing_rate']==0 else 28)


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen all-eight complete native replay source changed: '+pin['path'])
    if (config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
            or config['equality_gate']!='bitwise' or config['additional_generator_training_seeds']!=0
            or config['required_full_generations']!=60 or config['required_full_trained_evaluators']!=1200
            or config['required_native_network_checkpoint_artifacts']!=236
            or config['required_full_generated_windows']!=717270 or config['cpu_threads']!=2):
        raise ValueError('All eight full commands, all60 generations/1200 critics and original research scope required')
    if (set(config['original_queues'])!={'stock','energy'} or len(config['routes'])!=8
            or len({r['canonical_job']['id'] for r in config['routes']})!=8):
        raise ValueError('Every one of the eight distinct original commands required')
    expected_ids=set();periods=0;checkpoints=0;windows=0
    for dataset in ['stock','energy']:
        entries=[r for r in config['routes'] if r['dataset']==dataset]
        if len(entries)!=4:raise ValueError('All four commands on each original dataset required')
        queues={r['original_queue'] for r in entries}
        if queues!={config['original_queues'][dataset]}:raise ValueError('One exact immutable original queue per dataset required')
        queue=read(ROOT/next(iter(queues)))
        expected_ids.update(j['id'] for j in queue['jobs'])
        for route in entries:
            canonical=next(j for j in queue['jobs'] if j['id']==route['canonical_job']['id'])
            b=budget(route)
            if (route['canonical_job']!=canonical or route['settings']!=queue['settings']
                    or config['settings']!=queue['settings'] or config['resource_lock']!=queue['resource_lock']
                    or config['python']!=queue['python']
                    or route['result_path']!=(Path(canonical['output_directory'])/'result.json').as_posix()
                    or route['full_generation_shape']!=[b['full_generation_windows'],24,b['variables']]
                    or route['full_noise_shape']!=[b['full_generation_windows'],24,b['latent_noise_variables']]
                    or route['full_periodic_evaluations']!=b['full_periodic_evaluations']
                    or route['full_trained_evaluators']!=20*b['full_periodic_evaluations']
                    or route['original_truth_audit']['full_shape']!=route['full_generation_shape']
                    or not route['native_completion_receipt']):
                raise ValueError('Exact original full native job, shapes, budgets and guards required')
            periods+=b['full_periodic_evaluations'];windows+=b['full_periodic_evaluations']*b['full_generation_windows']
            checkpoints+=2 if b['native_release_checkpoint_evaluation'] else 4*b['full_periodic_evaluations']
    if (expected_ids!={r['canonical_job']['id'] for r in config['routes']}
            or (periods,checkpoints,windows)!=(60,236,717270)):
        raise ValueError('No full original native command or periodic evaluation may be dropped')
    return config['routes']


def metric_output_contract(role,function_name,source):
    """Bind restored graph outputs to the untouched native scope/Sigmoid constructor."""
    tree=ast.parse(source)
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name)
    name='discriminator' if role=='discriminative' else 'predictor'
    functions=[n for n in native.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(functions)!=1:raise ValueError('One untouched native evaluator network constructor required')
    network=functions[0]
    scopes=[n for n in ast.walk(network) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='variable_scope']
    sigmoids=[n for n in ast.walk(network) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sigmoid']
    if (len(scopes)!=1 or len(sigmoids)!=1 or not scopes[0].args
            or not isinstance(scopes[0].args[0],ast.Constant) or scopes[0].args[0].value!=name
            or sigmoids[0].keywords):
        raise ValueError('Exact original evaluator graph output scope and default Sigmoid name required')
    return [name+'/Sigmoid:0',name+'_1/Sigmoid:0'] if role=='discriminative' else [name+'/Sigmoid:0']


def energy_mode_contract(source):
    """The author always sets generator/recovery train/eval together in Energy."""
    tree=ast.parse(source);pairs=0
    for parent in ast.walk(tree):
        for _,value in ast.iter_fields(parent):
            if not isinstance(value,list):continue
            for index,n in enumerate(value):
                if (not isinstance(n,ast.Expr) or not isinstance(n.value,ast.Call)
                        or not isinstance(n.value.func,ast.Attribute) or n.value.func.attr not in ['train','eval']
                        or not isinstance(n.value.func.value,ast.Name) or n.value.func.value.id!='generator'):continue
                method=n.value.func.attr
                following=value[index+1:index+3]
                recovery=[v for v in following if isinstance(v,ast.Expr) and isinstance(v.value,ast.Call)
                    and isinstance(v.value.func,ast.Attribute) and isinstance(v.value.func.value,ast.Name)
                    and v.value.func.value.id=='recovery' and v.value.func.attr==method]
                if len(recovery)!=1 or n.value.args or n.value.keywords:
                    raise ValueError('Original Energy generator/recovery modes no longer paired')
                pairs+=1
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='recovery' and n.func.attr in ['train','eval']]
    if pairs!=len(calls) or pairs!=4:
        raise ValueError('All four original Energy mode pairs required')
    return dict(generator_recovery_modes_paired=True,original_mode_pairs=pairs,
        source_ast_sha256=hashlib.sha256(ast.dump(tree).encode('utf-8')).hexdigest())


def compiled_energy_verifier():
    path=ROOT/'scripts/flow_matching/run_gtgan_energy_native_queue_v1.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_result']
    if len(tree.body)!=1:raise ValueError('One full original Energy result verifier required')
    namespace=dict(ROOT=ROOT,json=json,Path=Path,sha=sha,full_budget=energy_budget)
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace['verify_result']


def successful_native_receipt(route):
    receipt_path=ROOT/route['native_completion_receipt']
    if not receipt_path.exists():return None
    receipt=read(receipt_path)
    if (receipt.get('exit_code')==0 and receipt.get('resource_guard_aborted') is False
            and receipt.get('full_native_result_verified') is True):return receipt
    return None


def original_result(config,route):
    queue,_,_=queue_and_case(route)
    if route['dataset']=='stock':
        return verify_extended_result(queue,route['canonical_job'],sha(ROOT/config['stock_observation_configuration']))
    return compiled_energy_verifier()(queue,route['canonical_job'])


def native_checkpoints(row,result,expected):
    if expected['native_release_checkpoint_evaluation']:
        checkpoints={Path(item['path']).stem:item for item in result['released_checkpoints']}
        if set(checkpoints)!={'generator','recovery'}:raise ValueError('Both original release input networks required')
    else:
        step=str(row['native_joint_step']);checkpoints={}
        for role in ['embedder','generator','recovery','discriminator']:
            items=[item for item in result['checkpoints'] if Path(item['path']).name==role+step+'.pt']
            if len(items)!=1:raise ValueError('Each complete native network checkpoint per source evaluation required')
            checkpoints[role]=items[0]
    for item in checkpoints.values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Actual native network checkpoint changed')
    return checkpoints


def tensor_digest(values,np):
    h=hashlib.sha256()
    for name,array in sorted(values.items()):
        value=np.asarray(array)
        h.update(name.encode('utf-8'));h.update(str(value.dtype).encode('ascii'))
        h.update(str(value.shape).encode('ascii'));h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def exact_arrays(actual,expected,label,np):
    if (actual.shape!=expected.shape or actual.dtype!=expected.dtype
            or not np.isfinite(actual).all()
            or np.ascontiguousarray(actual).tobytes()!=np.ascontiguousarray(expected).tobytes()):
        raise ValueError('Full native bitwise replay differs: '+label)


def exact_window_multiset(actual,expected,np):
    """Compare all window bytes including duplicates, allowing the native shuffle."""
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not np.isfinite(actual).all():
        raise ValueError('Every complete original truth window and native dtype required')
    width=actual.shape[1]*actual.shape[2]*actual.dtype.itemsize
    def rows(array):
        return np.ascontiguousarray(array).reshape(array.shape[0],-1).view(np.dtype('V'+str(width))).reshape(-1)
    if not np.array_equal(np.sort(rows(actual)),np.sort(rows(expected))):
        raise ValueError('Full native truth window multiset differs from original complete loader audit')


def verify_native_truth(row,route,np):
    audit=route['original_truth_audit']
    if sha(audit['artifact_path'])!=audit['artifact_sha256']:
        raise ValueError('Complete original source-loader truth audit changed')
    with np.load(audit['artifact_path'],allow_pickle=False) as saved:
        # The unchanged native loader's to_tensor is torch.from_numpy(data).float().
        truth=np.asarray(saved['truth']).astype(np.float32)
    with np.load(row['artifact_path'],allow_pickle=False) as saved:actual=np.array(saved['real'],copy=True)
    if list(actual.shape)!=audit['full_shape']:raise ValueError('Complete original raw truth shape required')
    exact_window_multiset(actual,truth,np)
    return dict(original_source_loader_audit=audit,full_truth_shape=list(actual.shape),
        all_native_float32_truth_window_bytes_and_duplicate_counts_match=True,
        native_window_order_may_differ=True,no_truth_window_omitted=True)


def import_native_runtime(route,case):
    runtime=Path(route['canonical_job']['runtime_root'])
    preparation=read(runtime/'runtime_preparation_receipt.json')
    repairs={p['path']:p['repaired_sha256'] for p in preparation['runtime_repairs']}
    if len(preparation['complete_copied_assets'])!=245:raise ValueError('Complete original native runtime required')
    # Native training writes private pretrained .pt files in some Stock cases.
    # Replays use the immutable completed-checkpoint receipts, never those files.
    for item in preparation['complete_copied_assets']:
        if Path(item['path']).suffix.lower()=='.pt':continue
        if sha(runtime/item['path'])!=repairs.get(item['path'],item['original_sha256']):
            raise ValueError('Original prepared runtime source/raw/support asset changed: '+item['path'])
    sys.path.insert(0,str(runtime))
    entry=runtime/Path(case['source_entry']).name
    spec=importlib.util.spec_from_file_location('native_gtgan_full_replay',entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def construct_native_models(module,case,expected,torch):
    from scripts.flow_matching.audit_gtgan_native_models_v1 import original_gtgan_branch
    source=ROOT/case['source_entry']
    branch=original_gtgan_branch(source.read_text(encoding='utf-8'))
    arguments=dict(case['parsed_original_arguments'])
    arguments['effective_shape']=expected['latent_noise_variables']
    destination=Path(module.__file__).parent/'native_run_checkpoints'
    arguments['save_dir']=str(destination) if Path(module.__file__).name=='GTGAN_energy.py' else destination
    namespace=dict(module.__dict__,args=SimpleNamespace(**arguments),input_size=expected['variables'],
        hidden_size=24,num_layers=3,x_hidden=48,device=torch.device('cuda'))
    exec(compile(branch,str(source)+':unchanged_full_original_constructor','exec'),namespace)
    return namespace


def verify_native_model_counts(queue,case,namespace):
    proof=read(ROOT/queue['construction_proof'])
    matches=[row for row in proof['cases'] if row['missing_rate']==case['missing_rate']]
    if len(matches)!=1:raise ValueError('Exact full original native constructor case proof required')
    counts={}
    for name in ['embedder','generator','supervisor','recovery','discriminator']:
        count=sum(v.numel() for v in namespace[name].parameters());counts[name]=count
        original=matches[0]['counts'][name]
        expected=original['parameters'] if isinstance(original,dict) else original
        if count!=expected:raise ValueError('Full native constructor parameter count differs: '+name)
    return counts


def replay_generation(row,result,expected,namespace,folder,np,torch):
    checkpoints=native_checkpoints(row,result,expected)
    models={key:namespace[key] for key in ['embedder','generator','supervisor','recovery','discriminator']}
    loaded=[]
    for name,item in checkpoints.items():
        state=torch.load(item['path'],map_location='cpu')
        if not state or any(not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('All complete native state tensors must be finite')
        models[name].load_state_dict(state,strict=True)
        restored=models[name].state_dict()
        if set(restored)!=set(state):
            raise ValueError('Complete native checkpoint state did not restore bitwise')
        for key,value in state.items():
            exact_arrays(restored[key].detach().cpu().numpy(),value.detach().cpu().numpy(),
                name+' complete checkpoint '+key,np)
        loaded.append(dict(role=name,input_checkpoint=item,state_tensors=len(state),strict_full_state_bitwise_loaded=True))
    sampler=row['sampler_input'];path=Path(sampler['path'])
    if sha(path)!=sampler['sha256']:raise ValueError('Original native sampler inputs changed')
    with np.load(path,allow_pickle=False) as saved:
        noise=np.array(saved['latent_noise'],copy=True);times=np.array(saved['conditioning_times'],copy=True)
        cpu=np.array(saved['torch_cpu_rng'],copy=True)
        cuda_keys=sorted((k for k in saved.files if k.startswith('cuda_rng_')),key=lambda k:int(k.split('_')[-1]))
        cuda=[np.array(saved[k],copy=True) for k in cuda_keys]
        observations=np.array(saved['recovery_observation_times'],copy=True) if 'recovery_observation_times' in saved.files else None
    n=expected['full_generation_windows']
    if noise.shape!=(n,24,expected['latent_noise_variables']) or times.shape!=(n,24,1):
        raise ValueError('Every original latent noise and conditioning time required')
    with np.load(row['artifact_path'],allow_pickle=False) as generation:
        native=np.array(generation['generated'],copy=True)
        if observations is None:
            original_input=generation['conditioning_input']
            if original_input.shape!=(n,24,expected['variables']+1):
                raise ValueError('Every original Energy recovery observation time required')
            observations=np.array(original_input[:,:,-1],copy=True)
    if (observations.shape!=(n,24) or not np.isfinite(observations).all()
            or not np.isfinite(noise).all() or not np.isfinite(times).all()):
        raise ValueError('Full native replay input values required')
    mode=sampler['native_generator_training_mode']
    recovery_mode=sampler.get('native_recovery_training_mode',mode)
    grad=sampler.get('native_grad_enabled',False)
    if any(type(value) is not bool for value in [mode,recovery_mode,grad]):
        raise ValueError('Exact native generator/recovery/gradient modes required')
    models['generator'].train(mode);models['recovery'].train(recovery_mode)
    before=tensor_digest({name+'.'+key:value.detach().cpu().numpy()
        for name,model in models.items() for key,value in model.named_parameters()},np)
    z=torch.from_numpy(noise).cuda();conditioning=torch.from_numpy(times).cuda();obs=torch.from_numpy(observations).cuda()
    if len(cuda)!=torch.cuda.device_count() or not cuda or cpu.dtype!=np.uint8 or any(v.dtype!=np.uint8 for v in cuda):
        raise ValueError('Every original native CPU/CUDA RNG byte state required')
    torch.set_rng_state(torch.from_numpy(cpu));torch.cuda.set_rng_state_all([torch.from_numpy(v) for v in cuda])
    with torch.set_grad_enabled(grad):
        h_hat=namespace['run_model'](namespace['args'],models['generator'],z,conditioning,namespace['device'],z=True)
        x_hat=models['recovery'](h_hat,obs)
    actual=x_hat.detach().cpu().numpy()
    exact_arrays(actual,native,'all native generated windows',np)
    after=tensor_digest({name+'.'+key:value.detach().cpu().numpy()
        for name,model in models.items() for key,value in model.named_parameters()},np)
    if before!=after:raise ValueError('Native replay changed a model parameter')
    artifact=folder/('full_generation_replay_'+str(row['native_joint_step'])+'.npz')
    with artifact.open('xb') as stream:np.savez_compressed(stream,generated=actual)
    report=dict(native_joint_step=row['native_joint_step'],full_generated_shape=list(actual.shape),
        complete_original_native_sampler_and_recovery_executed=True,full_prediction_bitwise_equal=True,
        full_native_network_state_loads=loaded,model_parameters_unchanged=True,model_parameter_sha256=before,
        sampler_input=sampler,source_generation_artifact_sha256=row['artifact_sha256'],
        original_sampler_function=namespace['run_model'].__name__,
        full_noise_conditioning_and_recovery_time_shapes=[list(noise.shape),list(times.shape),list(observations.shape)],
        replay_artifact_path=str(artifact),replay_artifact_sha256=sha(artifact),
        numerical_optimizer_updates=0,additional_generator_training_seeds=0)
    del h_hat,x_hat,z,conditioning,obs;gc.collect();torch.cuda.empty_cache()
    return report


def prediction_reachable_ops(outputs):
    stack=[v.op for v in outputs];seen=set()
    while stack:
        operation=stack.pop()
        if operation.name in seen:continue
        seen.add(operation.name)
        if any(name in operation.type.lower() for name in ['applyadam','applygradient','applymomentum','applyrmsprop']):
            raise ValueError('Native prediction graph reaches an optimizer update')
        stack.extend(value.op for value in operation.inputs);stack.extend(operation.control_inputs)
    return len(seen)


def replay_evaluator(evaluator,expected,source,folder,np,tf):
    role=evaluator['role'];function=evaluator['evaluator_function']
    expected_name='discriminative_score_metrics' if role=='discriminative' else expected['predictive_function']
    if (function!=expected_name or evaluator['optimizer_updates']!=expected[role+'_evaluator_steps']
            or evaluator['expected_optimizer_updates']!=expected[role+'_evaluator_steps']
            or evaluator['evaluator_training_instances']!=1 or evaluator['additional_generator_training_seeds']!=0):
        raise ValueError('Actual complete original evaluator training and function branch required')
    names=metric_output_contract(role,function,source)
    checkpoints=evaluator['checkpoint_files']
    paths={Path(v['path']).name:Path(v['path']) for v in checkpoints}
    if (len(checkpoints)!=3 or set(paths)!={'native_evaluator.meta','native_evaluator.index','native_evaluator.data-00000-of-00001'}
            or len({v.parent for v in paths.values()})!=1):
        raise ValueError('All original trained graph/model/optimizer checkpoint components required')
    for artifact in checkpoints:
        if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Trained native evaluator checkpoint changed')
    if sha(evaluator['array_path'])!=evaluator['array_sha256']:raise ValueError('Original full test feed/target/prediction changed')
    with np.load(evaluator['array_path'],allow_pickle=False) as saved:
        arrays={key:np.array(saved[key],copy=True) for key in saved.files}
    n=expected['full_generation_windows'];length=24;variables=expected['variables']
    if role=='discriminative':
        test_n=n-int(.8*n)
        shapes={'myinput_x:0':(test_n,length,variables),'myinput_x_hat:0':(test_n,length,variables),
            'myinput_t:0':(test_n,),'myinput_t_hat:0':(test_n,),
            'prediction_real':(test_n,1),'prediction_fake':(test_n,1)}
        reference=None;native=[arrays['prediction_real'],arrays['prediction_fake']]
    else:
        out=expected['predictive_output_variables']
        shapes={'myinput_x:0':(n,23,variables if out==variables else variables-1),'myinput_t:0':(n,),
            'prediction':(n,23,out),'original_reference':(n,23,out)}
        reference=arrays['original_reference'];native=[arrays['prediction']]
    if set(arrays)!=set(shapes) or any(arrays[key].shape!=shape or not np.isfinite(arrays[key]).all() for key,shape in shapes.items()):
        raise ValueError('All original full evaluator test feeds and targets/predictions required')
    graph=tf.Graph()
    with graph.as_default():
        saver=tf.compat.v1.train.import_meta_graph(str(paths['native_evaluator.meta']),clear_devices=True)
        variables_list=tf.compat.v1.global_variables()
    if len(variables_list)!=evaluator['trained_variable_count'] or not variables_list:
        raise ValueError('Every original model and optimizer variable required')
    outputs=[graph.get_tensor_by_name(name) for name in names]
    reachable=prediction_reachable_ops(outputs)
    feed={graph.get_tensor_by_name(key):arrays[key] for key in shapes if key.endswith(':0')}
    placement=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1,device_count={'GPU':0})
    with tf.compat.v1.Session(graph=graph,config=placement) as session:
        saver.restore(session,str(paths['native_evaluator.meta'].with_suffix('')))
        before_values=session.run(variables_list)
        if any(not np.isfinite(value).all() for value in before_values):raise ValueError('Native trained evaluator variables must be finite')
        before=tensor_digest({v.name:data for v,data in zip(variables_list,before_values)},np)
        predictions=session.run(outputs,feed_dict=feed)
        after=tensor_digest({v.name:data for v,data in zip(variables_list,session.run(variables_list))},np)
    for index,(actual,stored) in enumerate(zip(predictions,native)):
        exact_arrays(actual,stored,role+' complete prediction '+str(index),np)
    if before!=after:raise ValueError('Independent native evaluator inference changed model/optimizer state')
    score=recompute_score(role,predictions if role=='discriminative' else predictions[0],reference)
    if score!=evaluator['full_saved_predictions_recomputed_score']:
        raise ValueError('Every original saved test prediction/target score must agree exactly')
    if not np.isclose(score,evaluator['original_returned_score'],rtol=1e-7,atol=1e-7):
        raise ValueError('Native source returned score does not agree with full replay')
    artifact=folder/'full_native_prediction_replay.npz'
    folder.mkdir(parents=True,exist_ok=False)
    with artifact.open('xb') as stream:np.savez_compressed(stream,**{'prediction_'+str(i):v for i,v in enumerate(predictions)})
    return dict(role=role,evaluator_function=function,original_optimizer_updates=evaluator['optimizer_updates'],
        all_original_model_and_optimizer_variables_restored=len(variables_list),original_checkpoint_files=checkpoints,
        original_array_sha256=evaluator['array_sha256'],native_tensor_names=names,
        full_prediction_shapes=[list(v.shape) for v in predictions],full_predictions_bitwise_equal=True,
        full_original_test_feed_and_targets_preserved=True,score=score,native_source_returned_score=evaluator['original_returned_score'],
        native_model_and_optimizer_state_unchanged=True,trained_variable_sha256=before,
        prediction_dependency_operations=reachable,numerical_optimizer_updates=0,additional_generator_training_seeds=0,
        replay_artifact_path=str(artifact),replay_artifact_sha256=sha(artifact))


def check_report(config,route,binding,report,receipt):
    expected=budget(route)
    frozen=read(binding)
    if (frozen['route']!=route or frozen['configuration_sha256']!=sha(ROOT/config['runtime_configuration'])
            or frozen['result_sha256']!=sha(ROOT/route['result_path'])
            or frozen['native_receipt_sha256']!=sha(ROOT/route['native_completion_receipt'])
            or successful_native_receipt(route) is None):
        raise ValueError('Unchanged actual native completion receipt and full result binding required')
    native_result=original_result(config,route)
    if (receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('complete_full_native_replay_passed') is not True
            or report['canonical_job']!=route['canonical_job'] or report['binding_sha256']!=sha(binding)
            or report['configuration_sha256']!=sha(ROOT/config['runtime_configuration'])
            or report['original_native_result_sha256']!=frozen['result_sha256']
            or report['original_scope']!=config['original_scope']
            or report['native_release_checkpoint_evaluation'] is not expected['native_release_checkpoint_evaluation']
            or report['author_equivalence_certified'] is not False or report['strict_TAB_result'] is not False
            or report['all_original_native_generations_and_evaluators_replayed'] is not True
            or len(report['full_periodic_replays'])!=expected['full_periodic_evaluations']
            or report['numerical_optimizer_updates']!=0 or report['additional_generator_training_seeds']!=0):
        raise ValueError('Complete actual full-generation/trained-evaluator native replay required')
    import numpy as np
    for index,row in enumerate(report['full_periodic_replays'],1):
        generation=row['generation'];metrics=row['evaluators']
        native=native_result['evaluations'][index-1]
        step=0 if expected['native_release_checkpoint_evaluation'] else 500*index
        roles={'generator','recovery'} if expected['native_release_checkpoint_evaluation'] else {'embedder','generator','recovery','discriminator'}
        if (generation['full_generated_shape']!=route['full_generation_shape'] or generation['full_prediction_bitwise_equal'] is not True
                or row['native_joint_step']!=step or generation['native_joint_step']!=step
                or generation['complete_original_native_sampler_and_recovery_executed'] is not True
                or generation['model_parameters_unchanged'] is not True
                or {v['role'] for v in generation['full_native_network_state_loads']}!=roles
                or len(generation['full_native_network_state_loads'])!=len(roles)
                or any(v['strict_full_state_bitwise_loaded'] is not True for v in generation['full_native_network_state_loads'])
                or row['original_truth']['all_native_float32_truth_window_bytes_and_duplicate_counts_match'] is not True
                or len(metrics)!=20 or [m['role'] for m in metrics]!=['discriminative']*10+['predictive']*10):
            raise ValueError('All full native periodic generation and20 independent trained critic replays required')
        checkpoints=native_checkpoints(native,native_result,expected)
        if (generation['sampler_input']!=native['sampler_input']
                or generation['source_generation_artifact_sha256']!=native['artifact_sha256']
                or generation['original_sampler_function']!=('run_latent_ctfp_model5_3' if route['dataset']=='stock' else 'run_latent_ctfp_model5_2')
                or {v['role']:v['input_checkpoint'] for v in generation['full_native_network_state_loads']}!=checkpoints
                or row['original_truth']!=verify_native_truth(native,route,np)):
            raise ValueError('Every replay must bind to its exact native sampler, networks and original full truth')
        with np.load(generation['replay_artifact_path'],allow_pickle=False) as saved,np.load(native['artifact_path'],allow_pickle=False) as original:
            if set(saved.files)!={'generated'}:raise ValueError('Complete replay generation artifact required')
            exact_arrays(saved['generated'],original['generated'],'parent full generation review',np)
        for metric,evaluator in zip(metrics,native['evaluator_training_receipts']):
            role=metric['role'];n=expected['full_generation_windows']
            shapes=[[n-int(.8*n),1]]*2 if role=='discriminative' else [[n,23,expected['predictive_output_variables']]]
            if (metric['full_predictions_bitwise_equal'] is not True
                    or metric['full_original_test_feed_and_targets_preserved'] is not True
                    or metric['native_model_and_optimizer_state_unchanged'] is not True
                    or metric['all_original_model_and_optimizer_variables_restored']<1
                    or metric['original_optimizer_updates']!=expected[role+'_evaluator_steps']
                    or metric['full_prediction_shapes']!=shapes
                    or metric['evaluator_function']!=evaluator['evaluator_function']
                    or metric['original_checkpoint_files']!=evaluator['checkpoint_files']
                    or metric['original_array_sha256']!=evaluator['array_sha256']
                    or metric['all_original_model_and_optimizer_variables_restored']!=evaluator['trained_variable_count']
                    or metric['score']!=evaluator['full_saved_predictions_recomputed_score']
                    or metric['native_source_returned_score']!=evaluator['original_returned_score']):
                raise ValueError('Every trained original evaluator/full test prediction/model-optimizer state required')
            keys=['prediction_real','prediction_fake'] if role=='discriminative' else ['prediction']
            with np.load(metric['replay_artifact_path'],allow_pickle=False) as saved,np.load(evaluator['array_path'],allow_pickle=False) as original:
                if set(saved.files)!={'prediction_'+str(i) for i in range(len(keys))}:
                    raise ValueError('Every complete original evaluator prediction array required')
                for i,key in enumerate(keys):exact_arrays(saved['prediction_'+str(i)],original[key],'parent trained evaluator review',np)
        for artifact in [generation]+metrics:
            if (artifact['numerical_optimizer_updates']!=0 or artifact['additional_generator_training_seeds']!=0
                    or sha(artifact['replay_artifact_path'])!=artifact['replay_artifact_sha256']):
                raise ValueError('Read-only complete native replay artifact changed')
    return report


def worker(config,route,binding_path):
    binding=read(binding_path)
    if (binding['route']!=route or binding['configuration_sha256']!=sha(ROOT/config['runtime_configuration'])
            or binding['result_sha256']!=sha(ROOT/route['result_path'])
            or binding['native_receipt_sha256']!=sha(ROOT/route['native_completion_receipt'])
            or successful_native_receipt(route) is None):
        raise ValueError('Exact complete original native result and successful guarded receipt required')
    queue,model,case=queue_and_case(route);verify_author_environment(queue)
    result=original_result(config,route);expected=budget(route)
    data=read(ROOT/model['dataset_config'])
    if sha(ROOT/data['raw_file'])!=data['raw_sha256']:raise ValueError('Complete original raw data changed')
    folder=ROOT/config['output_root']/route['canonical_job']['id']
    if folder.exists():raise FileExistsError('Preserve every full native replay attempt')
    folder.mkdir(parents=True)
    import numpy as np
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.compat.v1.disable_eager_execution();torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('Unchanged full native GT-GAN sampler requires CUDA')
    module=import_native_runtime(route,case)
    if route['dataset']=='energy':energy_mode_contract((ROOT/case['source_entry']).read_text(encoding='utf-8'))
    namespace=construct_native_models(module,case,expected,torch)
    original_counts=verify_native_model_counts(queue,case,namespace)
    reports=[]
    for index,row in enumerate(result['evaluations']):
        if sha(row['artifact_path'])!=row['artifact_sha256']:raise ValueError('Full original generation changed')
        original_truth=verify_native_truth(row,route,np)
        generation=replay_generation(row,result,expected,namespace,folder,np,torch)
        metrics=[]
        for repetition,evaluator in enumerate(row['evaluator_training_receipts']):
            source_file='discriminative_metrics.py' if evaluator['role']=='discriminative' else 'predictive_metrics.py'
            source=(ROOT/model['source_root']/'metrics'/source_file).read_text(encoding='utf-8')
            metrics.append(replay_evaluator(evaluator,expected,source,folder/('period_'+str(index+1))/('critic_'+str(repetition+1)),np,tf))
        reports.append(dict(native_joint_step=row['native_joint_step'],original_truth=original_truth,generation=generation,evaluators=metrics))
        write(folder/'progress.json',dict(phase='full_native_readonly_replay',full_periodic_evaluations_completed=len(reports),
            expected_full_periodic_evaluations=expected['full_periodic_evaluations'],full_trained_evaluators_replayed=20*len(reports),
            numerical_optimizer_updates=0,all_original_experiments_complete=False))
        gc.collect()
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),canonical_job=route['canonical_job'],
        configuration_sha256=sha(ROOT/config['runtime_configuration']),binding_sha256=sha(binding_path),
        original_native_result_sha256=binding['result_sha256'],full_periodic_replays=reports,
        all_original_native_generations_and_evaluators_replayed=True,numerical_optimizer_updates=0,
        additional_generator_training_seeds=0,native_release_checkpoint_evaluation=expected['native_release_checkpoint_evaluation'],
        original_raw_sha256=data['raw_sha256'],original_scope=config['original_scope'],
        full_original_native_constructor_parameter_counts=original_counts,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
    exclusive(folder/'complete_full_native_replay.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--worker-probe-only',action='store_true');parser.add_argument('--canonical-id');parser.add_argument('--binding')
    args=parser.parse_args();config=read(ROOT/args.config)
    if config['runtime_configuration']!=args.config:raise ValueError('Exact frozen all-eight replay configuration required')
    routes=verify(config)
    if args.worker_probe_only:
        queue=read(ROOT/config['original_queues']['stock']);verify_author_environment(queue)
        compiled_energy_verifier()
        from scripts.flow_matching.audit_gtgan_native_models_v1 import original_gtgan_branch
        for route in routes:
            _,model,case=queue_and_case(route)
            compile(original_gtgan_branch((ROOT/case['source_entry']).read_text(encoding='utf-8')),'<complete-original-constructor>','exec')
            if route['dataset']=='energy':energy_mode_contract((ROOT/case['source_entry']).read_text(encoding='utf-8'))
            source=ROOT/model['source_root']/'metrics'
            for role,filename,name in [('discriminative','discriminative_metrics.py','discriminative_score_metrics'),
                    ('predictive','predictive_metrics.py',budget(route)['predictive_function'])]:
                metric_output_contract(role,name,(source/filename).read_text(encoding='utf-8'))
        print(json.dumps(dict(passed=True,all_original_commands=8,all_full_native_generations=60,all_full_trained_evaluators=1200,
            exact_original_environment_verified=True,numerical_training_executed=False,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,tensorflow_imported='tensorflow' in sys.modules,
            psutil_imported='psutil' in sys.modules,
            all_original_experiments_complete=False)));return
    if args.worker:
        route=next(r for r in routes if r['canonical_job']['id']==args.canonical_id)
        worker(config,route,args.binding);return
    import psutil
    from scripts.flow_matching.run_spectral_original_queue import guarded_api
    from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
    api=guarded_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Complete all-eight native replay queue already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),configuration_sha256=sha(ROOT/args.config),
        state='starting_all_eight_full_native_replays',verified_original_ids=[],pending_original_commands=8,failed={},active_pid=None,
        required_full_native_generations=60,required_full_trained_evaluators=1200,all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while len(state['verified_original_ids'])<8:
            for route in routes:
                identity=route['canonical_job']['id']
                if identity in state['verified_original_ids'] or identity in state['failed']:continue
                if not (ROOT/route['result_path']).exists() or successful_native_receipt(route) is None:continue
                try:
                    binding=dict(route=route,configuration_sha256=sha(ROOT/args.config),
                        result_sha256=sha(ROOT/route['result_path']),native_receipt_sha256=sha(ROOT/route['native_completion_receipt']))
                    binding_path=base/'bindings'/(identity+'.json');exclusive(binding_path,binding)
                    argv=['complete_native_gtgan_replay','--config',args.config,'--worker','--canonical-id',identity,'--binding',str(binding_path)]
                    bootstrap=('import sys,runpy;sys.path.insert(0,'+repr(str(ROOT))+');sys.argv='+repr(argv)+';'
                        "runpy.run_module('scripts.flow_matching.run_gtgan_all8_full_native_replay_v1',run_name='__main__')")
                    command=[str(ROOT/config['python']),'-X','utf8','-u','-c',bootstrap]
                    with full_gpu_memory_slot(api,config,update):
                        logs=base/'logs';logs.mkdir(exist_ok=True)
                        with (logs/(identity+'.stdout.log')).open('xb') as out,(logs/(identity+'.stderr.log')).open('xb') as err:
                            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',MPLBACKEND='Agg',
                                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='2'),
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            update(dict(state='running_all_full_original_native_generations_and_trained_evaluators',
                                active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),active_job=identity,actual_command=command))
                            code,usage=api['guarded_gpu_wait'](child,config['settings'],api,update)
                    report_path=ROOT/config['output_root']/identity/'complete_full_native_replay.json'
                    passed=code==0 and usage['resource_guard_aborted'] is False and report_path.exists()
                    receipt=dict(exit_code=code,complete_full_native_replay_passed=passed,**usage)
                    write(base/'receipts'/(identity+'.json'),receipt)
                    if not passed:raise ValueError('Complete native generation/evaluator replay failed; preserve full attempt')
                    check_report(config,route,binding_path,read(report_path),receipt)
                    state['verified_original_ids'].append(identity)
                except Exception as error:
                    state['failed'][identity]=str(error)
                    exclusive(base/'failures'/(identity+'.json'),dict(route=route,error=str(error),all_original_experiments_complete=False))
                update(dict(active_pid=None,active_job=None,pending_original_commands=8-len(state['verified_original_ids'])))
                time.sleep(config['settings']['controller_yield_seconds'])
            owners=[]
            for expected in config['expected_original_native_owners']:
                try:
                    p=psutil.Process(expected['pid'])
                    if p.create_time()==expected['create_time'] and p.cmdline()==expected['command']:owners.append(dict(expected,live_now=True))
                except psutil.NoSuchProcess:pass
            update(dict(state='waiting_actual_complete_all_eight_original_native_commands',actual_verified_live_original_owners=owners))
            if len(state['verified_original_ids'])<8:time.sleep(config['poll_seconds'])
        update(dict(state='all_eight_full_native_generation_and_trained_evaluator_replays_verified',all8_GTGAN_native_replays_complete=True))
    finally:owner.close()


if __name__=='__main__':main()
