"""Replay complete original LS4 evaluations with restored model/optimizer/RNG state."""
import ast
import functools
import json
from pathlib import Path
import random
import struct

from scripts.flow_matching.ls4_physio_full_evaluation_observation_v1 import (
    ROOT,read,sha,budget,periodic_epochs,encode,backend_flags,exclusive,
)


def constructor_ast(source):
    """Keep the entire author main prefix before its optimization loop."""
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main']
    if len(functions)!=1:raise ValueError('One complete original LS4 main required')
    main=functions[0]
    boundary=ast.parse('not args.test',mode='eval').body
    positions=[i for i,n in enumerate(main.body) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(boundary)]
    if len(positions)!=1:raise ValueError('Exactly one original optimization-loop boundary required')
    prefix=main.body[:positions[0]]
    for statement in ['model = VAE(config.model)','model = model.to(device)',
            'valloader = data_objs["test_dataloader"]']:
        wanted=ast.parse(statement).body[0]
        if sum(ast.dump(n)==ast.dump(wanted) for n in prefix)!=1:
            raise ValueError('Complete original model/device/test-loader construction required')
    main.body=prefix+[ast.parse('return model, valloader, optimizer, scheduler').body[0]]
    return ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))


def bitwise_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def output_schema(value):
    if isinstance(value,dict):
        if value.get('kind')=='scalar' and type(value.get('value')) in (float,int):
            return dict(kind='scalar',numeric_type=type(value['value']).__name__)
        return {k:output_schema(v) for k,v in value.items()}
    if isinstance(value,list):return [output_schema(v) for v in value]
    return value


def descriptor_equal(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(descriptor_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(descriptor_equal(x,y) for x,y in zip(a,b))
    if type(a) is float:return struct.pack('d',a)==struct.pack('d',b)
    return a==b


def tensor_descriptors(value):
    result={}
    if isinstance(value,dict):
        if value.get('kind')=='tensor':result[value['key']]=value
        else:
            for child in value.values():result.update(tensor_descriptors(child))
    elif isinstance(value,list):
        for child in value:result.update(tensor_descriptors(child))
    return result


def compare_call(expected,name,actual_inputs,actual_outputs,actual_arrays,np):
    if (name!=expected['method'] or not descriptor_equal(actual_inputs,expected['input_structure'])
            or output_schema(actual_outputs)!=output_schema(expected['output_structure'])):
        raise ValueError('Exact full native method order and complete input/output structures required')
    if sha(expected['array_path'])!=expected['array_sha256']:
        raise ValueError('Original complete method tensor archive changed')
    diagnostics={};scalar_match=descriptor_equal(actual_outputs,expected['output_structure']);matches=scalar_match
    with np.load(expected['array_path'],allow_pickle=False) as saved:
        if set(saved.files)!=set(actual_arrays) or set(saved.files)!=set(expected['array_shapes']):
            raise ValueError('Every original method tensor must be compared')
        for key in saved.files:
            reference,actual=saved[key],actual_arrays[key]
            if (list(actual.shape)!=expected['array_shapes'][key] or actual.shape!=reference.shape or actual.dtype!=reference.dtype):
                raise ValueError('Complete native tensor shape and dtype must match: '+key)
            agrees=bitwise_equal(reference,actual)
            if key.startswith('input'):
                if not agrees:raise ValueError('Fresh original full-loader method input differs: '+key)
            else:
                if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                    raise ValueError('Complete finite original and replay model output required')
                matches=matches and agrees
            kind=np.complex128 if np.iscomplexobj(actual) else np.float64
            delta=float(np.max(np.abs(actual.astype(kind)-reference.astype(kind)))) if actual.size else 0.
            diagnostics[key]=dict(shape=list(actual.shape),dtype=str(actual.dtype),bitwise_equal=agrees,maximum_absolute_difference=delta)
    return dict(method=name,all_complete_original_tensor_inputs_equal=True,all_complete_outputs_bitwise_equal=bool(matches),
        original_complete_output_structure=expected['output_structure'],replayed_complete_output_structure=actual_outputs,
        output_structure_and_scalars_bitwise_equal=scalar_match,
        full_tensor_comparisons=diagnostics,original_model_call_count=1,additional_optimizer_updates=0)


class FullCallReview:
    def __init__(self,model,context,np,torch):
        self.target=getattr(model,'module',model);self.context=context;self.np=np;self.torch=torch
        self.originals={};self.calls=[]
    def __enter__(self):
        for name in ['reconstruct','full_nll']:
            original=getattr(self.target,name);self.originals[name]=(original,name in self.target.__dict__)
            def call(*args,_name=name,_original=original,**kwargs):
                index=len(self.calls)
                if index>=len(self.context['full_method_calls']):raise ValueError('Unexpected extra original model call')
                expected=self.context['full_method_calls'][index]
                if expected['method']!=_name:raise ValueError('Original reconstruction/NLL call order differs')
                arrays={};inputs=encode(dict(args=args,kwargs=kwargs),arrays,'input',self.np,self.torch)
                value=_original(*args,**kwargs)
                outputs=encode(value,arrays,'output',self.np,self.torch)
                self.calls.append(compare_call(expected,_name,inputs,outputs,arrays,self.np))
                return value
            setattr(self.target,name,functools.wraps(original)(call))
        return self
    def __exit__(self,kind,error,traceback):
        for name,(original,owned) in self.originals.items():
            if owned:setattr(self.target,name,original)
            else:delattr(self.target,name)
        return False
    def finish(self):
        if len(self.calls)!=len(self.context['full_method_calls']):raise ValueError('All original full method calls required')
        return dict(full_original_method_calls=len(self.calls),full_original_method_inputs_and_outputs_compared=True,
            all_original_method_outputs_bitwise_equal=all(r['all_complete_outputs_bitwise_equal'] for r in self.calls),calls=self.calls)


def restore_snapshot(snapshot,queue,job,config,row,index,model,loader,optimizer,scheduler,np,torch,observation):
    expected=budget(config)
    updates=(row['epoch']+1)*expected['train_batches'] if row['epoch']>=0 else expected['optimizer_updates']
    if (snapshot['job']!=job or snapshot['config_sha256']!=sha(ROOT/job['model_config'])
            or snapshot['queue_sha256']!=sha(ROOT/queue['queue_path'])
            or snapshot['observation_configuration_sha256']!=sha(ROOT/observation)
            or snapshot['native_epoch']!=row['epoch'] or snapshot['evaluation_index']!=index
            or snapshot['optimizer_updates']!=updates or bool(snapshot['classify'])
            or bool(snapshot['extrap'])!=config['extrapolation']
            or len(loader.dataset)!=1600 or len(loader)!=expected['test_batches'] or loader.batch_size!=config['batch_size']
            or loader.num_workers!=0 or loader.drop_last):
        raise ValueError('Every complete original evaluation identity, optimizer budget and full loader required')
    model.load_state_dict(snapshot['model'],strict=True)
    for key,value in model.state_dict().items():
        original=snapshot['model'][key]
        if value.dtype!=original.dtype or value.shape!=original.shape or not torch.equal(value.detach().cpu(),original.detach().cpu()):
            raise ValueError('Every original saved model parameter/buffer must load exactly')
    optimizer.load_state_dict(snapshot['optimizer']);scheduler.load_state_dict(snapshot['scheduler'])
    model.train(snapshot['model_training_mode'])
    flags=snapshot['backend_flags']
    torch.backends.cudnn.benchmark=flags['cudnn_benchmark'];torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32'];torch.backends.cuda.matmul.allow_tf32=flags['cuda_matmul_allow_tf32']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms_enabled'])
    if backend_flags(torch)!=flags:raise ValueError('All captured original numerical backend flags required')
    generators={}
    for name,owner in [('loader',loader),('sampler',loader.sampler)]:
        generator=getattr(owner,'generator',None)
        if generator is not None:generators[name]=generator
    if set(generators)!=set(snapshot['loader_generators']):raise ValueError('Exact original loader-generator bindings required')
    for name,generator in generators.items():generator.set_state(snapshot['loader_generators'][name])
    np.random.set_state(snapshot['numpy_rng']);random.setstate(snapshot['python_rng'])
    torch.random.set_rng_state(snapshot['torch_rng'])
    if len(snapshot['cuda_rng'])!=torch.cuda.device_count():raise ValueError('All original CUDA random generators required')
    torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
    if not torch.equal(torch.random.get_rng_state(),snapshot['torch_rng']):raise ValueError('Original CPU random state differs')
    if any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),snapshot['cuda_rng'])):
        raise ValueError('Original CUDA random state differs')


def replay_job(configuration,path,job,binding):
    from scripts.flow_matching.run_ls4_native_physionet_v1 import verify,create_runtime
    from scripts.flow_matching.ls4_physio_full_evaluation_observation_v1 import verify_extended_result
    queue=read(ROOT/configuration['original_queue']);verify(queue)
    receipt_path=ROOT/configuration['training_state_root']/'receipts'/(job['id']+'.json')
    if sha(receipt_path)!=binding['training_receipt_sha256']:raise ValueError('Original terminal training receipt changed')
    result=verify_extended_result(queue,job,configuration['observation_configuration'])
    result_path=ROOT/job['output_directory']/'result.json'
    if sha(result_path)!=binding['original_result_sha256']:raise ValueError('Original full training/evaluation result changed')
    config=read(ROOT/job['model_config']);artifact=Path(configuration['artifact_root'])/job['id']
    torch,np,native,yaml,resolved,correction,logs,data=create_runtime(queue,config,artifact)
    if not torch.cuda.is_available() or yaml.optim.get('use_ema',False):
        raise ValueError('Exact original CUDA non-EMA PhysioNet profiles required')
    namespace=dict(vars(native))
    exec(compile(constructor_ast(Path(native.__file__).read_text(encoding='utf-8')),native.__file__,'exec'),namespace)
    args=native.get_parse_args().parse_args(['--config',str(resolved),'--seed',str(config['seed']),'--log_dir',str(artifact/'native_outputs')])
    if args.resume:raise ValueError('Fresh independent original constructor before full saved-state restoration required')
    model,loader,optimizer,scheduler=namespace['main'](args)
    if not isinstance(model,torch.nn.DataParallel) or sum(p.numel() for p in model.parameters())!=result['parameters']:
        raise ValueError('Full original CUDA DataParallel architecture required')
    def no_optimizer_update(*a,**kw):raise RuntimeError('Independent original evaluation replay must never optimize')
    optimizer.step=no_optimizer_update
    records=[]
    for index,row in enumerate(result['evaluations']):
        context=row['full_native_evaluation_observation']
        if sha(context['path'])!=context['sha256']:raise ValueError('Complete original evaluation snapshot changed')
        snapshot=torch.load(context['path'],map_location='cpu')
        restore_snapshot(snapshot,queue,job,config,row,index,model,loader,optimizer,scheduler,np,torch,configuration['observation_configuration'])
        start=len(logs);review=FullCallReview(model,context,np,torch)
        directory=artifact/('evaluation_'+str(index+1).zfill(3));directory.mkdir(parents=True,exist_ok=False)
        with torch.set_grad_enabled(snapshot['grad_enabled']):
            with review:
                native.eval(model,'cuda',snapshot['native_epoch'],loader,snapshot['best_mse'],str(directory),
                    classify=snapshot['classify'],extrap=snapshot['extrap'])
        metrics=[r for r in logs[start:] if 'eval_mse' in r]
        if len(metrics)!=1:raise ValueError('One complete original native MSE/CRPS/NLL log required')
        observed=metrics[0];expected={k:row[k] for k in observed}
        record=dict(evaluation_index=index,native_epoch=row['epoch'],original_snapshot_sha256=context['sha256'],
            all_original_model_optimizer_scheduler_rng_and_backend_flags_restored=True,
            original_full_test_patients=1600,original_full_test_batches=len(loader),additional_optimizer_updates=0,
            original_metrics=expected,replayed_metrics=observed,all_original_metrics_exactly_equal=observed==expected,**review.finish())
        exclusive(directory/'complete_original_evaluation_replay.json',record);records.append(record)
        del snapshot
    if len(records)!=50 or sum(r['full_original_method_calls'] for r in records)!=50*2*budget(config)['test_batches']:
        raise ValueError('All fifty original native full evaluations and every model call required')
    strict=all(r['all_original_method_outputs_bitwise_equal'] and r['all_original_metrics_exactly_equal'] for r in records)
    target=ROOT/configuration['proof_root']/job['id']/'complete_original_full_evaluation_replay.json'
    report=dict(job=job,configuration=path,configuration_sha256=sha(ROOT/path),original_result_sha256=binding['original_result_sha256'],
        training_receipt_sha256=binding['training_receipt_sha256'],full_native_evaluations_executed=50,
        all_original_full_model_calls_executed=sum(r['full_original_method_calls'] for r in records),records=records,
        all_full_original_model_inputs_and_outputs_compared=True,strict_full_saved_outputs_and_metrics_match=strict,
        additional_training_seed_slots=0,additional_optimizer_updates=0,original_scope=configuration['original_scope'],
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary='All fifty complete original native evaluations, fresh original constructor/data loader, exact saved model/optimizer/scheduler/random/backend state, and every original reconstruction/NLL tensor. Numerical output mismatches remain recorded across the complete job and are never accepted as strict reproduction. No new training, reduced batch, shorter grid, native-code edit, paper or TAB equivalence claim.')
    exclusive(target,report)
    return report
