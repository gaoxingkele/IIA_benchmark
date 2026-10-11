"""Observe every complete original LS4 patient evaluation without extra model calls."""
import ast
from datetime import datetime,timezone
import functools
import json
from pathlib import Path
import random

from scripts.flow_matching.run_ls4_native_physionet_v1 import ROOT,read,sha,budget,periodic_epochs


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def backend_flags(torch):
    return dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled())


def encode(value,arrays,prefix,np,torch):
    """Copy every original tensor input/output; retain tuple/list/dict/None structure."""
    if torch.is_tensor(value):
        array=value.detach().cpu().numpy().copy()
        arrays[prefix]=array
        return dict(kind='tensor',key=prefix,shape=list(array.shape),dtype=str(array.dtype),device=str(value.device))
    if isinstance(value,(list,tuple)):
        return dict(kind='tuple' if isinstance(value,tuple) else 'list',items=[encode(v,arrays,prefix+'.'+str(i),np,torch) for i,v in enumerate(value)])
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):raise ValueError('Original method argument dictionary keys must be strings')
        return dict(kind='dict',items={k:encode(v,arrays,prefix+'.'+k,np,torch) for k,v in value.items()})
    if value is None or type(value) in (str,bool,int):return dict(kind='scalar',value=value)
    if type(value) is float and np.isfinite(value):return dict(kind='scalar',value=value)
    raise ValueError('Unsupported original full method input/output; never silently omit it')


class FullEvaluationObservation:
    def __init__(self,configuration,queue,job,config,artifact,model,loader,optimizer,scheduler,
            epoch,index,best_mse,classify,extrap,updates,np,torch):
        self.np,self.torch,self.model,self.loader=np,torch,model,loader
        self.target=getattr(model,'module',model);self.originals={};self.calls=[]
        self.config=config;self.queue=queue;self.job=job
        expected=budget(config)
        if (len(loader.dataset)!=1600 or len(loader)!=expected['test_batches']
                or loader.batch_size!=config['batch_size'] or loader.num_workers!=0 or loader.drop_last):
            raise ValueError('Exact full original1600-patient loader and batch size required')
        self.folder=Path(artifact)/'full_native_evaluation_contexts'/('evaluation_'+str(index+1).zfill(3))
        self.folder.mkdir(parents=True,exist_ok=False)
        generators={}
        for name,owner in [('loader',loader),('sampler',loader.sampler)]:
            generator=getattr(owner,'generator',None)
            if generator is not None:generators[name]=generator.get_state()
        snapshot=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            torch_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_generators=generators,
            model_training_mode=model.training,grad_enabled=torch.is_grad_enabled(),backend_flags=backend_flags(torch),
            native_epoch=epoch,evaluation_index=index,best_mse=best_mse,classify=classify,extrap=extrap,
            optimizer_updates=updates,job=job,config_sha256=sha(ROOT/job['model_config']),
            queue_sha256=sha(ROOT/queue['queue_path']),observation_configuration_sha256=sha(ROOT/configuration))
        path=self.folder/'complete_original_evaluation_input.pt'
        with path.open('xb') as stream:torch.save(snapshot,stream)
        self.context=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,native_epoch=epoch,
            evaluation_index=index,full_test_patients=1600,full_test_batches=len(loader),original_batch_size=loader.batch_size,
            original_model_training_mode=model.training,original_grad_enabled=snapshot['grad_enabled'],
            original_backend_flags=snapshot['backend_flags'],complete_original_model_optimizer_scheduler_and_rng_saved=True,
            original_loader_generator_states_saved=sorted(generators),numerical_optimizer_updates_from_observation=0,
            additional_training_seed_slots=0,full_method_calls=[],full_observation_complete=False)
        exclusive(self.folder/'input_context.json',self.context)

    def __enter__(self):
        for name in ['reconstruct','full_nll']:
            original=getattr(self.target,name)
            self.originals[name]=(original,name in self.target.__dict__)
            def wrapped(*args,_name=name,_original=original,**kwargs):
                arrays={}
                inputs=encode(dict(args=args,kwargs=kwargs),arrays,'input',self.np,self.torch)
                value=_original(*args,**kwargs)
                outputs=encode(value,arrays,'output',self.np,self.torch)
                if any(not self.np.isfinite(v).all() for key,v in arrays.items() if key.startswith('output')):
                    raise ValueError('Full native evaluation method output must be finite')
                path=self.folder/('method_'+str(len(self.calls)+1).zfill(3)+'_'+_name+'.npz')
                with path.open('xb') as stream:self.np.savez_compressed(stream,**arrays)
                self.calls.append(dict(method=_name,input_structure=inputs,output_structure=outputs,
                    array_path=str(path),array_sha256=sha(path),array_shapes={k:list(v.shape) for k,v in arrays.items()},
                    every_original_input_and_output_tensor_saved=True,original_model_call_count=1,
                    numerical_optimizer_updates_from_observation=0,additional_training_seed_slots=0))
                return value
            setattr(self.target,name,functools.wraps(original)(wrapped))
        return self

    def __exit__(self,kind,error,traceback):
        for name,(original,owned) in self.originals.items():
            if owned:setattr(self.target,name,original)
            else:delattr(self.target,name)
        if kind is not None:
            exclusive(self.folder/'failed_observation.json',dict(error=str(error),full_method_calls=self.calls,
                full_observation_complete=False,all_original_experiments_complete=False))
        return False

    def finish(self):
        expected=budget(self.config)
        if [row['method'] for row in self.calls]!=['reconstruct','full_nll']*expected['test_batches']:
            raise ValueError('Every full original reconstruction/NLL call per test batch required')
        self.context.update(full_method_calls=self.calls,full_observation_complete=True)
        exclusive(self.folder/'complete_original_evaluation_observation.json',self.context)
        return self.context


def worker_ast(source):
    """Compile the whole original worker, adding only complete evaluation receipts."""
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_job']
    if len(functions)!=1:raise ValueError('One complete original native run_job function required')
    worker=functions[0]
    evaluate=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    call=ast.parse('value = original_eval(model, device, epoch, observed, best_mse, log_dir, classify=classify, extrap=extrap)').body[0]
    matches=[i for i,n in enumerate(evaluate.body) if ast.dump(n)==ast.dump(call)]
    if len(matches)!=1:raise ValueError('One unchanged original full native evaluation invocation required')
    index=matches[0]
    additions=ast.parse("observation = FullEvaluationObservation(observation_configuration,queue,job,config,artifact,model,loader,optimizer_ref[0],scheduler_ref[0],epoch,len(evaluations),best_mse,classify,extrap,updates,np,torch)\nwith observation:\n    value = original_eval(model, device, epoch, observed, best_mse, log_dir, classify=classify, extrap=extrap)").body
    evaluate.body[index:index+1]=additions
    marker=ast.parse('write(output / "evaluation_observation.json", dict(evaluations=evaluations))').body[0]
    matches=[i for i,n in enumerate(evaluate.body) if ast.dump(n)==ast.dump(marker)]
    if len(matches)!=1:raise ValueError('One original complete evaluation-row write required')
    evaluate.body.insert(matches[0],ast.parse('evaluations[-1]["full_native_evaluation_observation"] = observation.finish()').body[0])
    marker=ast.parse('write(output / "result.json", result)').body[0]
    matches=[i for i,n in enumerate(worker.body) if ast.dump(n)==ast.dump(marker)]
    if len(matches)!=1:raise ValueError('One complete original full native result write required')
    worker.body.insert(matches[0],ast.parse('result["full_native_evaluation_observation_configuration_sha256"] = sha(ROOT/observation_configuration)').body[0])
    return ast.fix_missing_locations(ast.Module(body=[worker],type_ignores=[]))


def verify_extended_result(queue,job,configuration):
    from scripts.flow_matching.run_ls4_native_physionet_queue_v3 import verify_result
    result=verify_result(queue,job);config=read(ROOT/job['model_config']);expected=budget(config)
    if result['full_native_evaluation_observation_configuration_sha256']!=sha(ROOT/configuration):
        raise ValueError('Exact frozen full evaluation observation configuration required')
    for index,row in enumerate(result['evaluations']):
        context=row['full_native_evaluation_observation']
        if (context['native_epoch']!=row['epoch'] or context['evaluation_index']!=index
                or context['full_test_patients']!=1600 or context['full_test_batches']!=expected['test_batches']
                or context['original_batch_size']!=config['batch_size'] or not context['full_observation_complete']
                or not context['complete_original_model_optimizer_scheduler_and_rng_saved']
                or sha(context['path'])!=context['sha256']
                or context['numerical_optimizer_updates_from_observation']!=0 or context['additional_training_seed_slots']!=0
                or [call['method'] for call in context['full_method_calls']]!=['reconstruct','full_nll']*expected['test_batches']):
            raise ValueError('All50 complete native evaluation contexts and every original method call required')
        for call in context['full_method_calls']:
            if (not call['every_original_input_and_output_tensor_saved'] or call['original_model_call_count']!=1
                    or call['numerical_optimizer_updates_from_observation']!=0 or call['additional_training_seed_slots']!=0
                    or sha(call['array_path'])!=call['array_sha256']):
                raise ValueError('Every complete original evaluation method tensor receipt required')
    return result
