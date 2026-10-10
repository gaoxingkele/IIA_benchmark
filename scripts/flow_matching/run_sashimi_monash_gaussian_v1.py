"""Run full Monash Gaussian SaShiMi reconstructions, with original evaluators."""
import argparse
from datetime import datetime,timezone
import inspect
import importlib
import json
import math
from pathlib import Path
import random
import time

from scripts.flow_matching.run_ls4_monash_extended_v2 import (
    ROOT,read,sha,write as native_write,workspace,source_imports as native_imports,checked_data,original_module)
from scripts.flow_matching.sashimi_gaussian_adapter_v1 import backbone_parameters,make_model
from scripts.flow_matching.ls4_cauchy_fallback_repair_v1 import install

CORRECTION_RECEIPT={}


def source_imports(target,queue):
    value=native_imports(target)
    CORRECTION_RECEIPT.update(install(importlib.import_module('models.s4'),
        read(ROOT/queue['cauchy_correction_config']),ROOT))
    return value


def write(path,value):
    return native_write(path,dict(value,backend_correction=dict(CORRECTION_RECEIPT)))


def budget(config):
    batches=math.ceil(config['train_samples']/64)
    return dict(batches=batches,generator_updates=batches*config['optim']['epochs'],
        classifier_updates=100*math.ceil(int(2*config['test_samples']*.8)/128),
        predictor_updates=100*math.ceil(config['test_samples']/128),
        classifier_test_batches=math.ceil((2*config['test_samples']-int(2*config['test_samples']*.8))/128),
        predictor_test_batches=math.ceil(config['test_samples']/128),
        evaluation_recipes=['last_model','last_ema'])


def test_batch_summary(records,expected_batches):
    if len(records)!=expected_batches or any(r['elements']<=0 or not math.isfinite(r['mean']) for r in records):
        raise ValueError('Full native test-batch loss observations incomplete')
    return dict(batches=len(records),elements=sum(r['elements'] for r in records),
        native_sum_batch_means=sum(r['mean'] for r in records),
        unweighted_mean_batch_means=sum(r['mean'] for r in records)/len(records),
        element_weighted_mean=sum(r['mean']*r['elements'] for r in records)/sum(r['elements'] for r in records),
        records=records)


def verify(queue):
    for item in queue['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen SaShiMi resource changed: '+item['path'])
    identities=set()
    for job in queue['jobs']:
        config=read(ROOT/job['model_config'])
        if sha(ROOT/job['model_config'])!=job['model_config_sha256']:raise ValueError('SaShiMi frozen model differs')
        key=(config['dataset'],config['profile'],config['seed'])
        if key in identities:raise ValueError('Duplicate SaShiMi seed slot')
        identities.add(key)
        if config['backbone']!=backbone_parameters(config['profile']):raise ValueError('Full SaShiMi backbone changed')
        expected=1000 if config['dataset']=='temperature_rain' else 7000
        optim=config['optim']
        if (optim['epochs']!=expected or optim['batch_size']!=64 or optim['lr']!=.001 or
            optim['weight_decay']!=0 or optim['lamb']!=.999 or optim['start_step']!=200 or config['sigma']!=.1):
            raise ValueError('Full reconstructed paper budget changed')
        if config['budget']!=budget(config) or config['diagnostic'] or config['author_equivalence_certified']:
            raise ValueError('SaShiMi full budget or fidelity status changed')
        data=queue['datasets'][config['dataset']]
        if (config['samples'],config['length'])!=(data['samples'],data['length']):raise ValueError('Full SaShiMi dataset changed')
        if config['train_samples']!=int(config['samples']*.8) or config['test_samples']!=config['samples']-config['train_samples']:
            raise ValueError('Full SaShiMi trajectory split changed')
        if config['activation']!=('sigmoid' if config['dataset']=='temperature_rain' else 'identity'):
            raise ValueError('Original full Monash output activation changed')
        if config['source_config']!=queue['source_root']+'/configs/monash/'+data['yaml']:
            raise ValueError('Original full Monash data configuration changed')
    expected={(d,p,s) for d in queue['datasets'] for p in queue['profiles'] for s in queue['seeds']}
    if identities!=expected or len(identities)!=40:raise ValueError('All40 SaShiMi reconstructions required')


def data_for(queue,config,target,torch,np,datasets):
    from omegaconf import OmegaConf
    args=OmegaConf.load(ROOT/config['source_config']).data
    args.path=str(target/'author_data'/config['dataset'])
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    return checked_data(config,datasets,torch,np,args)


def audit_data(queue):
    target=workspace(queue,queue['data_audit_artifact']);torch,np,datasets=source_imports(target,queue)
    torch.set_num_threads(queue['settings']['cpu_threads']);records=[]
    for job in queue['jobs']:
        config=read(ROOT/job['model_config']);data,record=data_for(queue,config,target,torch,np,datasets)
        records.append(dict(record,id=job['id'],profile=config['profile']))
        write(ROOT/queue['data_audit_progress'],dict(queue_sha256=sha(ROOT/queue['queue_path']),
            cases_completed=len(records),required_cases=40,records=records))
        del data
        import gc
        gc.collect()
    if torch.cuda.is_initialized():raise ValueError('CPU SaShiMi data audit initialized CUDA')
    write(ROOT/queue['data_audit_output'],dict(passed=True,diagnostic_only=True,cases=40,
        queue_sha256=sha(ROOT/queue['queue_path']),records=records,cuda_context_initialized=False))


def setup(target,config,queue):
    torch,np,datasets=source_imports(target,queue);torch.set_num_threads(queue['settings']['cpu_threads'])
    import wandb
    wandb.init(mode='disabled')
    module=original_module(target)
    from models.s4models import Model
    data,record=data_for(queue,config,target,torch,np,datasets)
    model=make_model(torch,Model,config)
    return torch,np,module,data,record,model


def integration(queue):
    target=workspace(queue,queue['integration_artifact']);cases=[]
    torch,np,datasets=source_imports(target,queue);torch.set_num_threads(2)
    from models.s4models import Model
    module=original_module(target)
    for profile in queue['profiles']:
        job=next(j for j in queue['jobs'] if read(ROOT/j['model_config'])['profile']==profile
                 and read(ROOT/j['model_config'])['dataset']=='fred_md')
        config=read(ROOT/job['model_config']);data,record=data_for(queue,config,target,torch,np,datasets)
        x=data['train_dataloader'].dataset.tensors[0][:2].clone()
        model=make_model(torch,Model,config);model.eval()
        with torch.no_grad():
            parallel=model.conditional_mean(x);changed=x.clone();cut=x.shape[1]//2;changed[:,cut:]+=2
            future=model.conditional_mean(changed)
            if not torch.allclose(parallel[:,:cut+1],future[:,:cut+1],rtol=1e-4,atol=1e-4):
                raise ValueError('SaShiMi reconstruction uses future targets')
            model.setup_rnn();recurrent=model.recurrent_means(x)
            if not torch.allclose(parallel,recurrent,rtol=1e-4,atol=1e-4):
                raise ValueError('Native SaShiMi convolution and recurrence differ')
            generated=model.generate(2,config['length'],device='cpu')
        model.train();loss=model(x)
        if not torch.isfinite(loss):raise ValueError('Gaussian objective is nonfinite')
        optimizer,_=module.setup_optimizer(model,.001,0.,config['optim']['epochs'])
        optimizer.zero_grad();loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError('Gaussian AR objective did not backpropagate')
        optimizer.step()
        if not bool(torch.isfinite(model(x))):raise ValueError('Actual diagnostic optimizer update became nonfinite')
        if not bool(torch.isfinite(generated).all()):raise ValueError('Full-length diagnostic generation became nonfinite')
        arrays=Path(queue['integration_artifact'])/(profile+'_full_fred_arrays.npz')
        np.savez_compressed(arrays,reference=x.detach().numpy(),parallel=parallel.numpy(),
            recurrent=recurrent.numpy(),generated=generated.numpy())
        evaluator=evaluate(torch,np,module,model,data,config,Path(queue['integration_artifact']),
            profile+'_full_native_cpu_evaluators',device='cpu')
        cases.append(dict(profile=profile,full_original_architecture=True,diagnostic_rows=2,
            diagnostic_length=config['length'],real_dataset='fred_md',causality_passed=True,
            recurrence_max_abs_error=float((parallel-recurrent).abs().max()),
            gaussian_nll_backward_passed=True,actual_optimizer_updates=1,generated_shape=list(generated.shape),
            parameters=sum(p.numel() for p in model.parameters()),
            arrays=str(arrays),arrays_sha256=sha(arrays),
            s4_blocks=sum(isinstance(m,importlib.import_module('models.s4').S4) for m in model.modules())))
        cases[-1]['full_native_cpu_evaluators']=evaluator
    if torch.cuda.is_initialized():raise ValueError('CPU AR integration initialized CUDA')
    write(ROOT/queue['integration_output'],dict(passed=True,diagnostic_only=True,
        queue_sha256=sha(ROOT/queue['queue_path']),cases=cases,no_benchmark_result=True))


def train_epoch(torch,model,ema,loader,optimizer,scheduler,step,start=200):
    model.train();losses=[];updates=0;samples=0
    for data,masks in loader:
        if not bool(masks.all()):raise ValueError('SaShiMi full Monash mask changed')
        data=data.cuda();optimizer.zero_grad();loss=model(data)
        if not torch.isfinite(loss):raise ValueError('Full Gaussian AR loss nonfinite')
        loss.backward();optimizer.step();step+=1;updates+=1;samples+=len(data)
        if ema is not None and step>start:ema.update_parameters(model)
        losses.append(float(loss.detach()))
    scheduler.step(sum(losses)/len(losses))
    return step,updates,samples


def evaluate(torch,np,module,model,data,config,artifact,recipe,device='cuda'):
    import metrics
    active=dict(classifier_updates=0,predictor_updates=0);frames={}
    old_step=torch.optim.AdamW.step;old_classify=metrics.compute_classification_score
    native_bce=torch.nn.BCEWithLogitsLoss.forward;native_mse=torch.nn.MSELoss.forward
    final_test_batches={'classification':[],'prediction':[]}
    def criterion(original,name):
        def observed(instance,input,target,*args,**kw):
            value=original(instance,input,target,*args,**kw)
            if not torch.is_grad_enabled():
                frame=inspect.currentframe().f_back
                while frame:
                    if (Path(frame.f_code.co_filename).resolve()==Path(metrics.__file__).resolve()
                        and frame.f_code.co_name==('compute_classification_score' if name=='classification' else 'compute_predictive_score')):
                        if frame.f_locals.get('i')==99:
                            final_test_batches[name].append(dict(mean=float(value.detach()),elements=input.numel()))
                        break
                    frame=frame.f_back
            return value
        return observed
    def step(optimizer,*args,**kw):
        frame=inspect.currentframe().f_back;result=old_step(optimizer,*args,**kw)
        if frame.f_code.co_name in ('compute_classification_score','compute_predictive_score'):
            name='classifier_updates' if frame.f_code.co_name=='compute_classification_score' else 'predictor_updates'
            active[name]+=1;frames[name]=frame
        return result
    def classify(fake,real,*args,**kw):
        path=artifact/(recipe+'_evaluation.npz');fake_array=fake.detach().cpu().numpy();real_array=real.detach().cpu().numpy()
        if not np.isfinite(fake_array).all() or not np.isfinite(real_array).all():raise ValueError('Nonfinite complete SaShiMi evaluation arrays')
        np.savez_compressed(path,generated=fake_array,reference=real_array)
        active.update(arrays=str(path),arrays_sha256=sha(path),generated_shape=list(fake_array.shape),reference_shape=list(real_array.shape))
        return old_classify(fake,real,*args,**kw)
    torch.optim.AdamW.step=step;metrics.compute_classification_score=classify
    torch.nn.BCEWithLogitsLoss.forward=criterion(native_bce,'classification')
    torch.nn.MSELoss.forward=criterion(native_mse,'prediction')
    rng=artifact/(recipe+'_evaluation_rng.pt')
    torch.save(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate()),rng)
    try:
        model.eval()
        values=metrics.compute_all_metrics(model,data['test_dataloader'],module.setup_optimizer,device,
            torch.nn.Sigmoid() if config['activation']=='sigmoid' else torch.nn.Identity())
        for name,frame in frames.items():
            path=artifact/(recipe+'_'+name+'.pt')
            value=dict(model=frame.f_locals['model'].state_dict(),optimizer=frame.f_locals['optimizer'].state_dict())
            if name=='classifier_updates':value['randperm']=frame.f_locals['randperm']
            torch.save(value,path);active[name+'_checkpoint']=str(path);active[name+'_checkpoint_sha256']=sha(path)
        active.update(recipe=recipe,metrics={k:float(v) for k,v in values.items()},
            evaluation_rng=str(rng),evaluation_rng_sha256=sha(rng),
            metric_boundary='Original100-epoch classifiers/predictors; SUM of test batch means; no paper-average certification')
        if not all(math.isfinite(v) for v in active['metrics'].values()):
            raise ValueError('Nonfinite complete SaShiMi metrics')
        required=budget(config)
        if active['classifier_updates']!=required['classifier_updates'] or active['predictor_updates']!=required['predictor_updates']:
            raise ValueError('Full native100-epoch SaShiMi evaluator budget differs')
        shape=[config['test_samples'],config['length'],1]
        if active['generated_shape']!=shape or active['reference_shape']!=shape:
            raise ValueError('Full SaShiMi test trajectory coverage differs')
        active['final_test_batch_diagnostics']={name:test_batch_summary(values,
            required['classifier_test_batches' if name=='classification' else 'predictor_test_batches'])
            for name,values in final_test_batches.items()}
        for name,key in [('classification','clf_score'),('prediction','predictive_score')]:
            if not math.isclose(active['final_test_batch_diagnostics'][name]['native_sum_batch_means'],
                active['metrics'][key],rel_tol=1e-5,abs_tol=1e-6):
                raise ValueError('Observed native batch losses do not reproduce native returned score')
        return active
    finally:
        torch.optim.AdamW.step=old_step;metrics.compute_classification_score=old_classify
        torch.nn.BCEWithLogitsLoss.forward=native_bce;torch.nn.MSELoss.forward=native_mse


def gpu_preflight(queue):
    target=workspace(queue,queue['gpu_preflight_artifact']);cases=[]
    for job in queue['jobs']:
        config=read(ROOT/job['model_config'])
        if config['seed']!=queue['seeds'][0]:continue
        torch,np,module,data,record,model=setup(target,config,queue)
        if not torch.cuda.is_available():raise ValueError('Full native SaShiMi GPU proof requires CUDA')
        model.cuda();optimizer,scheduler=module.setup_optimizer(model,.001,0.,config['optim']['epochs'])
        step,updates,samples=train_epoch(torch,model,None,data['train_dataloader'],optimizer,scheduler,0)
        case=Path(queue['gpu_preflight_artifact'])/job['id'];case.mkdir()
        evaluation=evaluate(torch,np,module,model,data,config,case,'diagnostic')
        if updates!=budget(config)['batches'] or samples!=config['train_samples']:
            raise ValueError('Full SaShiMi epoch capacity proof incomplete')
        cases.append(dict(dataset=config['dataset'],profile=config['profile'],full_epoch_updates=updates,
            full_train_samples=samples,full_test_samples=config['test_samples'],length=config['length'],
            classifier_updates=evaluation['classifier_updates'],predictor_updates=evaluation['predictor_updates'],passed=True))
        del model,optimizer,scheduler,data
        import gc
        gc.collect();torch.cuda.empty_cache()
    if len(cases)!=8:raise ValueError('All8 full dataset/profile GPU cases required')
    write(ROOT/queue['gpu_preflight_output'],dict(passed=True,diagnostic_only=True,
        queue_sha256=sha(ROOT/queue['queue_path']),cases=cases))


def validate_result(result,config):
    required=budget(config);epochs=config['optim']['epochs']
    if result['generator_updates']!=required['generator_updates']:
        raise ValueError('Full SaShiMi generator budget incomplete')
    if result['epoch_updates']!={str(e):required['batches'] for e in range(epochs)} or result['epoch_samples']!={str(e):config['train_samples'] for e in range(epochs)}:
        raise ValueError('Full SaShiMi epoch/sample coverage incomplete')
    if [r['recipe'] for r in result['metric_history']]!=required['evaluation_recipes']:
        raise ValueError('Both last-model and last-EMA recipes required')
    for r in result['metric_history']:
        if r['classifier_updates']!=required['classifier_updates'] or r['predictor_updates']!=required['predictor_updates']:
            raise ValueError('Full SaShiMi evaluator updates incomplete')
        shape=[config['test_samples'],config['length'],1]
        if r['reference_shape']!=shape or r['generated_shape']!=shape or not all(math.isfinite(v) for v in r['metrics'].values()):
            raise ValueError('Full SaShiMi generated coverage/metrics invalid')


def run_job(queue,job):
    proof=read(ROOT/queue['gpu_preflight_output'])
    if not proof['passed'] or proof['queue_sha256']!=sha(ROOT/queue['queue_path']):
        raise ValueError('Full SaShiMi GPU proof missing')
    config=read(ROOT/job['model_config']);artifact=Path(job['artifact_directory']);target=workspace(queue,artifact)
    torch,np,module,data,record,model=setup(target,config,queue)
    if not torch.cuda.is_available():raise ValueError('Full SaShiMi reconstruction requires CUDA')
    model.cuda();optimizer,scheduler=module.setup_optimizer(model,.001,0.,config['optim']['epochs'])
    ema=torch.optim.swa_utils.AveragedModel(model,avg_fn=lambda avg,param,count:.999*avg+.001*param)
    output=ROOT/job['output_directory'];output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    state=dict(id=job['id'],status='running',queue_sha256=sha(ROOT/queue['queue_path']),
        model_config_sha256=sha(ROOT/job['model_config']),generator_updates=0,epoch_updates={},epoch_samples={},
        data_record=record,metric_history=[],diagnostic_only=False,author_equivalence_certified=False,
        reproduction_status='paper_informed_Gaussian_AR_wrapper_unmodified_source_backbone')
    try:
        for epoch in range(config['optim']['epochs']):
            steps,updates,samples=train_epoch(torch,model,ema,data['train_dataloader'],optimizer,scheduler,state['generator_updates'])
            state['generator_updates']=steps;state['epoch_updates'][str(epoch)]=updates;state['epoch_samples'][str(epoch)]=samples
            write(output/'progress.json',state)
        # Preserve the completed generator before expensive original evaluators.
        checkpoint=artifact/'final_full_checkpoint.pt'
        evaluation_rng=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate())
        torch.save(dict(model=model.state_dict(),ema_model=ema.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),config=config,generator_updates=state['generator_updates'],
            epoch_updates=state['epoch_updates'],epoch_samples=state['epoch_samples'],**evaluation_rng),checkpoint)
        state.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),generator_training_completed=True)
        write(output/'progress.json',state)
        for recipe,current in [('last_model',model),('last_ema',ema)]:
            torch.set_rng_state(evaluation_rng['torch_rng']);torch.cuda.set_rng_state_all(evaluation_rng['cuda_rng'])
            np.random.set_state(evaluation_rng['numpy_rng']);random.setstate(evaluation_rng['python_rng'])
            state['metric_history'].append(evaluate(torch,np,module,current,data,config,artifact,recipe))
            write(output/'progress.json',state)
        validate_result(state,config)
        state.update(status='completed',seconds=time.perf_counter()-start,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            completed_utc=datetime.now(timezone.utc).isoformat(),independent_model_reinference_complete=False)
        write(output/'result.json',state)
    except BaseException as error:
        write(output/'failure.json',dict(state,status='failed_or_partial_preserved',error=repr(error)));raise


def verify_result(queue,job):
    result=read(ROOT/job['output_directory']/'result.json');validate_result(result,read(ROOT/job['model_config']))
    if result['status']!='completed' or result['queue_sha256']!=sha(ROOT/queue['queue_path']) or result['model_config_sha256']!=sha(ROOT/job['model_config']):
        raise ValueError('SaShiMi frozen result binding differs')
    if sha(result['checkpoint'])!=result['checkpoint_sha256']:raise ValueError('SaShiMi trained checkpoint differs')
    correction=read(ROOT/queue['cauchy_correction_config'])
    if not result.get('backend_correction',{}).get('algorithmic_correction') or result['backend_correction']['official_source_sha256']!=correction['official_cauchy_source_sha256']:
        raise ValueError('SaShiMi result lacks actual corrected backend proof')
    for r in result['metric_history']:
        for key in ['arrays','evaluation_rng','classifier_updates_checkpoint','predictor_updates_checkpoint']:
            if sha(r[key])!=r[key+'_sha256']:raise ValueError('SaShiMi complete evaluator artifact differs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True);parser.add_argument('--job-id')
    parser.add_argument('--audit-data',action='store_true');parser.add_argument('--gpu-preflight',action='store_true')
    parser.add_argument('--integration-check',action='store_true');args=parser.parse_args()
    queue=read(ROOT/args.queue);verify(queue)
    if args.audit_data:audit_data(queue)
    elif args.gpu_preflight:gpu_preflight(queue)
    elif args.integration_check:integration(queue)
    else:run_job(queue,next(j for j in queue['jobs'] if j['id']==args.job_id))


if __name__=='__main__':main()
