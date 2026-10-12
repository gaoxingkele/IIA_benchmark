"""Independently restore full SHCL state and replay every saved stochastic frame."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import random

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.shcl_author_audits_v1 import environment


def restore_rng(state):
    import numpy as np
    import torch
    if state['torch_cuda']:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=len(state['torch_cuda']):
            raise ValueError('Exact original visible CUDA generator count required')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'])


def verify_rng(state):
    import numpy as np
    import torch
    actual=np.random.get_state()
    if (random.getstate()!=state['python'] or actual[0]!=state['numpy'][0]
            or not np.array_equal(actual[1],state['numpy'][1]) or actual[2:]!=state['numpy'][2:]
            or not torch.equal(torch.get_rng_state(),state['torch_cpu'])):
        raise ValueError('Full Python/NumPy/native CPU RNG post-state differs')
    if state['torch_cuda']:
        actual_cuda=torch.cuda.get_rng_state_all()
        if len(actual_cuda)!=len(state['torch_cuda']) or any(not torch.equal(a,b) for a,b in zip(actual_cuda,state['torch_cuda'])):
            raise ValueError('Full native CUDA RNG post-state differs')


def set_backend_flags(flags):
    import torch
    if set(flags)!={'cudnn_benchmark','cudnn_deterministic','cudnn_enabled','cudnn_allow_tf32','matmul_allow_tf32','deterministic_algorithms'}:
        raise ValueError('Every original saved backend setting required')
    torch.backends.cudnn.benchmark=flags['cudnn_benchmark']; torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
    torch.backends.cudnn.enabled=flags['cudnn_enabled']; torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32']; torch.use_deterministic_algorithms(flags['deterministic_algorithms'])


def author_adjustment_ast(source):
    tree=ast.parse(source)
    solver=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Solver')
    test=next(n for n in solver.body if isinstance(n,ast.FunctionDef) and n.name=='test')
    start=next(i for i,n in enumerate(test.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='anomaly_state' for t in n.targets))
    end=next(i for i,n in enumerate(test.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='pa_accuracy' for t in n.targets))
    function=ast.parse('def adjustment(pred,gt):\n    pass').body[0]
    function.body=test.body[start:end]+ast.parse('return pred,gt').body
    return ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))


def adjusted_metrics(model,prediction,truth):
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support
    source=(native.verify(model)/'solver.py').read_text(encoding='utf-8'); namespace=dict(np=np)
    exec(compile(author_adjustment_ast(source),'<untouched_original_SHCL_adjustment>','exec'),namespace)
    pred,gt=namespace['adjustment'](prediction.copy(),truth.copy())
    precision,recall,f1,_=precision_recall_fscore_support(gt,pred,average='binary')
    return dict(adjusted_accuracy=float(accuracy_score(gt,pred)),adjusted_precision=float(precision),
        adjusted_recall=float(recall),adjusted_f_score=float(f1))


def constructor(model,job,device):
    if job.get('synthetic_fixture'):
        parameters=dict(model['constructor_parameters']); parameters['device']=device
        return parameters
    args=native.parse_author_cli(model,job['argv'])
    return dict(win_size=args.win_size,seq_size=args.seq_size,c_in=args.input_c,c_out=args.output_c,
        d_model=args.d_model,e_layers=args.e_layers,fr=args.fr,tr=args.tr,device=device)


def replay_training(model_config,training,original_checkpoint,report):
    import torch
    from scripts.flow_matching.audit_shcl_native_observer_v1 import equal
    job=training['job']
    if training['synthetic_operator_only_not_benchmark_performance']!=job.get('synthetic_fixture',False):
        raise ValueError('Diagnostic training must remain distinct from original benchmark training')
    if len(training['epochs'])!=job['num_epochs'] or training['optimizer_updates']!=job['expected_optimizer_updates']:
        raise ValueError('Every original registered full training state and budget required')
    restored=[]
    for index,receipt in enumerate(training['epochs']):
        path=Path(receipt['full_state'])
        if native.sha(path)!=receipt['full_state_sha256']: raise ValueError('Original full training checkpoint changed')
        checkpoint=torch.load(path,map_location='cpu')
        if (checkpoint['source_job']!=job or checkpoint['completed_epoch']!=index+1
                or receipt['epoch']!=index+1 or checkpoint['optimizer_updates']!=receipt['optimizer_updates']
                or receipt['optimizer_updates']!=(index+1)*(job['expected_optimizer_updates']//job['num_epochs'])):
            raise ValueError('Full original epoch/update identity required')
        model,_=native.build_model(model_config,constructor(model_config,job,'cpu'))
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=checkpoint['optimizer']['param_groups'][0]['lr'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if not equal(model.state_dict(),checkpoint['model']) or not equal(optimizer.state_dict(),checkpoint['optimizer']):
            raise ValueError('All native model and Adam variables must restore bitwise')
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()): raise ValueError('Every restored model tensor must be finite')
        for value in optimizer.state.values():
            if any(isinstance(v,torch.Tensor) and not torch.isfinite(v).all() for v in value.values()):
                raise ValueError('Every restored original Adam tensor must be finite')
        restored.append(dict(epoch=index+1,checkpoint=str(path),checkpoint_sha256=native.sha(path),
            all_model_and_Adam_state_bitwise_restored=True,original_optimizer_updates=receipt['optimizer_updates'],
            new_optimizer_updates_executed=0))
        if index==len(training['epochs'])-1:
            if native.sha(original_checkpoint)!=receipt['original_model_checkpoint_sha256']:
                raise ValueError('Final untouched author checkpoint changed')
            native_state=torch.load(original_checkpoint,map_location='cpu')
            if not equal(native_state,checkpoint['model']): raise ValueError('Final native checkpoint differs from complete captured state')
        del model,optimizer,checkpoint
    report['training_checkpoints']=restored
    report['all_original_epochs_and_last_native_checkpoint_verified']=True


def replay_test(model_config,test,checkpoint_path,target,report):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support
    if test['synthetic_operator_only_not_benchmark_performance']!=test['job'].get('synthetic_fixture',False):
        raise ValueError('Diagnostic outputs must never be relabeled as real-data source results')
    job=test['job']; frames=test['full_source_score_frames']; sequences={phase:[] for phase in ('threshold','test')}; labels=[]
    expected={phase:job['expected_'+phase+'_windows'] for phase in sequences}
    if sum(frame['shape'][0] for frame in frames if frame['phase']=='threshold')!=expected['threshold'] or \
            sum(frame['shape'][0] for frame in frames if frame['phase']=='test')!=expected['test']:
        raise ValueError('Every complete source threshold/test window required')
    device=None; model=None; loss=None; replayed=[]
    native_state=torch.load(checkpoint_path,map_location='cpu')
    for frame in frames:
        if native.sha(frame['context'])!=frame['context_sha256'] or native.sha(frame['arrays'])!=frame['arrays_sha256']:
            raise ValueError('Original full native context/score frame changed')
        context=torch.load(frame['context'],map_location='cpu')
        if context['phase']!=frame['phase'] or context['original_batch_index']!=len(sequences[frame['phase']]):
            raise ValueError('All original ordered frame identities required')
        frame_device='cuda:0' if context['pre_forward_rng']['torch_cuda'] else 'cpu'
        if model is None:
            device=frame_device; model,loss=native.build_model(model_config,constructor(model_config,job,device))
            model.load_state_dict(native_state,strict=True); model.eval()
            before_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif device!=frame_device: raise ValueError('Original device changed within complete native test')
        set_backend_flags(context['backend_flags']); restore_rng(context['pre_forward_rng'])
        inputs=context['input'].to(device)
        with torch.no_grad(): actual_loss=loss(model(inputs)); actual_score=torch.softmax(actual_loss,dim=-1)
        verify_rng(context['post_forward_rng'])
        with np.load(frame['arrays'],allow_pickle=False) as archive:
            saved_loss=archive['loss']; saved_score=archive['score']; saved_labels=archive['labels']
        if (list(actual_score.shape)!=frame['shape'] or not np.array_equal(actual_loss.cpu().numpy(),saved_loss)
                or not np.array_equal(actual_score.cpu().numpy(),saved_score)
                or not np.array_equal(context['labels'].numpy(),saved_labels)):
            raise ValueError('Every complete native source loss/score/label frame must replay bitwise; no tolerance relaxation')
        sequences[frame['phase']].append(actual_score.cpu().numpy())
        if frame['phase']=='test': labels.append(saved_labels)
        replayed.append(dict(frame=frame,all_full_native_losses_scores_labels_bitwise_verified=True,
            complete_original_CPU_NumPy_Python_CUDA_RNG_poststate_verified=True,optimizer_updates_executed=0))
    if any(not torch.equal(before_state[k],v.detach().cpu()) for k,v in model.state_dict().items()):
        raise ValueError('Read-only replay changed model state')
    threshold_scores=np.concatenate(sequences['threshold'],axis=0).reshape(-1)
    scores=np.concatenate(sequences['test'],axis=0).reshape(-1); truth=np.concatenate(labels,axis=0).reshape(-1).astype(int)
    threshold=float(np.percentile(threshold_scores,100-job['anormly_ratio'])); pred=(scores>threshold).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(truth,pred,average='binary')
    metrics=dict(threshold=threshold,accuracy=float(accuracy_score(truth,pred)),precision=float(precision),recall=float(recall),f_score=float(f1))
    for key,value in metrics.items():
        if float(test['original_results'][key])!=value: raise ValueError('Complete source metric replay differs: '+key)
    if not job.get('synthetic_fixture'):
        for key,value in adjusted_metrics(model_config,pred,truth).items():
            if float(test['original_results'][key])!=value: raise ValueError('Untouched author point-adjusted metric differs: '+key)
    target=Path(target); target.mkdir(parents=True,exist_ok=False); path=target/'all_replayed_scores.npz'
    with path.open('xb') as stream: np.savez_compressed(stream,threshold_scores=threshold_scores,scores=scores,truth=truth,prediction=pred)
    report.update(full_native_frames_replayed=replayed,all_source_original_metrics_exactly_verified=True,
        all_model_variables_unchanged=True,full_replayed_points=len(scores),original_test_prefix_only=True,
        strict_full_point_TAB=False,replayed_complete_arrays=str(path),replayed_complete_arrays_sha256=native.sha(path),
        synthetic_operator_only_not_benchmark_performance=job.get('synthetic_fixture',False),original_source_job=job,
        captured_utc=datetime.now(timezone.utc).isoformat(),environment=environment(),new_benchmark_results=0,
        original_paper_equivalence_certified=False,all_original_experiments_complete=False)
    return report
