"""Verify the saved local GRASP graph and every unchanged path/score evaluation."""
from __future__ import annotations


def verify_training_graph(model,training):
    import numpy as np,torch
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,FORMULA_TOLERANCE
    x=torch.as_tensor(np.asarray(training),dtype=torch.float32).transpose(1,2)
    flat=x.transpose(1,2).reshape(-1,x.shape[1]).double()
    corr=torch.corrcoef(flat.T).nan_to_num(0).abs();a=(corr>=model.graph_threshold).float()
    if model.variant=='er':
        i,j=torch.triu_indices(len(a),len(a),1);count=int((a[i,j]>0).sum())
        order=torch.randperm(len(i),generator=torch.Generator(device='cpu').manual_seed(model.seed))[:count]
        a=torch.zeros_like(a);a[i[order],j[order]]=1;a[j[order],i[order]]=1
    a.fill_diagonal_(0);require_identical(a.numpy(),model.adjacency_)
    basis=model.path.basis.detach().cpu().numpy().astype(np.float64);omega=model.path.omega.detach().cpu().numpy().astype(np.float64)
    adjacency=a.numpy().astype(np.float64);degree=adjacency.sum(-1);inv=np.zeros_like(degree)
    inv[degree>0]=1/np.sqrt(degree[degree>0]);lap=np.diag((degree>0).astype(float))-inv[:,None]*adjacency*inv[None,:]
    if not np.allclose(basis.T@basis,np.eye(len(basis)),**FORMULA_TOLERANCE):raise ValueError('Actual saved source basis must remain orthonormal')
    transformed=basis.T@lap@basis
    if not np.allclose(transformed,np.diag(np.diag(transformed)),**FORMULA_TOLERANCE):raise ValueError('Actual source basis must diagonalize the complete trained graph')
    tau=0. if model.variant=='lin' else model.tau
    if not np.allclose(omega**2,tau*np.maximum(np.diag(transformed),0),**FORMULA_TOLERANCE):raise ValueError('Actual saved source frequencies must retain the original graph and tau')
    return dict(full_training_windows=len(training),every_training_coordinate_used=True,
        original_train_only_Pearson_or_matched_edge_ER_graph_regenerated=True,actual_saved_eigenbasis_retained=True,
        new_eigenbasis_substituted=False,independent_FP64_graph_diagonalization_checked=True)


def checked_score(model):
    import numpy as np,torch
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import FORMULA_TOLERANCE
    from scripts.flow_matching.neural_full510_checkpoint_state_v1 import require_tensor_bytes
    basis=model.path.basis.detach().cpu().numpy().astype(np.float64);omega=model.path.omega.detach().cpu().numpy().astype(np.float64)
    report=dict(method='grasp_'+model.variant,score_calls=0,windows=0,window_points=0,path_forwards=0,velocity_forwards=0,
        source_origin_coordinates_regenerated=0,independent_path_coordinates_checked=0,independent_score_residual_coordinates_checked=0,
        maximum_FP64_path_absolute_error=0.,maximum_FP64_score_absolute_error=0.,
        independent_FP64_path_and_score_checked=True,independent_neural_kernel_implementation=False)
    def n(value):return value.detach().cpu().numpy().astype(np.float64)
    def score(windows):
        x=torch.as_tensor(np.asarray(windows),dtype=torch.float32).transpose(1,2);position=0;calls=0;pending={};finished=[];result=None
        origins=torch.randn((model.source_samples,1,x.shape[1],model.window),
            generator=torch.Generator(device=model.device).manual_seed(model.seed+1),device=model.device)
        times=torch.linspace(0,1,model.flow_evaluations+2,device=model.device)[1:-1]
        report['source_origin_coordinates_regenerated']+=origins.numel()
        def path(module,args,out):
            nonlocal calls
            if pending:raise ValueError('Every original GRASP path must feed its corresponding velocity call')
            stop=min(position+model.batch_size,len(x));source,target,time=args
            index=calls%(model.source_samples*model.flow_evaluations);s=index//model.flow_evaluations;t=index%model.flow_evaluations
            require_tensor_bytes(source,origins[s].expand(len(target),-1,-1));require_tensor_bytes(target,x[position:stop])
            require_tensor_bytes(time,times[t].expand(len(target)))
            values=n(time).reshape(-1,1,1);w=omega[None,:,None];positive=w>1e-7;safe=np.where(positive,w,1.)
            a=np.where(positive,np.sinh(safe*(1-values))/np.sinh(safe),1-values)
            b=np.where(positive,np.sinh(safe*values)/np.sinh(safe),values)
            da=np.where(positive,-safe*np.cosh(safe*(1-values))/np.sinh(safe),-np.ones_like(values))
            db=np.where(positive,safe*np.cosh(safe*values)/np.sinh(safe),np.ones_like(values))
            s0=np.einsum('ck,bcl->bkl',basis,n(source));s1=np.einsum('ck,bcl->bkl',basis,n(target))
            expected=(np.einsum('ck,bkl->bcl',basis,a*s0+b*s1),np.einsum('ck,bkl->bcl',basis,da*s0+db*s1))
            for actual,wanted in zip(out,expected):
                observed=n(actual)
                if observed.shape!=wanted.shape or not np.allclose(observed,wanted,**FORMULA_TOLERANCE):raise ValueError('Independent complete source GRASP path/velocity mismatch')
                report['maximum_FP64_path_absolute_error']=max(report['maximum_FP64_path_absolute_error'],float(np.max(np.abs(observed-wanted))))
                report['independent_path_coordinates_checked']+=observed.size
            weight=np.where(positive,(np.sinh(safe*values)/safe)**2,values**2) if model.variant=='grasp' else 1.
            pending.update(state=out[0],velocity=out[1],time=time,weight=weight,stop=stop)
            calls+=1;report['path_forwards']+=1
        def velocity(module,args,out):
            nonlocal position,result
            if not pending:raise ValueError('Every original velocity must follow its exact checked path')
            require_tensor_bytes(args[0],pending['time']);require_tensor_bytes(args[1],pending['state'])
            residual=np.einsum('ck,bcl->bkl',basis,n(out)-n(pending['velocity']))
            contribution=(residual**2*pending['weight']).sum((1,2))
            result=contribution if result is None else result+contribution
            report['independent_score_residual_coordinates_checked']+=residual.size;report['velocity_forwards']+=1
            if calls%(model.source_samples*model.flow_evaluations)==0:
                finished.append(result/model.source_samples);result=None;position=pending['stop']
            pending.clear()
        h1=model.path.register_forward_hook(path);h2=model.model.register_forward_hook(velocity)
        try:actual=model.score(windows)
        finally:h1.remove();h2.remove()
        independent=np.concatenate(finished) if finished else np.empty(0)
        if (position!=len(x) or pending or result is not None or actual.shape!=independent.shape
                or not np.isfinite(actual).all() or not np.allclose(actual,independent,**FORMULA_TOLERANCE)):
            raise ValueError('Every source/window/time/sample and independent complete GRASP score required')
        report['maximum_FP64_score_absolute_error']=max(report['maximum_FP64_score_absolute_error'],float(np.max(np.abs(actual.astype(np.float64)-independent))))
        report['score_calls']+=1;report['windows']+=len(x);report['window_points']+=len(x)*model.window
        return actual
    return score,report
