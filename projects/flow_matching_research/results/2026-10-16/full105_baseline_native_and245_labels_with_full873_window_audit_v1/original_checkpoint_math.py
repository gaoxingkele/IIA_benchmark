"""Restore complete legacy checkpoints and check every unchanged score forward.

Uses source network kernels; independently evaluates scoring from their outputs
in NumPy FP64. This verifies scoring,not an independent neural implementation.
"""
from __future__ import annotations


def restore_model(checkpoint,job,configuration,features):
    import torch
    from iia_benchmark.models.mtsad_detectors import build_detector
    from iia_benchmark.models.fm_legacy_window_adapter import LegacyWindowAdapter
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    if (checkpoint['configuration']!=configuration or checkpoint['overrides']!=job['parameter_overrides']
            or checkpoint['window']!=job['window'] or set(checkpoint['modules'])!={'_model'}
            or checkpoint['adjacency'] is not None):
        raise ValueError('Complete original legacy configuration and checkpoint required')
    params=configuration['parameters']
    adapter=LegacyWindowAdapter(**dict(params,**job['parameter_overrides']))
    delegate=build_detector(adapter.method,dict(adapter.parameters,epochs=adapter.epochs,seed=adapter.seed,device='cpu'))
    delegate._device=torch.device('cpu')
    built=delegate._build(features)
    modules=torch.nn.ModuleList(built) if isinstance(built,(tuple,list)) else built
    state=checkpoint['modules']['_model'];expected=modules.state_dict()
    if set(state)!=set(expected):raise ValueError('Every full original module tensor key required')
    for key,value in expected.items():
        if value.dtype!=state[key].dtype or value.shape!=state[key].shape or not torch.isfinite(state[key]).all():
            raise ValueError('Original complete finite parameter and buffer shapes/dtypes required')
    modules.load_state_dict(state,strict=True)
    for key,value in modules.state_dict().items():tensor_bytes(value,state[key])
    # Preserve source requires_grad flags. Even under no_grad,the CPU kernel
    # selection can depend on these flags (verified for the pinned iTransformer).
    # USAD has only Linear/ReLU modules;no stochastic/stateful scoring layers.
    # Its source score does not change training flags;do not infer saved flags.
    if adapter.method!='usad':modules.eval()
    delegate._model=tuple(modules) if isinstance(built,(tuple,list)) else modules
    adapter._model=modules;adapter._delegate=delegate
    return adapter


def association_numpy(series,priors,temperature,*,average_layers=True):
    import numpy as np
    if not series or len(series)!=len(priors):raise ValueError('All attention layers and priors required')
    left=0.;right=0.;coordinates=0
    for series_value,prior_value in zip(series,priors):
        s=np.asarray(series_value,dtype=np.float64);p=np.asarray(prior_value,dtype=np.float64)
        if (s.ndim!=4 or p.shape!=s.shape or s.shape[-1]!=s.shape[-2] or not np.isfinite(s).all()
                or not np.isfinite(p).all() or np.any(s<0) or np.any(p<0)):
            raise ValueError('Complete finite per-head query/key distributions required')
        mass=p.sum(-1,keepdims=True)
        if np.any(mass<=0):raise ValueError('Every prior row must have positive mass')
        p=p/mass
        delta=np.log(s+1e-4)-np.log(p+1e-4)
        # The source reduces keys before heads;keep all heads,not head means.
        left=left+(s*delta).sum(-1).mean(1)*temperature
        right=right+(p*(-delta)).sum(-1).mean(1)*temperature
        coordinates+=s.size+p.size
    divisor=len(series) if average_layers else 1
    energy=-(left/divisor+right/divisor);energy-=energy.max(-1,keepdims=True)
    weights=np.exp(energy);weights/=weights.sum(-1,keepdims=True)
    return weights,coordinates


def checked_score(adapter):
    import numpy as np,torch
    from types import FunctionType,MethodType
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import FORMULA_TOLERANCE
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    delegate=adapter._delegate;method=adapter.method
    if method not in ['anomaly_transformer','dcdetector','timesnet','itransformer','tranad','usad']:
        raise ValueError('Every registered baseline needs explicit score semantics')
    report=dict(method=method,score_calls=0,windows=0,window_points=0,forward_calls=0,
        attention_probability_coordinates_checked=0,reconstruction_coordinates_checked=0,
        maximum_FP64_score_absolute_error=0.,independent_FP64_score_program_checked=True,
        independent_neural_kernel_implementation=False,source_score_batch_size=512 if method=='usad' else delegate.batch_size,
        per_head_KL_calls_checked=0,source_FP32_energy_program_checked=method in ['anomaly_transformer','dcdetector'],
        ideal_FP64_complete_score_equivalence_certified=True,ideal_FP64_score_failure_coordinates=0,
        maximum_ideal_FP64_score_absolute_error=0.)

    def score(windows):
        if windows.ndim!=3 or windows.shape[1]!=delegate.window or not np.isfinite(windows).all():
            raise ValueError('Every complete original input window required')
        position=0;expected_scores=[];ideal_scores=[];handles=[];pending={}
        def np64(value):return value.detach().cpu().numpy().astype(np.float64)
        def expected_batch():
            stop=min(position+report['source_score_batch_size'],len(windows))
            return torch.from_numpy(windows[position:stop].astype(np.float32)),stop
        def common(module,args,out):
            nonlocal position
            batch,stop=expected_batch()
            if method=='tranad':
                tensor_bytes(args[0],batch.permute(1,0,2))
                tensor_bytes(args[1],batch[:,-1,:].view(1,len(batch),batch.shape[-1]))
                if len(out)!=2 or any(x.shape!=args[1].shape for x in out):raise ValueError('Both full TranAD decoder outputs required')
                target=np64(args[1])[0];output=np64(out[1])[0]
                expected=np.square(output-target).mean(-1)
                report['reconstruction_coordinates_checked']+=target.size
            else:
                tensor_bytes(args[0],batch)
                if method in ['anomaly_transformer','dcdetector']:
                    if method=='anomaly_transformer':
                        output,series,prior=out
                        if output.shape!=batch.shape:raise ValueError('Complete reconstruction required')
                    else:series,prior=out
                    arrays=[[x.detach().cpu().numpy() for x in values] for values in [series,prior]]
                    expected,coordinates=association_numpy(*arrays,delegate.temperature,average_layers=method=='dcdetector')
                    if expected.shape!=tuple(batch.shape[:2]):raise ValueError('Every query timestamp required')
                    report['attention_probability_coordinates_checked']+=coordinates
                    reconstruction=np.ones(batch.shape[:2],dtype=np.float64)
                    if method=='anomaly_transformer':
                        reconstruction=np.square(np64(batch)-np64(output)).mean(-1)
                        expected=expected*reconstruction
                        report['reconstruction_coordinates_checked']+=batch.numel()
                    if pending:raise ValueError('Every attention batch must complete its own KL and softmax program')
                    pending.update(attention=arrays,KL=[],ideal_score=expected,reconstruction=reconstruction)
                else:
                    if out.shape!=batch.shape:raise ValueError('Complete original reconstruction required')
                    expected=np.square(np64(batch)-np64(out)).mean(-1)
                    report['reconstruction_coordinates_checked']+=batch.numel()
            if not np.isfinite(expected).all():raise ValueError('Finite independent score formula required')
            if method not in ['anomaly_transformer','dcdetector']:
                expected_scores.append(expected);ideal_scores.append(expected)
            position=stop;report['forward_calls']+=1

        original_KL=delegate._kl if method in ['anomaly_transformer','dcdetector'] else None
        def observed_KL(p,q):
            if 'attention' not in pending:raise ValueError('Original attention output required before KL')
            calls=len(pending['KL']);series,prior=pending['attention'];index=calls//2
            if index>=len(series):raise ValueError('Unexpected additional per-head KL call')
            s=np.asarray(series[index],dtype=np.float64);prior=np.asarray(prior[index],dtype=np.float64)
            norm=prior/prior.sum(-1,keepdims=True)
            expected_p,expected_q=(s,norm) if calls%2==0 else (norm,s)
            a,b=np64(p),np64(q)
            for actual,wanted in [(a,expected_p),(b,expected_q)]:
                if actual.shape!=wanted.shape or not np.allclose(actual,wanted,**FORMULA_TOLERANCE):
                    raise ValueError('Every original prior normalization and per-head KL input must agree')
            out=original_KL(p,q)
            independent=(a*(np.log(a+1e-4)-np.log(b+1e-4))).sum(-1).mean(1)
            if tuple(out.shape)!=independent.shape or not np.allclose(np64(out),independent,**FORMULA_TOLERANCE):
                raise ValueError('Independent FP64 full per-head KL primitive mismatch')
            pending['KL'].append(out.detach().cpu().numpy().copy());report['per_head_KL_calls_checked']+=1
            return out

        class SourceTorch:
            def __getattr__(self,name):return getattr(torch,name)
            def softmax(self,energy,dim):
                if 'attention' not in pending or dim!=-1:raise ValueError('Complete original attention energy required')
                layers=len(pending['attention'][0]);values=pending['KL']
                if len(values)!=2*layers:raise ValueError('Every forward/reverse KL layer required')
                left=np.zeros_like(values[0]);right=np.zeros_like(values[0])
                for index in range(layers):
                    left=left+values[2*index]*np.float32(delegate.temperature)
                    right=right+values[2*index+1]*np.float32(delegate.temperature)
                if method=='dcdetector':left=left/np.float32(layers);right=right/np.float32(layers)
                expected=-left-right
                if tuple(energy.shape)!=expected.shape or not np.allclose(np64(energy),expected,**FORMULA_TOLERANCE):
                    raise ValueError('Independent source-FP32 KL/temperature/layer-reduction energy mismatch')
                # Source-rounded energy is independently checked above. Retain
                # the all-FP64 end-to-end result separately,including failures.
                independent=np64(energy);independent-=independent.max(-1,keepdims=True)
                independent=np.exp(independent);independent/=independent.sum(-1,keepdims=True)
                out=torch.softmax(energy,dim=dim)
                if not np.allclose(np64(out),independent,**FORMULA_TOLERANCE):raise ValueError('Independent FP64 softmax primitive mismatch')
                expected_scores.append(np64(out)*pending['reconstruction']);ideal_scores.append(pending['ideal_score'])
                pending.clear();return out

        def encoder(module,args,out):
            batch,stop=expected_batch();target=batch.reshape(len(batch),-1)
            if pending:raise ValueError('Each USAD scoring batch requires exactly one shared encoder forward')
            tensor_bytes(args[0],target)
            if out.shape!=(len(batch),delegate.latent_size):raise ValueError('Complete shared latent shape required')
            pending.update(target=np64(target),latent=out.detach().clone(),stop=stop)
            report['forward_calls']+=1
        def decoder1(module,args,out):
            if 'latent' not in pending or 'first' in pending:raise ValueError('First USAD decoder order required')
            tensor_bytes(args[0],pending['latent'])
            if tuple(out.shape)!=pending['target'].shape:raise ValueError('Full first USAD reconstruction required')
            pending['first']=np.square(pending['target']-np64(out)).mean(-1);report['forward_calls']+=1
        def decoder2(module,args,out):
            nonlocal position
            if 'first' not in pending:raise ValueError('Both USAD decoder outputs required')
            tensor_bytes(args[0],pending['latent'])
            if tuple(out.shape)!=pending['target'].shape:raise ValueError('Full second USAD reconstruction required')
            second=np.square(pending['target']-np64(out)).mean(-1)
            expected_scores.append(delegate.alpha*pending['first']+delegate.beta*second)
            ideal_scores.append(expected_scores[-1])
            report['reconstruction_coordinates_checked']+=2*pending['target'].size
            position=pending['stop'];pending.clear();report['forward_calls']+=1
        try:
            original_score=delegate.score
            if method in ['anomaly_transformer','dcdetector']:
                delegate._kl=observed_KL
                function=original_score.__func__;original_modules=function.__globals__['torch_modules']
                def private_modules():
                    _,nn=original_modules();return SourceTorch(),nn
                cloned=FunctionType(function.__code__,dict(function.__globals__,torch_modules=private_modules),function.__name__,function.__defaults__)
                delegate.score=MethodType(cloned,delegate)
            if method=='usad':
                for module,hook in zip(delegate._model,[encoder,decoder1,decoder2]):handles.append(module.register_forward_hook(hook))
            else:handles.append(delegate._model.register_forward_hook(common))
            actual=adapter.score(windows)
        finally:
            for handle in handles:handle.remove()
            if method in ['anomaly_transformer','dcdetector']:
                del delegate._kl;del delegate.score
        independent=np.concatenate(expected_scores) if expected_scores else np.empty(0)
        if (position!=len(windows) or pending or actual.shape!=independent.shape or not np.isfinite(actual).all()
                or not np.allclose(actual,independent,**FORMULA_TOLERANCE)):
            raise ValueError('Every full original forward and independent FP64 scoring formula must agree')
        report['maximum_FP64_score_absolute_error']=max(report['maximum_FP64_score_absolute_error'],float(np.max(np.abs(actual.astype(np.float64)-independent))))
        ideal=np.concatenate(ideal_scores)
        failures=int(np.count_nonzero(~np.isclose(actual,ideal,**FORMULA_TOLERANCE)))
        report['ideal_FP64_score_failure_coordinates']+=failures
        if failures:report['ideal_FP64_complete_score_equivalence_certified']=False
        report['maximum_ideal_FP64_score_absolute_error']=max(report['maximum_ideal_FP64_score_absolute_error'],float(np.max(np.abs(actual.astype(np.float64)-ideal))))
        report['score_calls']+=1;report['windows']+=len(windows);report['window_points']+=len(windows)*delegate.window
        return actual
    return score,report
