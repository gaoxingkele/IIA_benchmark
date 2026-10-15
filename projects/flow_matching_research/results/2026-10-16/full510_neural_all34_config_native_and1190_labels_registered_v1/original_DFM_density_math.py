"""Unchanged DFM Euler/autograd score with an analytic TimeMLP divergence audit."""
from __future__ import annotations


def analytic_field(model,x,time,eps=None):
    """FP64 MLP and Jacobian contraction;no finite differences or autograd."""
    import numpy as np
    state=model.state_dict();w0,w1,w2=[state[f'net.{i}.weight'].detach().cpu().numpy().astype(np.float64) for i in [0,2,4]]
    b0,b1,b2=[state[f'net.{i}.bias'].detach().cpu().numpy().astype(np.float64) for i in [0,2,4]]
    x=np.asarray(x,dtype=np.float64);time=np.asarray(time,dtype=np.float64).reshape(-1,1)
    h1=np.tanh(np.concatenate([x,time],axis=1)@w0.T+b0)
    h2=np.tanh(h1@w1.T+b1);velocity=h2@w2.T+b2
    d1,d2=1-h1*h1,1-h2*h2
    if eps is None:
        # Cyclic trace contraction avoids a batch*D*D Jacobian allocation.
        product=w1*(w0[:,:x.shape[1]]@w2).T
        divergence=((d1@product.T)*d2).sum(-1)
    else:
        eps=np.asarray(eps,dtype=np.float64)
        gradient=(((eps@w2)*d2)@w1)*d1
        divergence=((gradient@w0[:,:x.shape[1]])*eps).sum(-1)
    return velocity,divergence


def checked_score(model):
    import numpy as np,torch,math
    from types import FunctionType,MethodType
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import FORMULA_TOLERANCE
    from scripts.flow_matching.neural_full510_checkpoint_state_v1 import require_tensor_bytes
    function=model.score.__func__;original_cnf=function.__globals__['cnf_log_density']
    report=dict(method='dfm_'+model.trace,score_calls=0,windows=0,window_points=0,
        CNF_batches=0,velocity_forwards=0,autograd_calls=0,trace_coordinates_checked=0,source_Rademacher_coordinates_checked=0,
        independent_FP64_MLP_and_divergence_checked=True,source_FP32_Euler_and_correction_program_checked=True,
        full_ideal_FP64_ODE_trajectory_replayed=False,maximum_velocity_absolute_error=0.,maximum_divergence_absolute_error=0.,
        maximum_source_precision_density_absolute_error=0.,maximum_FP64_density_on_source_states_absolute_error=0.)
    def score(windows):
        position=0;generator=torch.Generator(device=model.device).manual_seed(model.seed+1)
        def audited_cnf(field,observations,steps,trace,source_generator):
            nonlocal position
            if field is not model.reverse_field or steps!=model.steps or trace!=model.trace:
                raise ValueError('Original complete reverse field,Euler budget and trace branch required')
            stop=min(position+model.batch_size,len(windows));target=torch.as_tensor(np.asarray(windows[position:stop],dtype=np.float32)).flatten(1)
            require_tensor_bytes(observations,target)
            state=dict(index=0,grad_calls=0,x=target.numpy().copy(),correction=np.zeros(len(target),np.float32),
                correction64=np.zeros(len(target),np.float64),eps=None)
            def close_step():
                if 'velocity' not in state:return
                required=state['x'].shape[1] if trace=='exact' else 1
                if state['grad_calls']!=required:raise ValueError('Every original Jacobian diagonal or Hutchinson gradient required')
                actual=state['divergence'];wanted=state['analytic_divergence']
                if not np.allclose(actual,wanted,**FORMULA_TOLERANCE):raise ValueError('Independent analytic TimeMLP divergence mismatch')
                report['maximum_divergence_absolute_error']=max(report['maximum_divergence_absolute_error'],float(np.max(np.abs(actual-wanted))))
                state['correction']=state['correction']-actual/np.float32(steps)
                state['correction64']=state['correction64']-wanted/steps
                state['x']=state['x']-state['velocity']/np.float32(steps)
                state['index']+=1;state['grad_calls']=0
                del state['velocity']
            def observed_field(time,x):
                close_step()
                if state['index']>=steps:raise ValueError('Extra original DFM Euler step')
                require_tensor_bytes(x,torch.from_numpy(state['x']))
                expected_time=torch.full((len(target),),1-state['index']/steps,dtype=torch.float32)
                require_tensor_bytes(time,expected_time)
                actual=field(time,x);independent,divergence=analytic_field(field,state['x'],time.detach().cpu().numpy(),state['eps'])
                value=actual.detach().cpu().numpy().copy()
                if not np.isfinite(value).all() or not np.allclose(value,independent,**FORMULA_TOLERANCE):raise ValueError('Independent full TimeMLP velocity mismatch')
                report['maximum_velocity_absolute_error']=max(report['maximum_velocity_absolute_error'],float(np.max(np.abs(value-independent))))
                state.update(velocity=value,velocity_tensor=actual,x_tensor=x,analytic_divergence=divergence,divergence=np.zeros(len(target),np.float32))
                report['velocity_forwards']+=1
                return actual
            class Grad:
                def grad(self,outputs,inputs,**kwargs):
                    if inputs is not state['x_tensor']:raise ValueError('Every source derivative must use the actual current CNF state')
                    j=state['grad_calls'];v=state['velocity_tensor']
                    expected=v[:,j].sum() if trace=='exact' else (v*torch.from_numpy(state['eps']).to(v.device)).sum()
                    require_tensor_bytes(outputs,expected)
                    result=torch.autograd.grad(outputs,inputs,**kwargs);gradient=result[0]
                    if gradient is None:raise ValueError('The configured dense TimeMLP must expose all source state derivatives')
                    if trace=='exact':
                        state['divergence']=state['divergence']+gradient[:,j].detach().cpu().numpy()
                        report['trace_coordinates_checked']+=len(target)
                    else:
                        value=gradient.detach().cpu().numpy();state['divergence']=(value*state['eps']).sum(-1,dtype=np.float32)
                        report['trace_coordinates_checked']+=value.size
                    state['grad_calls']+=1;report['autograd_calls']+=1
                    return result
            class Torch:
                autograd=Grad()
                def __getattr__(self,name):return getattr(torch,name)
                def randint(self,low,high,shape,**kwargs):
                    if trace!='hutchinson' or state['eps'] is not None or kwargs['generator'] is not source_generator:
                        raise ValueError('One original ordered Rademacher draw per full scoring batch required')
                    actual=torch.randint(low,high,shape,**kwargs)
                    expected=torch.randint(low,high,shape,**dict(kwargs,generator=generator));require_tensor_bytes(actual,expected)
                    state['eps']=(actual.to(torch.float32)*2-1).cpu().numpy();report['source_Rademacher_coordinates_checked']+=actual.numel()
                    return actual
            cloned=FunctionType(original_cnf.__code__,dict(original_cnf.__globals__,torch=Torch()),original_cnf.__name__,original_cnf.__defaults__)
            actual=cloned(observed_field,observations,steps,trace,source_generator);close_step()
            if state['index']!=steps:raise ValueError('All original backward Euler steps must complete')
            x=state['x'];prior=np.float32(-.5)*((x*x)+np.float32(math.log(2*math.pi))).sum(-1,dtype=np.float32)
            expected=prior+state['correction'];observed=actual.detach().cpu().numpy()
            if not np.isfinite(observed).all() or not np.allclose(observed,expected,**FORMULA_TOLERANCE):
                raise ValueError('Independent complete source-FP32 Euler/log-prior/density program mismatch')
            ideal=-.5*((x.astype(np.float64)**2)+math.log(2*math.pi)).sum(-1)+state['correction64']
            report['maximum_source_precision_density_absolute_error']=max(report['maximum_source_precision_density_absolute_error'],float(np.max(np.abs(observed-expected))))
            report['maximum_FP64_density_on_source_states_absolute_error']=max(report['maximum_FP64_density_on_source_states_absolute_error'],float(np.max(np.abs(observed-ideal))))
            report['CNF_batches']+=1;position=stop
            return actual
        cloned=FunctionType(function.__code__,dict(function.__globals__,cnf_log_density=audited_cnf),function.__name__,function.__defaults__)
        actual=MethodType(cloned,model)(windows)
        if position!=len(windows):raise ValueError('Every original full DFM scoring batch must be observed')
        report['score_calls']+=1;report['windows']+=len(windows);report['window_points']+=len(windows)*model.window
        return actual
    return score,report
