"""Independent NumPy path equations and every Monte Carlo scoring coordinate."""
import math
from types import FunctionType, MethodType

TOLERANCE=dict(rtol=2e-6,atol=2e-6)


def path_numpy(x0,x1,t,noise,objective,sigma,precision='float64'):
    import numpy as np
    dtype=np.float64 if precision=='float64' else np.float32 if precision=='float32' else None
    if dtype is None:raise ValueError('Explicit numerical path precision required')
    a,b,e=[np.asarray(v,dtype=dtype) for v in [x0,x1,noise]]
    time=np.asarray(t,dtype=dtype).reshape((-1,)+(1,)*(a.ndim-1))
    one,two=dtype(1),dtype(2)
    if objective=='target':
        scale=dtype(1-sigma);std=one-scale*time;x=time*b+std*e;v=(b-scale*x)/std
    elif objective in ['sb','sf2m']:
        mean=(one-time)*a+time*b;std=dtype(sigma)*np.sqrt(time*(one-time));x=mean+std*e
        v=b-a+(one-two*time)/(two*time*(one-time)+dtype(1e-8))*(x-mean)
    elif objective=='vp':
        angle=time*dtype(math.pi)/two;std=np.full_like(time,sigma)
        x=np.cos(angle)*a+np.sin(angle)*b+std*e;v=dtype(math.pi/2)*(np.cos(angle)*b-np.sin(angle)*a)
    elif objective in ['independent','ot','rectified']:
        std=np.full_like(time,0 if objective=='rectified' else sigma);x=(one-time)*a+time*b+std*e;v=b-a
    else:raise ValueError('Unknown frozen FM objective')
    return x,v,std


def checked_score(model):
    """Clone only the fresh instance's score globals; never patch a live module."""
    import numpy as np
    function=model.score.__func__;original_path=function.__globals__['probability_path']
    report=dict(score_calls=0,path_calls=0,forward_calls=0,windows=0,window_points=0,
        source_FP32_path_program_checked=True,ideal_float64_path_allclose_passed=True,
        float64_path_failure_coordinate_counts=dict(state=0,velocity=0,std=0),
        maximum_path_absolute_errors=dict(state=0.,velocity=0.,std=0.),
        maximum_FP32_program_absolute_errors=dict(state=0.,velocity=0.,std=0.),maximum_score_absolute_error=0.)
    def score(windows):
        pending=[];completed=[];residual=[None];contributions=[0]
        dimensions=int(np.prod(model._shape));repeats=model.monte_carlo_samples*len(model.score_times)
        def path(a,b,t,noise,objective,sigma):
            actual=original_path(a,b,t,noise,objective,sigma)
            inputs=[v.detach().cpu().numpy() for v in [a,b,t,noise]]
            independent=path_numpy(*inputs,objective,sigma)
            finite_precision=path_numpy(*inputs,objective,sigma,precision='float32')
            for name,x,y,z in zip(['state','velocity','std'],actual,independent,finite_precision):
                values=x.detach().cpu().numpy()
                if (values.shape!=y.shape or not np.isfinite(values).all() or not np.isfinite(y).all()
                        or not np.allclose(values,z,**TOLERANCE)):
                    raise ValueError('Independent source-FP32 probability path program mismatch: '+name)
                count=int(np.count_nonzero(~np.isclose(values,y,**TOLERANCE)))
                report['float64_path_failure_coordinate_counts'][name]+=count
                if count:report['ideal_float64_path_allclose_passed']=False
                report['maximum_path_absolute_errors'][name]=max(report['maximum_path_absolute_errors'][name],float(np.max(np.abs(values-y))))
                report['maximum_FP32_program_absolute_errors'][name]=max(report['maximum_FP32_program_absolute_errors'][name],float(np.max(np.abs(values-z))))
            pending.append(actual[1].detach().cpu().numpy().astype(np.float64));report['path_calls']+=1
            return actual
        def forward(module,args,out):
            if len(pending)!=1:raise ValueError('Every source score forward needs exactly one checked path')
            velocity=pending.pop();prediction=out[:,:dimensions].detach().cpu().numpy().astype(np.float64)
            error=np.square(prediction-velocity)
            if residual[0] is None:residual[0]=np.zeros_like(error)
            if residual[0].shape!=error.shape:raise ValueError('Source scoring batch partition changed')
            residual[0]+=error;contributions[0]+=1;report['forward_calls']+=1
            if contributions[0]==repeats:
                completed.append((residual[0]/repeats).reshape((-1,)+model._shape).mean(axis=-1))
                report['windows']+=len(error);report['window_points']+=len(error)*model.window
                residual[0]=None;contributions[0]=0
        cloned=FunctionType(function.__code__,dict(function.__globals__,probability_path=path),function.__name__,function.__defaults__)
        hook=model._model.register_forward_hook(forward)
        try:actual=MethodType(cloned,model)(windows)
        finally:hook.remove()
        expected=np.concatenate(completed) if completed else np.empty((0,model.window))
        if (pending or contributions[0] or actual.shape!=expected.shape or not np.isfinite(actual).all()
                or not np.allclose(actual,expected,**TOLERANCE)):
            raise ValueError('Independent complete Monte Carlo residual score mismatch')
        if actual.size:report['maximum_score_absolute_error']=max(report['maximum_score_absolute_error'],float(np.max(np.abs(actual-expected))))
        report['score_calls']+=1
        return actual
    return score,report
