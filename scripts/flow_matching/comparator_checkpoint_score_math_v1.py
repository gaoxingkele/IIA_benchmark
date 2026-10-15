"""Observe original comparator Python frames;verify every score primitive.

The original NN and source score operations remain. Full ideal FP64 differences
are separate diagnostics,not automatically certified source equivalence.
"""
from __future__ import annotations


def checked_score(detector):
    import numpy as np,torch,sys
    from types import FunctionType,MethodType
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import FORMULA_TOLERANCE
    from scripts.flow_matching.neural_full510_checkpoint_state_v1 import require_tensor_bytes
    core=detector._model;method=detector.method
    report=dict(method=method,score_calls=0,windows=0,window_points=0,source_core_score_calls=0,
        gaussian_calls=0,reconstruction_decoder_calls=0,masked_representation_calls=0,
        primitive_checks=0,primitive_coordinates=0,maximum_primitive_absolute_errors={},
        independent_FP64_primitives_and_source_precision_program_checked=True,
        ideal_FP64_complete_score_equivalence_certified=True,ideal_FP64_failure_coordinates=0,
        maximum_ideal_FP64_score_absolute_error=0.,independent_neural_kernel_implementation=False)
    def n(value):
        return value.detach().cpu().numpy().astype(np.float64) if isinstance(value,torch.Tensor) else np.asarray(value,dtype=np.float64)
    def check(actual,expected,label):
        a,b=n(actual),n(expected)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,**FORMULA_TOLERANCE):
            raise ValueError('Independent full comparator primitive mismatch: '+label)
        report['primitive_checks']+=1;report['primitive_coordinates']+=a.size
        report['maximum_primitive_absolute_errors'][label]=max(report['maximum_primitive_absolute_errors'].get(label,0.),float(np.max(np.abs(a-b))))
    def softmax(value):
        value=n(value);value=value-value.max(-1,keepdims=True);value=np.exp(value);return value/value.sum(-1,keepdims=True)
    def gaussian_ideal(me,le,mo,lo):
        me,le,mo,lo=map(n,[me,le,mo,lo]);return .5*(lo-le+(np.exp(le)+(me-mo)**2)/np.exp(lo)-1).sum(-1)

    def score(windows):
        if sys.getprofile() is not None:raise ValueError('A private isolated frame observer is required')
        position=0;actual_batches=[];ideal_batches=[];captured=dict(gaussians=[],decoders=[],representations=[])
        original_values=getattr(core,'_values',None)
        handles=[];original_gaussian=None
        if method=='shcl_transformer':
            function=original_values.__func__;original_gaussian=function.__globals__['gaussian_kl']
            def checked_gaussian(me,le,mo,lo):
                actual=original_gaussian(me,le,mo,lo)
                exp_e,exp_o=le.exp(),lo.exp();check(exp_e,np.exp(n(le)),'Gaussian_exp_even');check(exp_o,np.exp(n(lo)),'Gaussian_exp_odd')
                delta=me-mo;check(delta,n(me)-n(mo),'Gaussian_mean_difference')
                square=delta.square();check(square,n(delta)**2,'Gaussian_mean_square')
                numerator=exp_e+square;check(numerator,n(exp_e)+n(square),'Gaussian_numerator')
                ratio=numerator/exp_o;check(ratio,n(numerator)/n(exp_o),'Gaussian_variance_ratio')
                difference=lo-le;check(difference,n(lo)-n(le),'Gaussian_logvariance_difference')
                added=difference+ratio;check(added,n(difference)+n(ratio),'Gaussian_add')
                component=added-1;check(component,n(added)-1,'Gaussian_subtract_one')
                complete=.5*component.sum(-1);check(complete,.5*n(component).sum(-1),'Gaussian_reduction')
                require_tensor_bytes(actual,complete)
                captured['gaussians'].append((actual,gaussian_ideal(me,le,mo,lo)));report['gaussian_calls']+=1
                return actual
            cloned=FunctionType(function.__code__,dict(function.__globals__,gaussian_kl=checked_gaussian),function.__name__,function.__defaults__)
            core._values=MethodType(cloned,core)
            target=original_values.__func__.__code__
            represent_code=core._represent.__func__.__code__;forward_code=core.forward.__func__.__code__
            if core.objective=='reconstruction':
                for branch in core.branches:
                    def decoder(module,args,out,branch=branch):
                        captured['decoders'].append((branch,args[0],out));report['reconstruction_decoder_calls']+=1
                    handles.append(core.reconstruction[branch].register_forward_hook(decoder))
        else:
            target=core.score.__func__.__code__;represent_code=forward_code=None
            if method=='kgl':
                def prediction(module,args,out):captured['prediction']=(args[0],out)
                handles.append(core.register_forward_hook(prediction))

        def observe(frame,event,arg):
            nonlocal position
            if method=='shcl_transformer' and frame.f_code is represent_code and event=='call':
                values=frame.f_locals;captured['representations'].append((values['branch'],values['h'].detach().clone()))
                report['masked_representation_calls']+=1
            if method=='shcl_transformer' and frame.f_code is forward_code and event=='return':
                values=frame.f_locals;h=values['h'];representations=captured['representations']
                if len(representations)!=2*len(core.branches):raise ValueError('Every source hierarchical branch and complementary input required')
                for index,branch in enumerate(core.branches):
                    if core.masking=='random':
                        even=(torch.rand(h.shape[1],generator=torch.Generator(device='cpu').manual_seed(1103))>=.5).to(h.device)
                    else:
                        indexes=torch.arange(h.shape[1],device=h.device);even=indexes%2==0 if branch=='point' else indexes%4>=2
                    for offset,mask in [(0,even),(1,~even)]:
                        name,actual=representations[2*index+offset]
                        if name!=branch:raise ValueError('Original branch order must remain')
                        require_tensor_bytes(actual,h*mask[None,:,None])
                representations.clear()
            if frame.f_code is not target or event!='return':return
            v=frame.f_locals;x=v['x'];stop=min(position+detector.batch_size,len(windows))
            wanted=torch.from_numpy(np.asarray(windows[position:stop],dtype=np.float32))
            require_tensor_bytes(x,wanted)
            if method=='dt_la':
                inp,lat=v['inp'],v['lat']
                full_inp=np.square(n(x)-n(v['reconstruction'])).sum(-1)
                full_lat=np.square(n(v['z'])-n(v['zhat'])).sum(-1)
                check(inp,full_inp,'DTLA_input_squared_sum');check(lat,full_lat,'DTLA_latent_squared_sum')
                if core.score_mode=='input':expected=n(inp);ideal=full_inp
                elif core.score_mode=='latent':expected=n(lat);ideal=full_lat
                else:
                    source_weight=v['weights'];weights=softmax(lat);ideal_weight=softmax(full_lat)
                    if core.score_mode=='scaled_softmax':weights=weights*np.abs(n(lat));ideal_weight=ideal_weight*np.abs(full_lat)
                    check(source_weight,weights,'DTLA_softmax_and_optional_magnitude')
                    expected=n(inp)*n(source_weight);ideal=full_inp*ideal_weight
            elif method=='kgl':
                inputs,prediction=captured.pop('prediction');require_tensor_bytes(inputs,x[:,:-1])
                error=np.square(n(prediction)-n(x[:,1:])).mean(-1);check(v['scores'],error,'KGL_next_step_MSE')
                expected=np.pad(n(v['scores']),((0,0),(1,0)));ideal=np.pad(error,((0,0),(1,0)))
            else:
                outputs=v['outputs'];program=torch.zeros(x.shape[:2],device=x.device);ideal=np.zeros(x.shape[:2],dtype=np.float64)
                if list(outputs)!=list(core.branches):raise ValueError('Every original SHCL branch required')
                for index,(branch,(even,odd)) in enumerate(outputs.items()):
                    if core.objective=='reconstruction':
                        if len(captured['decoders'])!=2*len(core.branches):raise ValueError('Both reconstruction decoders per branch required')
                        first,second=captured['decoders'][2*index:2*index+2]
                        for record,values in [(first,even),(second,odd)]:
                            if record[0]!=branch:raise ValueError('Original reconstruction branch ordering required')
                            require_tensor_bytes(record[1],values[0])
                        e=(first[2]-x).square().mean(-1);o=(second[2]-x).square().mean(-1)
                        ie=np.square(n(first[2])-n(x)).mean(-1);io=np.square(n(second[2])-n(x)).mean(-1)
                        check(e,ie,'SHCL_even_reconstruction_MSE');check(o,io,'SHCL_odd_reconstruction_MSE')
                        program=program+.5*(e+o);ideal=ideal+.5*(ie+io)
                    else:
                        if len(captured['gaussians'])!=len(core.branches):raise ValueError('Every directed branch Gaussian KL required')
                        source_KL,ideal_KL=captured['gaussians'][index]
                        error=(even[0]-odd[0]).square().mean(-1)
                        full_error=np.square(n(even[0])-n(odd[0])).mean(-1);check(error,full_error,'SHCL_heterogeneity_MSE')
                        program=program+error+source_KL;ideal=ideal+full_error+ideal_KL
                require_tensor_bytes(arg,program);expected=n(program)
                captured['gaussians'].clear();captured['decoders'].clear()
            check(arg,expected,'complete_source_precision_score_program')
            if tuple(arg.shape)!=tuple(x.shape[:2]):raise ValueError('Complete per-window timestamp score required')
            failures=int(np.count_nonzero(~np.isclose(n(arg),ideal,**FORMULA_TOLERANCE)))
            report['ideal_FP64_failure_coordinates']+=failures
            if failures:report['ideal_FP64_complete_score_equivalence_certified']=False
            report['maximum_ideal_FP64_score_absolute_error']=max(report['maximum_ideal_FP64_score_absolute_error'],float(np.max(np.abs(n(arg)-ideal))))
            actual_batches.append(arg.detach().cpu().numpy().copy());ideal_batches.append(ideal)
            position=stop;report['source_core_score_calls']+=1
        try:
            sys.setprofile(observe);actual=detector.score(windows)
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            if method=='shcl_transformer':del core._values
        expected=np.concatenate(actual_batches) if actual_batches else np.empty(0)
        if (position!=len(windows) or actual.dtype!=expected.dtype or actual.shape!=expected.shape
                or actual.tobytes()!=expected.tobytes() or captured['gaussians'] or captured['decoders'] or captured['representations']):
            raise ValueError('Every original full comparator scoring batch must be observed and returned unchanged')
        report['score_calls']+=1;report['windows']+=len(windows);report['window_points']+=len(windows)*windows.shape[1]
        return actual
    return score,report
