"""Restore every original neural module without fitting or changing device."""
from __future__ import annotations


def load_checkpoint(path):
    import numpy as np,torch
    # The source GRASP schema contains an explicit NumPy float32 adjacency.
    # Keep weights-only loading with precisely these required ndarray globals.
    allowed=[np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True)


def restore_model(checkpoint,job,config,features):
    import numpy as np,torch
    from scripts.flow_matching.model_registry import build_configured_method
    from iia_benchmark.models.fm_comparator_reconstructions import DTLACore,SHCLTransformerCore,KGLCore
    from iia_benchmark.models.fm_native import GraphSpectralPath,MixerVelocity,TimeMLP
    if (checkpoint['configuration']!=config or checkpoint['overrides']!=job['parameter_overrides']
            or checkpoint['window']!=job['window']):raise ValueError('Every original source model/config/device/seed required')
    model=build_configured_method(config,job['parameter_overrides']);device=torch.device(job['parameter_overrides']['device'])
    entry=config['entrypoint']
    if entry.endswith(':ComparatorWindowDetector'):
        constructors=dict(dt_la=DTLACore,shcl_transformer=SHCLTransformerCore,kgl=KGLCore)
        model._model=constructors[model.method](features,**model.options).to(device);expected={'_model'}
    elif entry.endswith(':GRASPDetector'):
        a=np.asarray(checkpoint['adjacency'])
        if a.dtype!=np.float32 or a.shape!=(features,features) or not np.isfinite(a).all() or not np.array_equal(a,a.T) or np.any(a<0) or np.any(np.diag(a)):
            raise ValueError('Complete source float32 symmetric adjacency required')
        model.adjacency_=a.copy();model.graph_source='train_only_abs_pearson_local_choice'
        # A structural placeholder only;load the actual saved basis,including
        # every tied eigenspace. Never substitute a newly diagonalized basis.
        model.path=GraphSpectralPath(torch.zeros(features,features),tau=0.).to(device)
        model.model=MixerVelocity(features,model.window,model.hidden,model.blocks,model.dropout).to(device)
        expected={'model','path'}
    elif entry.endswith(':DFMDetector'):
        model.forward_field=TimeMLP(job['window']*features,model.hidden).to(device)
        model.reverse_field=TimeMLP(job['window']*features,model.hidden).to(device)
        expected={'forward_field','reverse_field'}
    else:raise ValueError('An explicit original neural restoration schema is required')
    if set(checkpoint['modules'])!=expected or (entry.endswith(':GRASPDetector') is False and checkpoint['adjacency'] is not None):
        raise ValueError('Every original checkpoint module,without extras or omissions,required')
    for name in expected:
        module=getattr(model,name);wanted=checkpoint['modules'][name];actual=module.state_dict()
        if set(actual)!=set(wanted):raise ValueError('Every complete original tensor key required')
        for key,tensor in actual.items():
            if tensor.dtype!=wanted[key].dtype or tensor.shape!=wanted[key].shape or not torch.isfinite(wanted[key]).all():
                raise ValueError('Every original tensor dtype,shape and finite value required')
        module.load_state_dict(wanted,strict=True);module.eval()
        # Source trained parameters keep requires_grad=True,even in no_grad.
    check_state(model,checkpoint)
    return model


def check_state(model,checkpoint):
    import torch
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    modules={name:value for name,value in vars(model).items() if isinstance(value,torch.nn.Module)}
    if set(modules)!=set(checkpoint['modules']):raise ValueError('All complete original neural modules required')
    digests={}
    for name,module in modules.items():
        state=module.state_dict();wanted=checkpoint['modules'][name]
        if set(state)!=set(wanted):raise ValueError('All source parameter/buffer keys must remain')
        if any(not p.requires_grad for p in module.parameters()):raise ValueError('Original source trained parameter flags must remain')
        for key,tensor in state.items():
            value=tensor.detach().cpu()
            if not torch.isfinite(value).all():raise ValueError('Finite source tensors required')
            tensor_bytes(value,wanted[key]);digests[name+'.'+key]=array_digest(value.numpy())
    return digests


def require_tensor_bytes(actual,expected):
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    tensor_bytes(actual.detach().cpu(),expected.detach().cpu())
