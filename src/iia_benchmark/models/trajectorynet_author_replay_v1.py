"""Read original TrajectoryNet metadata safely and construct its actual model."""
import ast
import collections
import importlib
import inspect
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace


def logged_arguments(text):
    lines=sorted(set(line for line in text.splitlines() if line.startswith('Namespace(')))
    if len(lines)!=1:raise ValueError('Exactly one unique original logged Namespace required')
    call=ast.parse(lines[0],mode='eval').body
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='Namespace' or call.args:
        raise ValueError('Only the literal logged Namespace may be parsed')
    if any(k.arg is None for k in call.keywords):raise ValueError('Namespace expansion is not literal metadata')
    return {k.arg:ast.literal_eval(k.value) for k in call.keywords}


class _StorageToken:
    pass


def _tensor_descriptor(storage,offset,size,stride,*unused):
    return dict(shape=list(size),offset=offset,stride=list(stride),storage=storage)


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if module=='torch' and name in ['FloatStorage','DoubleStorage','LongStorage','IntStorage']:return _StorageToken
        if module=='torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:return _tensor_descriptor
        raise ValueError('Unapproved checkpoint pickle global: '+module+'.'+name)

    def persistent_load(self,value):
        if value[0]!='storage':raise ValueError('Only tensor storage metadata is permitted')
        return dict(storage_key=value[2],location=value[3],numel=value[4])


def checkpoint_metadata(path):
    with Path(path).open('rb') as stream:
        magic=_MetadataUnpickler(stream).load()
        protocol=_MetadataUnpickler(stream).load()
        _MetadataUnpickler(stream).load()
        if magic!=0x1950a86a20f9469cfc6c or protocol!=1001:
            raise ValueError('Require original legacy Torch checkpoint container')
        obj=_MetadataUnpickler(stream).load()
    if not isinstance(obj,dict) or set(obj)-{'state_dict','growth_state_dict'} or 'state_dict' not in obj:
        raise ValueError('Only original tensor state dictionaries are permitted')
    state=obj['state_dict']
    return dict(top_level_keys=list(obj),state_tensor_count=len(state),
        state_shapes={k:v['shape'] for k,v in state.items()},
        total_state_numel=sum(math.prod(v['shape']) for v in state.values()),
        non_counter_state_numel=sum(math.prod(v['shape']) for k,v in state.items() if '_num_evals' not in k),
        growth_weights_present='growth_state_dict' in obj,
        training_seed_metadata_present=False,selected_checkpoint_step_metadata_present=False,
        actual_tensor_storage_materialized=False)


def native_model(source_directory,overlay_directory,checkpoint,parameters):
    root,overlay=Path(source_directory).resolve(),Path(overlay_directory).resolve()
    if 'torchdiffeq' in sys.modules and not Path(sys.modules['torchdiffeq'].__file__).resolve().is_relative_to(overlay):
        raise ValueError('Never silently reuse the newer active-core torchdiffeq runtime')
    sys.path.insert(0,str(overlay))
    sys.path.insert(0,str(root))
    import torch
    import torchdiffeq
    if not Path(torchdiffeq.__file__).resolve().is_relative_to(overlay):
        raise ValueError('Original pinned torchdiffeq0.0.1 overlay required')
    native=importlib.import_module('TrajectoryNet.train_misc')
    if not Path(inspect.getfile(native.build_model_tabular)).resolve().is_relative_to(root):
        raise ValueError('Pinned original TrajectoryNet model builder required')
    args=SimpleNamespace(**parameters)
    functions,_=native.create_regularization_fns(args)
    model=native.build_model_tabular(args,5,functions).to('cpu')
    state=torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(state['state_dict'],strict=True)
    if sum(p.numel() for p in model.parameters())!=9621:
        raise ValueError('Complete original concat-squash64x3/Tanh model required')
    return torch,args,model


def native_EB_data(source_directory,path):
    # Bypass only EBData.__init__'s CWD-relative filename. The original .load
    # implements all normalization,velocity,ordering and dimension truncation.
    root=Path(source_directory).resolve()
    sys.path.insert(0,str(root))
    native=importlib.import_module('TrajectoryNet.dataset')
    data=native.EBData.__new__(native.EBData)
    native.SCData.__init__(data)
    data.embedding_name='pcs'
    data.use_velocity=True
    data.load(str(path),5)
    return data
