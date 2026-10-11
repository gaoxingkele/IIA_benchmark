"""Long Windows paths and native activation recomputation; no reduced training budget."""
import json
import os
from pathlib import Path
import uuid


def native_path(path):
    path = Path(path)
    if os.name != 'nt': return path
    value = os.path.abspath(path)
    if value.startswith('\\\\?\\'): return Path(value)
    if value.startswith('\\\\'): return Path('\\\\?\\UNC\\' + value[2:])
    return Path('\\\\?\\' + value)


def write_json(path, value):
    path = Path(path); native_path(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    native_path(temporary).write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(native_path(temporary), native_path(path))


def save_checkpoint(path, value, torch):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with native_path(temporary).open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    if native_path(path).exists():
        previous = path.with_name('resume_previous.pt')
        previous_temp = path.with_name('resume_previous.' + uuid.uuid4().hex + '.tmp')
        os.link(native_path(path), native_path(previous_temp))
        os.replace(native_path(previous_temp), native_path(previous))
    os.replace(native_path(temporary), native_path(path))


def enable_native_activation_recomputation(model):
    """Enable the original Meta block option, retaining original checkpoint code/RNG."""
    import torch.utils.checkpoint
    before = {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}
    counts = dict(ResBlock=0, AttentionBlock=0)
    for module in model.modules():
        name = type(module).__name__
        if name in counts and type(module).__module__.startswith('_iia_fm_unet_'):
            if not hasattr(module, 'use_checkpoint'):
                raise ValueError('Native checkpoint option missing')
            module.use_checkpoint = True; counts[name] += 1
    if any(value == 0 for value in counts.values()):
        raise ValueError('Original complete Meta residual and attention blocks required')
    if hasattr(model, 'model') and hasattr(model.model, 'use_checkpoint'):
        model.model.use_checkpoint = True
    after = {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}
    if before != after: raise ValueError('Activation recomputation must not change architecture/state dictionary')
    return counts
