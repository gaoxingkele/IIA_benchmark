"""Observe complete CFM-TS reconstruction epochs and restore their exact state.

Paper reference: https://openreview.net/forum?id=Hqn4Aj7xrQ. The frozen local
paper-derived model/solver is retained; unavailable author equivalence is not
claimed. Saving state adds no optimization or sampling operations.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
import random
import textwrap
from types import MethodType

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT, read, sha


EXECUTION_FIELDS = {'id', 'output_directory', 'artifact_directory', 'recovery_original_job_id'}


def canonical_science(job):
    return {k: v for k, v in job.items() if k not in EXECUTION_FIELDS}


def bitwise_state(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(bitwise_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) == type(right) and len(left) == len(right) and all(bitwise_state(a, b) for a, b in zip(left, right))
    return type(left) == type(right) and left == right


def backend_policy(torch):
    return dict(cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        default_dtype=str(torch.get_default_dtype()), cuda_context_initialized=torch.cuda.is_initialized())


def finite_tensor_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensor_tree(v) for v in value.values())
    if isinstance(value, (tuple,list)):
        return all(finite_tensor_tree(v) for v in value)
    return True


def gradients(model):
    return {name: None if p.grad is None else p.grad.detach().clone() for name,p in model.named_parameters()}


def optimizer_steps_match(state, expected):
    return bool(state['state']) and all('step' in value and float(value['step'])==expected for value in state['state'].values())


def tensor_identity(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def full_fit_data_identity(prepared, node_data):
    return dict(prepared=None if prepared is None else [tensor_identity(t) for t in prepared],
        node_data=None if node_data is None else [[tensor_identity(t), tensor_identity(y)] for t, y in node_data])


def observed_fit_ast(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    forks = [n for n in function.body if isinstance(n, ast.With)]
    if len(forks) != 1:
        raise ValueError('Exactly the original training RNG fork is required')
    position = function.body.index(forks[0])
    function.body.insert(position, ast.parse('_epoch_lane.enter_fork(self)').body[0])
    fork = forks[0]
    loops = [n for n in fork.body if isinstance(n, ast.For) and getattr(n.target, 'id', None) == 'epoch']
    if len(loops) != 1 or ast.dump(loops[0].iter) != ast.dump(ast.parse('range(self.epochs)', mode='eval').body):
        raise ValueError('Preserve the full original training loop')
    index = fork.body.index(loops[0])
    fork.body.insert(index, ast.parse('_first_epoch = _epoch_lane.restore_epoch(self, optimizer, prepared, node_data, count)').body[0])
    loops[0].iter = ast.parse('range(_first_epoch, self.epochs)', mode='eval').body
    loops[0].body.append(ast.parse('_epoch_lane.save_epoch(self, optimizer, epoch + 1, prepared, node_data, count)').body[0])
    return ast.fix_missing_locations(tree)


class CompleteEpochLane:
    def __init__(self, job, artifacts, interval=10, resume=None, resume_sha256=None):
        self.job = deepcopy(job)
        self.artifacts = Path(artifacts)
        self.interval = interval
        self.resume = Path(resume) if resume else None
        self.resume_sha256 = resume_sha256
        self.restored = None
        self.last_optimizer_state = None

    def enter_fork(self, model):
        import torch
        if torch.cuda.is_initialized():
            raise ValueError('The original CPU-only fit must not initialize CUDA')
        if self.resume is not None:
            if not self.resume_sha256 or sha(self.resume) != self.resume_sha256:
                raise ValueError('Only an explicitly hashed complete epoch checkpoint may be resumed')
            self.restored = torch.load(self.resume, map_location='cpu', weights_only=False)
            if (canonical_science(self.restored['job']) != canonical_science(self.job)
                    or not self.restored['complete_original_epoch_boundary']
                    or self.restored['required_epochs'] != model.epochs
                    or self.restored['backend_policy'] != backend_policy(torch)):
                raise ValueError('Checkpoint scientific identity, full budget or backend differs')
            torch.set_rng_state(self.restored['outer_torch_cpu_rng'])
        self.outer_rng = torch.get_rng_state().clone()

    def restore_epoch(self, model, optimizer, prepared, node_data, count):
        import numpy as np
        import torch
        self.fit_identity = full_fit_data_identity(prepared, node_data)
        if self.restored is None:
            return 0
        state = self.restored
        epoch = state['completed_epochs']
        if (not 0 < epoch <= model.epochs or state['training_items'] != count
                or state['optimizer_updates'] != epoch * math.ceil(count / model.batch_size)
                or not optimizer_steps_match(state['optimizer_state'],state['optimizer_updates'])
                or not state['model_training'] or state['model_nfe']<0
                or not finite_tensor_tree((state['model_state'],state['optimizer_state'],state['parameter_gradients']))
                or len(state['loss_history']) != epoch or state['fit_data_identity'] != self.fit_identity):
            raise ValueError('Only complete original epochs with identical full data/order may resume')
        model.model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        model.loss_history_ = list(state['loss_history'])
        model.optimizer_updates_ = state['optimizer_updates']
        model.preprocessing_ = deepcopy(state['preprocessing'])
        model.model.nfe = state['model_nfe']
        model.model.train(state['model_training'])
        for name, parameter in model.model.named_parameters():
            grad = state['parameter_gradients'][name]
            parameter.grad = None if grad is None else grad.clone()
        torch.set_rng_state(state['torch_cpu_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if (not bitwise_state(model.model.state_dict(), state['model_state'])
                or not bitwise_state(optimizer.state_dict(), state['optimizer_state'])
                or not bitwise_state(gradients(model.model),state['parameter_gradients'])
                or not bitwise_state(torch.get_rng_state(), state['torch_cpu_rng'])
                or not bitwise_state(np.random.get_state(), state['numpy_rng'])
                or random.getstate() != state['python_rng']):
            raise ValueError('Every model/optimizer and RNG state must restore bitwise')
        from scripts.flow_matching.run_cfm_ts_original_job import write
        write(self.artifacts/'complete_epoch_restore_verification.json', dict(completed_original_epochs=epoch,
            remaining_original_epochs=model.epochs-epoch, required_original_epochs=model.epochs,
            full_model_optimizer_CPU_numpy_python_rng_and_order_restored_bitwise=True,
            additional_optimizer_updates=0, benchmark_performance=False))
        return epoch

    def save_epoch(self, model, optimizer, epoch, prepared, node_data, count):
        import numpy as np
        import torch
        from scripts.flow_matching.run_cfm_ts_original_job import write
        if epoch % self.interval and epoch != model.epochs:
            return
        if (len(model.loss_history_) != epoch or model.optimizer_updates_ != epoch * math.ceil(count/model.batch_size)
                or full_fit_data_identity(prepared, node_data) != self.fit_identity):
            raise ValueError('Do not persist a partial epoch or altered training data')
        state = dict(job=self.job, completed_epochs=epoch, required_epochs=model.epochs,
            optimizer_updates=model.optimizer_updates_, training_items=count,
            model_state=deepcopy(model.model.state_dict()), optimizer_state=deepcopy(optimizer.state_dict()),
            parameter_gradients=gradients(model.model),
            model_training=model.model.training, model_nfe=model.model.nfe,
            loss_history=list(model.loss_history_), preprocessing=deepcopy(model.preprocessing_),
            torch_cpu_rng=torch.get_rng_state().clone(), outer_torch_cpu_rng=self.outer_rng.clone(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            backend_policy=backend_policy(torch), fit_data_identity=self.fit_identity,
            complete_original_epoch_boundary=True, additional_optimizer_updates=0,
            benchmark_performance=False, author_equivalence_certified=False)
        if (not finite_tensor_tree((state['model_state'],state['optimizer_state'],state['parameter_gradients']))
                or not optimizer_steps_match(state['optimizer_state'],model.optimizer_updates_)):
            raise ValueError('Nonfinite model, optimizer or gradient state cannot certify recovery')
        directory=self.artifacts/'complete_epoch_checkpoints'/('epoch_%04d'%epoch)
        directory.mkdir(parents=True,exist_ok=False)
        path=directory/'full_state.pt';torch.save(state,path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if (not bitwise_state(state,saved) or not bitwise_state(model.model.state_dict(),state['model_state'])
                or not bitwise_state(optimizer.state_dict(),state['optimizer_state'])
                or not bitwise_state(gradients(model.model),state['parameter_gradients'])
                or model.optimizer_updates_!=state['optimizer_updates'] or model.loss_history_!=state['loss_history']
                or model.model.nfe!=state['model_nfe'] or model.model.training!=state['model_training']
                or not bitwise_state(torch.get_rng_state(),state['torch_cpu_rng'])
                or not bitwise_state(np.random.get_state(),state['numpy_rng'])
                or random.getstate()!=state['python_rng'] or backend_policy(torch)!=state['backend_policy']):
            raise ValueError('Checkpoint IO must preserve every native state and counter bitwise')
        record=dict(completed_original_epochs=epoch,required_original_epochs=model.epochs,
            optimizer_updates=model.optimizer_updates_,full_state_path=str(path),full_state_sha256=sha(path),
            complete_original_epoch_boundary=True,all_native_model_optimizer_rng_gradients_history_and_order_values_reread_bitwise=True,
            no_training_state_or_random_state_changed_by_save=True,additional_optimizer_updates=0,
            benchmark_performance=False,author_equivalence_certified=False)
        write(directory/'receipt.json',record)
        ledger=self.artifacts/'completed_full_epoch_checkpoints.json'
        records=read(ledger) if ledger.exists() else []
        if records and records[-1]['completed_original_epochs']>=epoch:
            raise ValueError('Preserve all previous full checkpoint entries')
        records.append(record);write(ledger,records)
        self.last_optimizer_state=deepcopy(optimizer.state_dict())


def bind_fit(model, lane):
    module=__import__(type(model).__module__,fromlist=['CFMTSPaperTrajectory'])
    namespace=dict(module.__dict__,_epoch_lane=lane)
    exec(compile(observed_fit_ast(inspect.getsource(type(model).fit)),module.__file__,'exec'),namespace)
    return MethodType(namespace['fit'],model)
