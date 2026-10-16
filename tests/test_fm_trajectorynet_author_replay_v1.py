"""Original source/metric semantics; fixtures do not count as model results."""
import ast
import importlib.util
import io
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse
from scipy.optimize import linprog
from sklearn.metrics.pairwise import pairwise_distances

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_original_trajectorynet_replay_helper',ROOT/'src/iia_benchmark/models/trajectorynet_author_replay_v1.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
SOURCE=ROOT/'experiments/runs/fm_foundation_baseline_sources/sources/trajectorynet/original'


def test_released_checkpoint_is_complete_original_model_not_new_seed_or_final_step_proof():
    metadata=s.checkpoint_metadata(SOURCE/'results/fig8_results/checkpt.pth')
    assert metadata['state_tensor_count']==22 and metadata['total_state_numel']==9622
    assert metadata['non_counter_state_numel']==9621 and metadata['growth_weights_present']
    assert not metadata['training_seed_metadata_present'] and not metadata['selected_checkpoint_step_metadata_present']
    assert not metadata['actual_tensor_storage_materialized']


def test_literal_author_arguments_bind_density_only_and_exact_original_budget():
    args=s.logged_arguments((SOURCE/'results/fig8_results/logs').read_text())
    assert args['dataset']=='EB-PCA' and args['leaveout_timepoint']==3
    assert args['top_k_reg']==.1 and args['vecint']==0 and not args['use_growth']
    assert args['niters']==10000 and args['batch_size']==1000 and args['max_dim']==5
    assert args['dims']=='64-64-64' and args['nonlinearity']=='tanh'
    assert args['atol']==args['rtol']==1e-5


def test_nonliteral_namespace_and_pickle_code_execution_are_rejected():
    with pytest.raises(ValueError):s.logged_arguments("Namespace(x=__import__('os').getcwd())")
    with pytest.raises(ValueError):s.logged_arguments('Namespace(**dict(x=1))')
    with pytest.raises(ValueError):s.logged_arguments('Namespace(x=1)\nNamespace(x=2)')
    assert s.logged_arguments('Namespace(x=1)\nNamespace(x=1)')=={'x':1}
    class Unknown:
        def __reduce__(self):return (eval,('1+1',))
    with pytest.raises(ValueError,match='Unapproved'):
        s._MetadataUnpickler(io.BytesIO(pickle.dumps(Unknown()))).load()


def test_original_emd_default_is_W2_and_euclidean_keyword_would_be_sqrt_W1():
    text=(SOURCE/'TrajectoryNet/optimal_transport/emd.py').read_text()
    fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='earth_mover_distance')
    assert ast.literal_eval(fn.args.defaults[-1])=='sqeuclidean'
    def exact_reference_LP(a,b,costs,**kwargs):
        n,m=costs.shape
        A=np.vstack([np.kron(np.eye(n),np.ones((1,m))),np.kron(np.ones((1,n)),np.eye(m))])
        solved=linprog(costs.ravel(),A_eq=A,b_eq=np.r_[a,b],bounds=(0,None),method='highs')
        assert solved.success
        return solved.fun
    namespace=dict(np=np,scipy=scipy,pot=SimpleNamespace(emd2=exact_reference_LP),pairwise_distances=pairwise_distances)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original_EMD_independent_SciPy_LP_fixture>','exec'),namespace)
    p=np.array([[0.,0.],[0.,0.]])
    q=np.array([[0.,0.],[4.,0.]])
    assert math.isclose(namespace['earth_mover_distance'](p,q),math.sqrt(8))
    assert math.isclose(namespace['earth_mover_distance'](p,q,metric='euclidean'),math.sqrt(2))
    assert 2.0!=math.sqrt(8) and 2.0!=math.sqrt(2)  # actual raw W1 is2
