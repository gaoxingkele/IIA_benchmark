"""Execute pinned original regularizers against analytic values and gradients.

This is an implementation conformance gate, never a trained benchmark result.
No historical EB Jacobian coefficient is inferred from analytic fixtures.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

CONFIG='configs/reproducibility/fm_TrajectoryNet_RNODE_regularizer_semantics.v1.json'
MODEL='configs/models/fm_TrajectoryNet_RNODE_regularizer_semantics.v1.json'
MODULE='scripts.flow_matching.audit_trajectorynet_rnode_regularizer_semantics_v1'
TEST='tests/test_fm_trajectorynet_rnode_regularizer_semantics_v1.py'

def registration():
    acquisition=read(ROOT/'configs/acquisition/fm_TrajectoryNet_author_checkpoint_materials.v1.json')
    rnode=read(ROOT/'configs/models/fm_RegCNF_RNODE_author_source.v1.json')
    files=dict(trajectorynet=str(ROOT/acquisition['source_root']/'TrajectoryNet/lib/layers/wrappers/cnf_regularization.py'),
        rnode=str(Path(rnode['source_root'])/'lib/layers/wrappers/cnf_regularization.py'))
    pins=[MODEL,TEST,MODULE.replace('.','/')+'.py',
        'configs/acquisition/fm_TrajectoryNet_author_checkpoint_materials.v1.json',
        'configs/models/fm_RegCNF_RNODE_author_source.v1.json']
    return dict(original_regularizers=files,original_sha256={k:sha(v) for k,v in files.items()},
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        dimensions=[2,5,25],full_fixture_batch=1000,dtype='float64',optimizer_updates=0,
        artifact_root='F:/aicoding/IIA_Data/experiments/TrajectoryNet_RNODE_regularizer_semantics_v1',
        actual_full_EB_training_complete=False,EB_Jacobian_coefficient_certified=False,
        historical_RegCNF_EB_equivalence_certified=False,benchmark_performance_claim=False,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702))

def verify(c):
    if c!=registration():raise ValueError('Pinned original source, dimensions and all conformance checks required')

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

def compare(c,dim):
    import torch
    tn=load(c['original_regularizers']['trajectorynet'],'_original_trajectorynet_regularizers')
    rn=load(c['original_regularizers']['rnode'],'_original_rnode_regularizers')
    # A dense, non-isotropic, nonzero Jacobian distinguishes square/RMS scaling.
    batch=c['full_fixture_batch'];x=torch.linspace(-1,1,batch*dim,dtype=torch.float64).reshape(batch,dim).requires_grad_()
    matrix=(torch.arange(1,dim*dim+1,dtype=torch.float64).reshape(dim,dim)/(dim*dim)).requires_grad_()
    t=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    dx=x@matrix.T+t
    energy_tn=tn.squared_l2_regularization_fn(x,None,dx,None,t,None)
    energy_rn=rn.quadratic_cost(x,t,None,dx,None,None).mean()
    expected_energy=dx.square().sum(dim=1).mean()
    torch.testing.assert_close(energy_tn,expected_energy,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(energy_tn,energy_rn*(2*dim),rtol=1e-12,atol=1e-12)
    energy_tn_gradient=torch.autograd.grad(energy_tn,matrix,retain_graph=True)[0]
    energy_rn_gradient=torch.autograd.grad(energy_rn,matrix,retain_graph=True)[0]
    torch.testing.assert_close(energy_tn_gradient,2*dx.T@x/batch,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(energy_tn_gradient,energy_rn_gradient*(2*dim),rtol=1e-12,atol=1e-12)
    context=type('SharedContext',(),{})
    jacobian_tn=tn.jacobian_frobenius_regularization_fn(x,None,dx,None,t,context)
    expected_jacobian=matrix.square().sum().sqrt()/dim
    torch.testing.assert_close(jacobian_tn,expected_jacobian,rtol=1e-12,atol=1e-12)
    jacobian_gradient=torch.autograd.grad(jacobian_tn,matrix,retain_graph=True)[0]
    expected_gradient=matrix/(dim*matrix.square().sum().sqrt())
    torch.testing.assert_close(jacobian_gradient,expected_gradient,rtol=1e-12,atol=1e-12)
    # The original RNODE callback must preserve the ODE context estimator tensor;
    # supplying a fresh exact norm here would silently change its stochastic loss.
    estimator=torch.linspace(.2,1,batch,dtype=torch.float64,requires_grad=True)
    rncontext=type('NativeEstimatorContext',(),{'sqjacnorm':estimator})()
    returned=rn.jacobian_frobenius_regularization_fn(x,t,None,dx,None,rncontext)
    assert returned is estimator
    torch.testing.assert_close(torch.autograd.grad(returned.mean(),estimator)[0],torch.full_like(estimator,1/batch))
    return dict(dimensions=dim,batch=batch,trajectorynet_energy=float(energy_tn.detach()),
        rnode_mean_kinetic=float(energy_rn.detach()),energy_value_and_gradient_ratio=2*dim,
        trajectorynet_Jacobian_RMS=float(jacobian_tn.detach()),
        trajectorynet_Jacobian_is_squared_Frobenius=False,original_RNODE_context_tensor_identity_preserved=True,
        original_energy_value_and_gradient_closed_forms_passed=True,
        original_TrajectoryNet_Jacobian_value_and_gradient_closed_forms_passed=True,
        optimizer_updates=0,analytic_fixture_not_dataset_experiment=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');a=parser.parse_args()
    from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    import torch
    torch.set_num_threads(2)
    records=[compare(c,dim) for dim in c['dimensions']]
    path=Path(c['artifact_root']);path.mkdir(exist_ok=False)
    exclusive(path/'original_regularizer_value_and_gradient_conformance.json',dict(passed=True,records=records,
        configuration_sha256=sha(ROOT/CONFIG),torch_version=torch.__version__,
        original_source_files_modified=False,optimizer_updates=0,full_EB_training_complete=False,
        EB_Jacobian_coefficient_certified=False,historical_environment_certified=False,
        benchmark_performance_claim=False,all_original_experiments_complete=False))
    print(json.dumps(dict(records=records,report=str(path/'original_regularizer_value_and_gradient_conformance.json'))))

if __name__=='__main__':main()
