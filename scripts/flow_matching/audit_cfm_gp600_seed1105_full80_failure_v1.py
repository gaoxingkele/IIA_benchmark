"""One unchanged-solver full80 observation of actual GP600 seed1105 failure.

Preserve the full375000-update checkpoint and the original failed scoring at67
trajectories. No retraining or repeated attempts until pass. The original
observer/native simulate worker changes only its complete checkpoint update
binding; every original trajectory and failed grid stays explicit.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import ast
import inspect
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import audit_cfm_sine600_full80_failure_observation_v1 as previous

ROOT,read,sha,write=previous.ROOT,previous.read,previous.sha,previous.write
MODULE='scripts.flow_matching.audit_cfm_gp600_seed1105_full80_failure_v1'
IDENTITY='cfm_ts_original__appendix_d__gp_paper_literal_e600__sine__seed1105'


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen actual GP failure evidence changed: '+pin['path'])
    queue=read(ROOT/config['original_queue'])
    job=next(j for j in queue['jobs'] if j['id']==IDENTITY)
    model=read(ROOT/job['model_config'])
    guard=read(ROOT/config['guard_reference_configuration'])
    for key in ['settings','resource_lock','python','cpu_threads','environment_lock','original_scope']:
        if config[key]!=guard[key]:raise ValueError('Original full readonly guard changed: '+key)
    if (config['canonical_job']!=job or job['epochs']!=600 or job['expected_optimizer_updates']!=375000
            or job['test_trajectories']!=80 or job['test_points_per_trajectory']!=500
            or job['observed_dimensions']!=1 or job['method']!='gp'):
        raise ValueError('Exact actual full GP600/375000 original canonical failure required')
    receipt=read(ROOT/config['original_failed_receipt'])
    progress=read(ROOT/config['original_failed_progress'])
    if (receipt['exit_code']!=1 or receipt['complete'] or receipt['resource_observations']['resource_guard_aborted']
            or progress['phase']!='scoring' or progress['test_trajectories_completed']!=67
            or progress['required_test_trajectories']!=80
            or 'AssertionError: underflow in dt 0.0' not in (ROOT/config['original_failed_stderr']).read_text(encoding='utf-8')
            or (ROOT/job['output_directory']/'result.json').exists()):
        raise ValueError('Original numerical scoring failure, never a resource abort or completed score, required')
    if config['solver_parameters']!={k:model['parameters'][k] for k in ['solver','rtol','atol','max_num_steps']}:
        raise ValueError('Original solver tolerances and maximum steps required')
    if sha(config['checkpoint_path'])!=config['checkpoint_sha256']:
        raise ValueError('Saved complete trained checkpoint changed')
    return job,model


def worker_ast():
    tree=ast.parse(inspect.getsource(previous.worker))
    class CompleteUpdateBinding(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if node.value==37800:
                self.count+=1
                return ast.parse("job['expected_optimizer_updates']",mode='eval').body
            return node
    change=CompleteUpdateBinding();tree=change.visit(tree)
    if change.count!=1:raise ValueError('Only the actual original full optimizer budget binding may change')
    return ast.fix_missing_locations(tree)


def namespace():
    values=dict(previous.__dict__,verify=verify,__name__=MODULE)
    exec(compile(worker_ast(),__file__,'exec'),values)
    return values


def main():
    FunctionType(previous.main.__code__,namespace())()


def post_audit(config):
    from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as arrays
    native=SimpleNamespace(ROOT=ROOT,read=read,sha=sha,verify=verify,validity_summary=previous.validity_summary)
    return FunctionType(arrays.audit.__code__,dict(arrays.audit.__globals__,native=native))(config)


if __name__=='__main__':main()
