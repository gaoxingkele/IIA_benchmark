"""Observe all20 original main-text GP600 seed1104 held-out trajectories once.

Restore the saved600/93600-update model without retraining. Keep the original
solver, observer, resource limits and every failed grid. Adapt only original
coverage counts and diagnostic artifact names from the established full80 pass.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import ast
import inspect
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import audit_cfm_gp600_seed1105_full80_failure_v1 as previous

ROOT,read,sha,write=previous.ROOT,previous.read,previous.sha,previous.write
MODULE='scripts.flow_matching.audit_cfm_main_GP6001104_full20_failure_v1'
IDENTITY='cfm_ts_original__main_text__gp_paper_literal_e600__sine__seed1104'


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen actual main-text failure changed: '+pin['path'])
    queue=read(ROOT/config['original_queue']);job=next(j for j in queue['jobs'] if j['id']==IDENTITY)
    model=read(ROOT/job['model_config']);guard=read(ROOT/config['guard_reference_configuration'])
    for key in ['settings','resource_lock','python','cpu_threads','environment_lock','original_scope']:
        if config[key]!=guard[key]:raise ValueError('Unchanged original readonly limits required: '+key)
    if (config['canonical_job']!=job or job['epochs']!=600 or job['expected_optimizer_updates']!=93600
            or job['test_trajectories']!=20 or job['test_points_per_trajectory']!=500
            or job['observed_dimensions']!=1 or job['method']!='gp' or job['batch_size']!=258):
        raise ValueError('Exact main-text GP600/93600 and all20 original grids required')
    receipt=read(ROOT/config['original_failed_receipt']);progress=read(ROOT/config['original_failed_progress'])
    if (receipt['exit_code']!=1 or receipt['complete'] or receipt['resource_observations']['resource_guard_aborted']
            or progress['phase']!='scoring' or progress['test_trajectories_completed']!=12
            or progress['required_test_trajectories']!=20
            or 'AssertionError: underflow in dt 0.0' not in (ROOT/config['original_failed_stderr']).read_text(encoding='utf-8')
            or (ROOT/job['output_directory']/'result.json').exists()):
        raise ValueError('Actual original numerical failure at12/20 must remain preserved')
    if config['solver_parameters']!={k:model['parameters'][k] for k in ['solver','rtol','atol','max_num_steps']}:
        raise ValueError('Original solver settings required')
    if sha(config['checkpoint_path'])!=config['checkpoint_sha256']:raise ValueError('Saved model changed')
    return job,model


def adapt(tree,counts,names):
    expected=dict(counts);seen={k:0 for k in expected}
    class OriginalCoverage(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value in expected:
                seen[node.value]+=1
                node.value=20 if node.value==80 else names[node.value]
            return node
    tree=OriginalCoverage().visit(tree)
    if seen!=expected:raise ValueError('Only documented original coverage/name bindings may change: '+str(seen))
    return ast.fix_missing_locations(tree)


def worker_ast():
    return adapt(previous.worker_ast(),{80:2,'full80_native_failure_observation.json':1},
        {'full80_native_failure_observation.json':'full20_native_failure_observation.json'})


def main_ast():
    return adapt(ast.parse(inspect.getsource(previous.previous.main)),
        {80:1,'full80_native_failure_observation.json':1,'full80_failure_observation_captured':1},
        {'full80_native_failure_observation.json':'full20_native_failure_observation.json',
         'full80_failure_observation_captured':'full20_failure_observation_captured'})


def namespace():
    values=dict(previous.previous.__dict__,verify=verify,__name__=MODULE,__doc__=__doc__)
    exec(compile(worker_ast(),__file__,'exec'),values)
    exec(compile(main_ast(),__file__,'exec'),values)
    return values


def post_ast():
    from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as arrays
    return adapt(arrays.corrected_audit_ast(),
        {80:3,'full80_native_failure_observation.json':2,'full80_failure_observation_captured':1},
        {'full80_native_failure_observation.json':'full20_native_failure_observation.json',
         'full80_failure_observation_captured':'full20_failure_observation_captured'})


def post_audit(config):
    from scripts.flow_matching import audit_cfm_full80_observed_arrays_v2 as arrays
    native=SimpleNamespace(ROOT=ROOT,read=read,sha=sha,verify=verify,validity_summary=previous.previous.validity_summary)
    values=dict(arrays.audit.__globals__,native=native)
    exec(compile(post_ast(),__file__,'exec'),values)
    return values['audit'](config)


def main():namespace()['main']()


if __name__=='__main__':main()
