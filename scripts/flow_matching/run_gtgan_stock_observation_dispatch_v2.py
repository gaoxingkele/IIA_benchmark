"""Hand off only the unstarted stock parent to an observationally complete worker."""
import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys

from scripts.flow_matching.gtgan_stock_observation_runtime_v2 import ROOT, read, sha, extended_budget, compile_worker, compile_native_verifier, verify_author_environment, verify_extended_result
from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2 import exclusive, allowed_console_child


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete stock observation source changed: ' + pin['path'])
    queue = read(ROOT / config['original_queue'])
    for pin in queue['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Original full native stock queue/source changed')
    model = read(ROOT / queue['model_config'])
    if (len(queue['jobs']) != 4 or len(model['cases']) != 4
            or {j['case_id'] for j in queue['jobs']} != {c['id'] for c in model['cases']}
            or {c['missing_rate'] for c in model['cases']} != {0.,.3,.5,.7}
            or config['settings'] != queue['settings'] or config['resource_lock'] != queue['resource_lock']
            or config['cpu_threads'] != queue['cpu_threads']
            or config['original_scope'] != dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('All four original stock commands, full research scope and unchanged GPU safeguards required')
    budgets = [extended_budget(c) for c in model['cases']]
    if (sum(b['full_periodic_evaluations'] for b in budgets) != 29
            or config['required_trained_evaluator_receipts'] != 580
            or config['required_sampler_snapshots'] != 29 or config['required_native_network_checkpoints'] != 116
            or config['native_latent_noise_shape'] != [3662,24,24]
            or config['native_real_and_generated_shape'] != [3662,24,6]):
        raise ValueError('Every complete original periodic generation and trained evaluator required')
    return queue


def all_attempts_unstarted(queue):
    return all(not (ROOT/j['output_directory']).exists()
        and not (Path(j['runtime_root'])/'native_evaluator_receipts').exists()
        and not any(Path(j['runtime_root']).glob('full_sampler_inputs_*.npz')) for j in queue['jobs'])


def idle_parent_phase(state, descendants, all_unstarted):
    return (state.get('active_pid') is None and state.get('state') == 'waiting_full_gpu_resources'
        and state.get('shared_lock_owned') is False and all_unstarted
        and all(allowed_console_child(p['name'],p['exe']) for p in descendants))


def handoff(config,queue,base,update):
    from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as legacy
    if not all_attempts_unstarted(queue):
        raise RuntimeError('Original stock task started after registration; preserve it and inspect exact continuation')
    def predicate(role,state,output_exists,descendants):
        return idle_parent_phase(state,descendants,all_attempts_unstarted(queue))
    namespace = dict(legacy.__dict__,handoff_phase_allowed=predicate)
    exec(compile(inspect.getsource(legacy.safe_handoff),legacy.__file__,'exec'),namespace)
    route = dict(config['previous_owner_route'],runtime_configuration=config['runtime_configuration'])
    namespace['safe_handoff']('original',route,base,update,10)


def controller_ast(source):
    import ast
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('One complete original four-command GPU controller required')
    modules = 0; bases = 0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value == 'scripts.flow_matching.run_gtgan_native_job_v1':
            node.value = 'scripts.flow_matching.run_gtgan_stock_observation_dispatch_v2'; modules += 1
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base' for t in node.targets):
            expected = ast.dump(ast.parse("ROOT / queue['state_root']",mode='eval').body)
            if ast.dump(node.value) != expected: raise ValueError('Unexpected native controller state root')
            node.value = ast.parse('ROOT / _observed_state_root',mode='eval').body; bases += 1
    if modules != 1 or bases != 1:
        raise ValueError('Only the native worker entry and separate controller state root may change')
    return ast.fix_missing_locations(tree)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--runtime-config',default='configs/runtime/fm_gtgan_stock_observation_dispatch.v2.json')
    parser.add_argument('--controller',action='store_true')
    parser.add_argument('--queue');parser.add_argument('--job-id')
    parser.add_argument('--worker-probe-only',action='store_true')
    args=parser.parse_args()
    config=read(ROOT/args.runtime_config)
    if config['runtime_configuration']!=args.runtime_config:
        raise ValueError('Exact immutable native stock observation runtime required')
    queue=verify(config)
    if args.worker_probe_only:
        verify_author_environment(queue)
        compile_worker(sha(ROOT/args.runtime_config))
        compile_native_verifier()
        compile(controller_ast((ROOT/'scripts/flow_matching/run_gtgan_native_queue_v1.py').read_text(encoding='utf-8')),
            '<full-unchanged-native-stock-controller>','exec')
        print(json.dumps(dict(passed=True,original_stock_jobs=4,required_full_periodic_evaluations=29,
            required_full_trained_evaluator_receipts=580,source_training_budget_preserved=True,
            torch_imported='torch' in sys.modules,tensorflow_imported='tensorflow' in sys.modules,
            numpy_imported='numpy' in sys.modules,psutil_imported='psutil' in sys.modules,
            original_environment_lock_verified=True,full_worker_controller_and_verifier_compiled=True,numerical_training_executed=False,
            all_original_experiments_complete=False)))
        return
    if not args.controller:
        if args.queue!=config['original_queue'] or args.job_id not in {j['id'] for j in queue['jobs']}:
            raise ValueError('Exact original native queue and canonical stock job required')
        sys.argv=['native-stock-observation','--queue',args.queue,'--job-id',args.job_id]
        compile_worker(sha(ROOT/args.runtime_config))()
        job=next(j for j in queue['jobs'] if j['id']==args.job_id)
        verify_extended_result(queue,job,sha(ROOT/args.runtime_config))
        return
    import psutil
    from scripts.flow_matching import run_gtgan_native_queue_v1 as original
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.timegan_native_common_v1 import write
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=resource_api()['exclusive_lock'](base/'preparation.lock')
    if owner is None:raise RuntimeError('Corrected stock observation handoff already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        configuration_sha256=sha(ROOT/args.runtime_config),state='starting_observation_handoff',
        original_stock_commands=4,all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'preparation_status.json',state)
    try:
        update({});handoff(config,queue,base,update)
        namespace=dict(original.__dict__,_observed_state_root=config['state_root'],
            verify_result=lambda q,j:verify_extended_result(q,j,sha(ROOT/args.runtime_config)))
        exec(compile(controller_ast(Path(original.__file__).read_text(encoding='utf-8')),original.__file__,'exec'),namespace)
        sys.argv=['native-original-stock-controller','--queue',config['original_queue']]
        namespace['main']()
    finally:owner.close()


if __name__=='__main__':main()
