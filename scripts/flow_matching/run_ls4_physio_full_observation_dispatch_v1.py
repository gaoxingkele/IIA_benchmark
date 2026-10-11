"""Keep all10 original full LS4 patient jobs and observe all500 native evaluations."""
import argparse
import ast
import importlib.metadata
import json
import sys

from scripts.flow_matching.ls4_physio_full_evaluation_observation_v1 import ROOT,read,sha,budget,periodic_epochs,worker_ast,FullEvaluationObservation,verify_extended_result


def verify(configuration):
    for row in configuration['source_receipts']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Frozen complete LS4 observation source changed')
    queue=read(ROOT/configuration['original_queue'])
    if (configuration['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
            or configuration['full_original_jobs']!=queue['jobs'] or len(queue['jobs'])!=10
            or configuration['original_settings']!=queue['settings'] or configuration['original_gpu_mutex']!=queue['resource_lock']
            or configuration['additional_training_seed_slots']!=0 or configuration['required_full_native_evaluations']!=500
            or configuration['required_original_training_optimizer_updates']!=750000
            or configuration['required_full_native_reconstruction_and_nll_calls']!=37500):
        raise ValueError('All ten original jobs, scientific settings, guards and37500 full method calls required')
    calls=updates=0
    for job in queue['jobs']:
        config=read(ROOT/job['model_config']);b=budget(config)
        if len(periodic_epochs(500,10)+[-1])!=50:raise ValueError('Original complete evaluation cadence changed')
        calls+=50*2*b['test_batches'];updates+=b['optimizer_updates']
    if (calls,updates)!=(37500,750000):raise ValueError('Every original full epoch and model call required')
    return queue


def environment(queue):
    lock=read(ROOT/queue['environment_lock'])
    if not lock['passed'] or '.'.join(map(str,sys.version_info[:3]))!=lock['python_version']:
        raise ValueError('Exact original LS4 author Python environment required')
    for name,version in lock['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Native LS4 package version differs: '+name)


def controller_ast(source):
    tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments=[i for i,n in enumerate(main.body) if ast.dump(n)==ast.dump(ast.parse('queue = read(ROOT / args.queue)').body[0])]
    if len(assignments)!=1:raise ValueError('One exact original controller queue binding required')
    main.body.insert(assignments[0]+1,ast.parse('queue = dict(queue,state_root=observation_state_root)').body[0])
    class NativeEntry(ast.NodeTransformer):
        count=0
        def visit_Constant(self,node):
            if node.value=='scripts.flow_matching.run_ls4_native_physionet_v1':
                self.count+=1;node.value='scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1'
            return node
    replacement=NativeEntry();main=replacement.visit(main)
    if replacement.count!=1:raise ValueError('Exactly one original complete native worker entry required')
    launch=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='launch')
    additions=[i for i,n in enumerate(launch.body) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='command']
    if len(additions)!=1:raise ValueError('One original full native command argument selection required')
    launch.body.insert(additions[0]+1,ast.parse('command.extend(["--runtime-config",observation_configuration])').body[0])
    states=[i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='state']
    if len(states)!=1:raise ValueError('One original native controller state required')
    main.body.insert(states[0]+1,ast.parse('state["observation_configuration_sha256"] = sha(ROOT/observation_configuration)').body[0])
    return ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))


def all_jobs_unstarted(queue):
    return all(not (ROOT/j['output_directory']).exists() and not (ROOT/j['artifact_directory']).exists() for j in queue['jobs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--runtime-config',required=True);parser.add_argument('--controller',action='store_true')
    parser.add_argument('--worker-probe-only',action='store_true');parser.add_argument('--queue');parser.add_argument('--job-id')
    args=parser.parse_args();configuration=read(ROOT/args.runtime_config)
    if configuration['runtime_configuration']!=args.runtime_config:raise ValueError('Exact frozen observation configuration required')
    queue=verify(configuration)
    if args.worker_probe_only:
        environment(queue)
        compile(worker_ast((ROOT/'scripts/flow_matching/run_ls4_native_physionet_v1.py').read_text(encoding='utf-8')),'<unchanged-full-native-worker>','exec')
        compile(controller_ast((ROOT/'scripts/flow_matching/run_ls4_native_physionet_queue_v3.py').read_text(encoding='utf-8')),'<unchanged-full-native-controller>','exec')
        print(json.dumps(dict(passed=True,exact_native_author_environment_verified=True,full_original_jobs=10,
            full_native_evaluations=500,full_original_reconstruction_and_nll_calls=37500,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,numerical_training_executed=False,
            all_original_experiments_complete=False)));return
    if args.controller:
        # The launch helper performs an exclusive verified idle-owner handoff first.
        prior=read(ROOT/configuration['handoff_receipt'])
        if not prior['old_waiting_controller_terminal'] or prior['active_native_worker_stopped']:
            raise ValueError('Verified old idle controller terminal before replacement launch required')
        from scripts.flow_matching import run_ls4_native_physionet_queue_v3 as original
        namespace=dict(vars(original),observation_configuration=args.runtime_config,observation_state_root=configuration['state_root'])
        namespace['verify_result']=lambda q,j:verify_extended_result(q,j,args.runtime_config)
        exec(compile(controller_ast((ROOT/original.__file__).read_text(encoding='utf-8')),original.__file__,'exec'),namespace)
        sys.argv=['complete-ls4-native-observation-queue','--queue',configuration['original_queue']]
        namespace['main']();return
    if args.queue!=configuration['original_queue'] or not args.job_id:
        raise ValueError('Exact complete original worker queue and job required')
    environment(queue)
    from scripts.flow_matching import run_ls4_native_physionet_v1 as original
    original.verify(queue)
    job=next(j for j in queue['jobs'] if j['id']==args.job_id)
    namespace=dict(vars(original),FullEvaluationObservation=FullEvaluationObservation,observation_configuration=args.runtime_config)
    exec(compile(worker_ast((ROOT/original.__file__).read_text(encoding='utf-8')),original.__file__,'exec'),namespace)
    namespace['run_job'](queue,job)


if __name__=='__main__':main()
