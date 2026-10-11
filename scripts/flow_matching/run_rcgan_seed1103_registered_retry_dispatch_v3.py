"""Keep host scheduling outside the frozen Py37 full native worker entry point."""
import argparse
import ast
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v2 as previous
from scripts.flow_matching import rcgan_seed1103_registered_retry_runtime_v2 as runtime

ROOT,read,sha,verify=previous.ROOT,previous.read,previous.sha,previous.verify


def host_dispatch_ast(source):
    tree=previous.dispatch_ast(source)
    matches=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v2':
            node.value='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v3';matches+=1
    if matches!=1:raise ValueError('Exactly one corrected native worker module binding required')
    return ast.fix_missing_locations(tree)


def native_worker_arguments(remaining,expected_config):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--proof-binding',required=True)
    args=parser.parse_args(remaining)
    if args.config!=expected_config:raise ValueError('Exact immutable original native configuration required')
    return args


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--worker-probe-only',action='store_true')
    args,remaining=parser.parse_known_args()
    config=read(ROOT/args.runtime_config);native,queue,state_config=verify(config)
    if config.get('native_worker_entry_mode')!='direct_full_original_resume':
        raise ValueError('Native Py37 worker must bypass host-only scheduling AST')
    if '--worker' in remaining:
        worker=native_worker_arguments(remaining,config['native_configuration'])
        binding=read(ROOT/worker.proof_binding)
        if binding['configuration_sha256']!=sha(ROOT/config['native_configuration']):
            raise ValueError('Actual native proof binding must retain unchanged configuration')
        if args.worker_probe_only:
            compile(runtime.native_retry_ast(Path(runtime.original.__file__).read_text(encoding='utf-8')),
                '<complete unchanged native238+262 runtime accepting registered retry identity>','exec')
            print(json.dumps(dict(native_python=sys.version,native_interpreter=sys.executable,
                actual_native_worker_parser_and_frozen_source_registration_passed=True,
                full_native_resume_function_compiled=True,host_scheduling_AST_executed=False,
                original_jobs=5,full_epochs_per_job=500,verified_prior_epochs=238,remaining_full_epochs=262,
                exact_execution_id=native['recovery_job']['id'],
                tensorflow_imported='tensorflow' in sys.modules,numpy_imported='numpy' in sys.modules,
                training_executed=False,all_original_experiments_complete=False)))
            return
        runtime.run(native,queue,binding)
        return
    if args.worker_probe_only:raise ValueError('Native worker probe requires the exact actual --worker argv')
    if '--config' not in remaining or remaining[remaining.index('--config')+1]!=config['native_configuration']:
        raise ValueError('Exact immutable original native configuration required')
    source=Path(previous.original.__file__)
    tree=host_dispatch_ast(source.read_text(encoding='utf-8'))
    if '--probe-only' in remaining:
        compile(tree,str(source),'exec')
        print(json.dumps(dict(full_original_jobs=5,original_epochs_each=500,remaining_full_epochs=262,
            actual_host_dispatch_AST_compiled=True,native_worker_entry_mode=config['native_worker_entry_mode'],
            tensorflow_imported='tensorflow' in sys.modules,numpy_imported='numpy' in sys.modules,
            all_original_experiments_complete=False)))
        return
    namespace=dict(previous.original.__dict__,_retry_admission_settings=previous.admission_settings,
        _dispatch_configuration=config,_dispatch_configuration_path=args.runtime_config)
    exec(compile(tree,str(source),'exec'),namespace)
    sys.argv=[str(source)]+remaining
    namespace['main']()


if __name__=='__main__':main()
