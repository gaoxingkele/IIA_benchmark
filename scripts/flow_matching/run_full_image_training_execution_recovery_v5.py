"""Restore the verified native813 state using the original full GPU observer."""
import argparse
import ast
import inspect

from scripts.flow_matching import run_full_image_training_execution_recovery_v4 as previous


def resume_run_ast():
    tree=ast.parse(inspect.getsource(previous.run))
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant):
            if type(node.value) is int and node.value==401:node.value=813
            elif isinstance(node.value,str):node.value=node.value.replace('saved401_resume','saved813_resume').replace('401','813')
    return ast.fix_missing_locations(tree)


def run(config_path,job_path,attempt_directory,binding_path):
    # The original training AST, full CUDA restoration observer and all math
    # stay in the previous namespace. Only its frozen resume input advances.
    namespace=dict(previous.__dict__)
    exec(compile(resume_run_ast(),__file__,'exec'),namespace)
    namespace['run'](config_path,job_path,attempt_directory,binding_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--recovery-config',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--attempt-directory',required=True);parser.add_argument('--capacity-binding',required=True)
    args=parser.parse_args();run(args.recovery_config,args.config,args.attempt_directory,args.capacity_binding)
