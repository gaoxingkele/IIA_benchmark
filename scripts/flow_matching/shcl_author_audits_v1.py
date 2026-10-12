"""Run all original13/52 audit obligations in the pinned Torch1.11 environment."""
import argparse
import importlib.metadata
import json
import platform
import sys
from types import FunctionType

from scripts.flow_matching import shcl_released_native_v1 as native


def environment():
    return dict(python=platform.python_version(),executable=sys.executable,
        packages={name:importlib.metadata.version(name) for name in
            ('torch','numpy','pandas','scikit-learn','scipy','tqdm')})


def compiled(mode):
    if mode=='operators':
        from scripts.flow_matching import audit_shcl_released_operators_v1 as original
        verification=FunctionType(original.verify.__code__,dict(original.__dict__,environment=environment))
        return FunctionType(original.worker.__code__,dict(original.__dict__,environment=environment,verify=verification)),verification
    if mode=='data':
        from scripts.flow_matching import prepare_shcl_released_data_v1 as original
        from scripts.flow_matching.prepare_shcl_released_data_v3 import compiled_worker
        verification=FunctionType(original.verify.__code__,dict(original.__dict__,data_environment=environment))
        worker=compiled_worker()
        return FunctionType(worker.__code__,dict(worker.__globals__,data_environment=environment,verify=verification)),verification
    raise ValueError('Every complete original operator/data obligation remains required')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config'); parser.add_argument('--environment-only',action='store_true')
    args=parser.parse_args()
    if args.environment_only:
        print(json.dumps(environment())); return
    config=native.read(native.ROOT/args.config); worker,verification=compiled(config['audit_kind'])
    verification(config); worker(config,args.config)


if __name__=='__main__':main()
