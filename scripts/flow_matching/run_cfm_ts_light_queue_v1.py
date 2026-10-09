"""Run unchanged CFM-TS orchestration without importing training libraries."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest, load_functions


def load(runtime):
    for receipt in runtime['source_receipts']:
        if digest(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen lightweight runtime changed: '+receipt['path'])
    if digest(ROOT / runtime['queue']) != runtime['queue_sha256']:
        raise ValueError('Frozen original-task queue changed')
    namespace=dict(globals(),ROOT=ROOT,__name__='cfm_ts_pinned_controller')
    selections={
        'scripts/flow_matching/prepare_cfm_ts_original_data.py':['sha','read'],
        'scripts/flow_matching/run_cfm_ts_original_job.py':['fingerprint','verify','complete','write'],
        'scripts/flow_matching/run_tsad_queue.py':['exclusive_lock'],
        'scripts/flow_matching/heavy_resources.py':['resource_snapshot','has_headroom','terminate_tree','guarded_wait'],
        'scripts/flow_matching/run_cfm_ts_original_queue.py':['main']}
    for source,names in selections.items():
        load_functions(ROOT,source,names,namespace)
    return namespace['main']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();runtime=json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    bound=load(runtime)
    if args.probe_only:
        memory=psutil.Process().memory_info()
        print(json.dumps({'private_bytes':getattr(memory,'private',memory.rss),
                          'torch_imported':'torch' in sys.modules,'numpy_imported':'numpy' in sys.modules,
                          'scipy_imported':'scipy' in sys.modules,'queue_sha256':runtime['queue_sha256']}))
        return
    sys.argv=['pinned_cfm_ts_controller','--queue',runtime['queue']]
    bound()


if __name__=='__main__':
    main()
