"""Bind the missing CPU guard without changing the frozen author queue or worker."""
import argparse
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_spectral_table2_queue_v1 as original
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT,sha


def complete_resource_api(original_api):
    api = original_api()
    api['guarded_wait'] = resource_api()['guarded_wait']
    return api


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    args=parser.parse_args()
    runtime=json.loads((ROOT/args.runtime_config).read_text(encoding='utf-8'))
    if sha(ROOT/runtime['queue'])!=runtime['queue_sha256']:
        raise ValueError('Frozen thirty-job queue differs')
    for entry in runtime['source_receipts']:
        if sha(ROOT/entry['path'])!=entry['sha256']:
            raise ValueError('Guard runtime differs: '+entry['path'])
    old_api=original.guarded_api
    original.guarded_api=lambda:complete_resource_api(old_api)
    sys.argv=[str(Path(original.__file__)),'--queue',runtime['queue']]
    original.main()


if __name__=='__main__':main()
