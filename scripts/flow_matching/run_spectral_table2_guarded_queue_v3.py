"""Complete CPU guard and explicit GPU identity bindings for the frozen Table2 queue."""
import argparse
import json
import sys

from scripts.flow_matching import run_spectral_table2_queue_v1 as original
from scripts.flow_matching.run_spectral_table2_guarded_queue_v2 import complete_resource_api
from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT,sha


def bind_resource_api(original_api,gpu_uuid):
    api=complete_resource_api(original_api)
    snapshot=api['gpu_snapshot']
    def bound_snapshot(settings):
        return snapshot(dict(settings,gpu_uuid=gpu_uuid))
    api['gpu_snapshot']=bound_snapshot
    return api


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();runtime=json.loads((ROOT/args.runtime_config).read_text(encoding='utf-8'))
    if sha(ROOT/runtime['queue'])!=runtime['queue_sha256']:raise ValueError('Full native queue changed')
    for entry in runtime['source_receipts']:
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Registered resource binding changed')
    old_api=original.guarded_api
    original.guarded_api=lambda:bind_resource_api(old_api,runtime['gpu_uuid'])
    if args.probe_only:
        queue=json.loads((ROOT/runtime['queue']).read_text(encoding='utf-8'));api=original.guarded_api()
        print(json.dumps(dict(full_jobs=len(queue['jobs']),gpu=api['gpu_snapshot'](queue['settings']),
            resources=api['resource_snapshot'](),cpu_guard_callable=callable(api['guarded_wait']),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    sys.argv=[original.__file__,'--queue',runtime['queue']]
    original.main()


if __name__=='__main__':main()
