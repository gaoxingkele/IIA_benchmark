"""Keep full GRASP scientific jobs and guards; admit on the original GPU mutex."""
import argparse
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_grasp_protocol_queue as original
from scripts.flow_matching.run_grasp_full_fair_queue_v5 import bound_main
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot


def load_gpu_controller(root,runtime,name):
    selected,api,controller=original.load_controller(root,runtime,name)
    gpu_api=dict(api,gpu_snapshot=original.gpu_snapshot)
    def slot(lock,settings,update):
        return full_gpu_memory_slot(gpu_api,dict(resource_lock=str(Path(lock).relative_to(root)),
                                                settings=settings),update)
    return selected,dict(api,resource_slot=slot),controller


def full_main(yield_seconds):
    function=bound_main(yield_seconds)
    function.__globals__['load_controller']=load_gpu_controller
    return function


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    runtime=json.loads((original.ROOT/args.runtime_config).read_text(encoding='utf-8'))
    for source in runtime['source_receipts']:
        if original.sha(original.ROOT/source['path'])!=source['sha256']:
            raise ValueError('Frozen full GPU orchestration source differs')
    path=original.ROOT/runtime['queue']
    if original.sha(path)!=runtime['queue_sha256']:
        raise ValueError('Full native queue differs')
    queue=json.loads(path.read_text(encoding='utf-8'));original.verify(queue)
    function=full_main(runtime['yield_seconds'])
    if args.probe_only:
        print(json.dumps(dict(full_jobs=len(queue['jobs']),cohorts=queue['cohort_count'],
            post_job_yield_seconds=runtime['yield_seconds'],full_scientific_queue_unchanged=True,
            original_resource_settings=queue['settings'])));return
    sys.argv=['full-grasp-original-gpu-admission','--queue',str(path)]
    function()


if __name__=='__main__':main()
