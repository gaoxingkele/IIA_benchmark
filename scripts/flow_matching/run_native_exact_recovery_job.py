"""Dispatch the unchanged full native worker to an explicitly registered recovery attempt."""
import json
import sys
from scripts.flow_matching.native_exact_recovery import ROOT, worker_main


def main():
    path = sys.argv[sys.argv.index('--queue') + 1]
    queue = json.loads((ROOT / path).read_text(encoding='utf-8'))
    if queue['recovery_track'] == 'grasp':
        from scripts.flow_matching.grasp_protocol_cuda_runtime import initialize_cuda
        initialize_cuda()
    worker_main(queue['recovery_track'])()


if __name__ == '__main__':
    main()
