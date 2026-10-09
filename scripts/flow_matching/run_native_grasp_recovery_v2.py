"""Dispatch the contract repair with the same cold CUDA setup and full budgets."""
import sys

from scripts.flow_matching.native_exact_recovery_manifest_v2 import controller_main, worker_main


if __name__ == '__main__':
    if '--controller' in sys.argv:
        sys.argv.remove('--controller')
        controller_main()()
    else:
        from scripts.flow_matching.grasp_protocol_cuda_runtime import initialize_cuda
        initialize_cuda()
        worker_main()()
