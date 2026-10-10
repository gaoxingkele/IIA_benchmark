"""Exact original native recovery with the fixed manifest and GPU memory admission."""
import sys
from scripts.flow_matching.native_exact_recovery_manifest_v2 import controller_main, worker_main
from scripts.flow_matching.run_grasp_gpu_memory_fair_queue_v6 import load_gpu_controller


def load_recovery_gpu_controller(root,runtime,name):
    selected,api,controller=load_gpu_controller(root,runtime,name)
    def slot(lock,settings,update,poll_seconds=.25):
        # The full memory gate checks the same GPU mutex every10seconds,
        # within the35second native-family fairness yield. No scientific budget changes.
        return api['resource_slot'](lock,settings,update)
    return selected,dict(api,resource_slot=slot),controller


def full_controller():
    bound=controller_main()
    bound.__globals__['load_controller']=load_recovery_gpu_controller
    return bound


if __name__=='__main__':
    if '--controller' in sys.argv:
        sys.argv.remove('--controller');full_controller()()
    else:
        from scripts.flow_matching.grasp_protocol_cuda_runtime import initialize_cuda
        initialize_cuda();worker_main()()
