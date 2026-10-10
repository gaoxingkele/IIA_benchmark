"""Keep all30 Table2 jobs and native proofs; use full GPU memory admission."""
from scripts.flow_matching import run_spectral_table2_method_gated_queue_v1 as original
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot

original.complete_native_slot = full_gpu_memory_slot


if __name__ == '__main__':
    original.main()
