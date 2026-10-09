"""Create metadata parents before the unchanged atomic receipt writer.

The first exact recovery completed its full model run but its new receipts
directory was absent. Preserve the result and repair only metadata writing.
"""
from pathlib import Path
from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as original

atomic_write=original.write


def write_with_parents(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    atomic_write(path,data)


if __name__=='__main__':
    original.write=write_with_parents
    original.main()
