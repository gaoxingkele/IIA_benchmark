"""Version the prelaunch controller import repair, preserving failed registration."""
from copy import deepcopy
import json

from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT,read,sha,write
from scripts.flow_matching.register_cfm_ts_exact_recovery_v1 import validate_recovery


def main():
    previous='configs/experiments/fm_cfm_ts_exact_recovery_queue.v1.json'
    target='configs/experiments/fm_cfm_ts_exact_recovery_queue.v2.json'
    runtime_previous='configs/runtime/fm_cfm_ts_low_memory_controller.v2.json'
    runtime_target='configs/runtime/fm_cfm_ts_low_memory_controller.v3.json'
    if (ROOT/target).exists() or (ROOT/runtime_target).exists():raise FileExistsError('Preserve repaired registration')
    queue=deepcopy(read(ROOT/previous))
    queue['preflight_failure_predecessor']={'queue':previous,'queue_sha256':sha(ROOT/previous),
        'source_commit':'fd1a9b1','reason':'Controller SHA helper AST needs hashlib global; no training started under the failed registration.'}
    for receipt in queue['source_receipts']:receipt['sha256']=sha(ROOT/receipt['path'])
    queue['source_receipts'].append({'path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(Path(__file__))})
    validate_recovery(read(ROOT/queue['recovery_original_queue']),queue);write(ROOT/target,queue)
    runtime=deepcopy(read(ROOT/runtime_previous))
    runtime['controllers']['recovery']['queue']=target
    runtime['controllers']['recovery']['queue_sha256']=sha(ROOT/target)
    runtime['preflight_failure_predecessor']={'runtime':runtime_previous,'sha256':sha(ROOT/runtime_previous),'source_commit':'fd1a9b1'}
    for receipt in runtime['source_receipts']:receipt['sha256']=sha(ROOT/receipt['path'])
    runtime['source_receipts'].append({'path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(Path(__file__))})
    write(ROOT/runtime_target,runtime)
    print(json.dumps({'queue':target,'runtime':runtime_target,'full_original_slots':len(queue['jobs']),'recovery_jobs':len(queue['recovery_job_ids'])}))


if __name__=='__main__':
    from pathlib import Path
    main()
