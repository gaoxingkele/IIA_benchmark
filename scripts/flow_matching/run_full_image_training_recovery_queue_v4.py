"""Resume the preserved native401 state and retain every20FM/5WT full-budget job."""
import ast
import inspect

from scripts.flow_matching import run_full_image_training_recovery_queue_v3 as original
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot


def verify_registration(config):
    registered=original.verify_registration(config)
    resume=config['saved401_resume']
    if resume['job_configuration']!=config['ordered_original_jobs'][0] or resume['completed_epochs']!=401 or resume['remaining_epochs']!=1599:
        raise ValueError('The complete verified401 state and all1599 remaining original epochs required')
    original.verify_receipts([resume['CPU_proof'],resume['archive']])
    proof=original.read(original.ROOT/resume['CPU_proof']['path'])
    if not proof['passed'] or proof['completed_original_epochs']!=401 or proof['verified_original_optimizer_updates']!=45714:
        raise ValueError('Every saved native model/optimizer/CPU state and full epoch receipt must already pass')
    previous=original.read(original.ROOT/config['previous_configuration'])
    for key in ('settings','resource_lock','lane_lock','full_cohort_required_free_disk_bytes','original_runtimes','ordered_original_jobs'):
        if config[key]!=previous[key]:raise ValueError('All original jobs, budgets, lanes and full resource policies retained')
    return registered


def queue_api():
    api=original.queue_api();api['resource_snapshot']=resource_snapshot
    return api


def main_ast():
    tree=ast.parse(inspect.getsource(original.main));changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_full_image_training_execution_recovery_v3':
            node.value='scripts.flow_matching.run_full_image_training_execution_recovery_v4';changes+=1
    if changes!=1:raise ValueError('Only the next fresh native worker gains restoration verification')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(original.__dict__,verify_registration=verify_registration,queue_api=queue_api)
    exec(compile(main_ast(),__file__,'exec'),namespace);namespace['main']()


if __name__=='__main__':main()
