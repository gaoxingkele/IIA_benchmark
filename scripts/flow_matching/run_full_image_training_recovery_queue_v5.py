"""Retain all25 full image jobs and raise admission for verified813 continuation."""
import ast
from copy import deepcopy
import inspect

from scripts.flow_matching import run_full_image_training_recovery_queue_v4 as previous

original=previous.original


def verify_registration(config):
    prior=original.read(original.ROOT/config['previous_configuration'])
    expected=deepcopy(prior['settings']);expected['minimum_commit_headroom_bytes']=40*1024**3
    if config['settings']!=expected:
        raise ValueError('Only admission may increase to40GiB; every other original guard must remain')
    # Original scientific registration compares against canonical runtime
    # admission. The actual slot and worker guard receive the stronger policy.
    canonical=deepcopy(config);canonical['settings']=deepcopy(prior['settings'])
    registered=original.verify_registration(canonical)
    for key in ('resource_lock','lane_lock','full_cohort_required_free_disk_bytes','original_runtimes','ordered_original_jobs'):
        if config[key]!=prior[key]:raise ValueError('All25 original jobs, GPU lanes and checkpoint budgets retained')
    resume=config['saved813_resume']
    if (resume['job_configuration']!=config['ordered_original_jobs'][0]
            or resume['completed_epochs']!=813 or resume['remaining_epochs']!=1187):
        raise ValueError('Use all813 verified epochs and retain the full1187 original remainder')
    original.verify_receipts([resume['CPU_proof'],resume['CPU_terminal'],resume['archive']])
    proof=original.read(original.ROOT/resume['CPU_proof']['path'])
    terminal=original.read(original.ROOT/resume['CPU_terminal']['path'])
    if (not proof['passed'] or proof['completed_original_epochs']!=813
            or proof['verified_original_optimizer_updates']!=92682 or proof['complete_epoch_receipts_verified']!=813
            or proof['full_native_parameters']!=45289539 or proof['required_original_epochs']!=2000
            or proof['remaining_original_epochs']!=1187 or proof['CUDA_initialized']
            or not proof['all_model_and_StableAdamW_variables_bitwise_restored']
            or not proof['Python_NumPy_native_CPU_and_order_RNG_states_bitwise_restored']
            or not proof['all_model_and_optimizer_tensors_finite'] or proof['new_optimizer_updates_executed']!=0
            or terminal['exit_code']!=0 or not terminal['passed'] or terminal['resource_guard_aborted']
            or terminal['report_sha256']!=resume['CPU_proof']['sha256']
            or proof['checkpoint_receipt']!=resume['archive']):
        raise ValueError('Actual full813 model/optimizer/CPU RNG restoration and guarded success required')
    return registered


def main_ast():
    tree=ast.parse(inspect.getsource(original.main));changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_full_image_training_execution_recovery_v3':
            node.value='scripts.flow_matching.run_full_image_training_execution_recovery_v5';changes+=1
    if changes!=1:raise ValueError('Only the next isolated full native worker entry may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(original.__dict__,verify_registration=verify_registration,queue_api=previous.queue_api)
    exec(compile(main_ast(),__file__,'exec'),namespace);namespace['main']()


if __name__=='__main__':main()
