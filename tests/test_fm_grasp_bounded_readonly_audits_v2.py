from scripts.flow_matching.run_bounded_readonly_grasp_audits_v2 import worker_command


def test_original_full_worker_command_is_unchanged():
    command=worker_command('configs/original.json',dict(python='frozen-python'))
    assert command==['frozen-python','-X','utf8','-u','-m',
        'scripts.flow_matching.audit_registered_grasp_results_v1','--config','configs/original.json','--worker']
