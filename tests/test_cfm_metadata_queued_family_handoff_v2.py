from scripts.flow_matching import run_cfm_metadata_queued_family_handoff_v2 as current


def test_same_full_retry_dispatch_and_guard_bytecode_with_only_family_admission_binding():
    runtime=current.cold.original.read(current.ROOT/'configs/runtime/fm_cfm_metadata_permission_full_retry.v1.json')
    _,queue=current.previous.verify_runtime(runtime)
    old=current.previous.namespace();new=current.namespace(runtime,queue)
    assert old['main'].__code__.co_code==new['main'].__code__.co_code
    assert old['main'].__code__.co_consts==new['main'].__code__.co_consts
    assert old['external_complete'].__code__.co_code==new['external_complete'].__code__.co_code
    api=new['resource_api']();oldapi=old['resource_api']()
    assert api['guarded_wait'].__code__.co_code==oldapi['guarded_wait'].__code__.co_code
