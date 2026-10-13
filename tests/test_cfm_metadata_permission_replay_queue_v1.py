from pathlib import Path
from scripts.flow_matching import run_cfm_metadata_permission_replay_queue_v1 as current


def test_actual_full285_routes_preserve_every_other_source_and_guard():
    runtime=current.cold.original.read(current.ROOT/'configs/runtime/fm_cfm_metadata_permission_full_replay.v1.json')
    previous=current.cold.original.read(current.ROOT/current.PRIMARY)
    values=current.namespace();routes,template,verified=values['verify'](runtime)
    assert len(routes)==285 and len(verified)>=135 and len(runtime['selected_canonical_ids'])==2
    old={r['original_canonical_id']:r for r in previous['routes']}
    for route in routes:
        identity=route['original_canonical_id']
        if identity not in runtime['selected_canonical_ids']:assert route==old[identity]
    for key in ['settings','resource_lock','python','environment_lock','prediction_tolerance']:
        assert runtime[key]==previous[key]


def test_nonselected_complete_result_never_enters_supplemental_native_dispatch(tmp_path):
    p=tmp_path/'result.json';p.write_text('{}')
    runtime={'selected_canonical_ids':['selected']}
    ready=current.supplement.selected_ready
    assert ready({'original_canonical_id':'selected','result_path':str(p)},runtime)
    assert not ready({'original_canonical_id':'other','result_path':str(p)},runtime)
