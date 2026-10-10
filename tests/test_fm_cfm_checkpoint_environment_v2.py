from scripts.flow_matching.register_cfm_checkpoint_replay_v2 import corrected_runtime


def test_replay_uses_original_queue_interpreter_without_changing_scientific_inputs(tmp_path):
    previous = dict(python='system_python.exe', full_job_count=83,
        full_results=[dict(id='saved_seed1103', job=dict(epochs=400, seed=1103))],
        prediction_tolerance=dict(rtol=1e-7, atol=1e-8), environment_lock='original.lock')
    original = dict(python='.venv/original/Scripts/python.exe')
    corrected = corrected_runtime(previous, original, tmp_path)
    assert corrected['python']==str((tmp_path/original['python']).resolve())
    for field in ['full_job_count', 'full_results', 'prediction_tolerance', 'environment_lock']:
        assert corrected[field]==previous[field]
    assert previous['python']=='system_python.exe'
