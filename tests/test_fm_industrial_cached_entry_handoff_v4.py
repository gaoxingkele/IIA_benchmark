from scripts.flow_matching import run_industrial_cached_entry_handoff_v4 as sut


def test_original_main_retains_the_cached_loader_namespace(monkeypatch):
    import scripts.flow_matching.run_light_controller as original
    old_loader = original.load_functions
    from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as client
    old_client = client.load_functions
    try:
        function = sut.prepare_entry()
        assert function is original.main
        assert function.__globals__ is original.__dict__
        assert function.__globals__['load_functions'] is not old_loader
        namespace = dict(original.__dict__)
        functions = original.load_functions(original.ROOT, 'scripts/flow_matching/heavy_resources.py',
                                           ['resource_snapshot'], namespace)
        assert functions['resource_snapshot'] is namespace['resource_snapshot']
        assert namespace['resource_snapshot'].__name__ == 'cached_snapshot'
        assert namespace['resource_snapshot']()['commit_available_bytes'] > 0
    finally:
        original.load_functions = old_loader
        client.load_functions = old_client


def test_original_scope_cannot_be_narrowed():
    import pytest
    with pytest.raises(ValueError, match='global scope'):
        sut.verify(dict(original_scope={}, original_role='industrial9'))
