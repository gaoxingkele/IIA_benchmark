import ast
from pathlib import Path
from scripts.flow_matching import prepare_CITE13176_GEO_raw_backed_QC_v2 as s


def test_full_mathematical_worker_bytecode_and_all_original_guard_values_remain_exact():
    c=s.registration();old=s.read(s.ROOT/s.previous.CONFIG)
    assert s.namespace()['worker'].__code__ is s.previous.worker.__code__
    for key in ['settings','resource_lock','stable_samples','poll_seconds','numerical_threads','normalization',
                'normalization_candidates_audited','candidate_audit_tolerance','independent_normalization_library',
                'input_records','expected_QC_cells','expected_RNA_features','pca_dimensions','pca_initial_vector_seed','pca_arpack_tolerance']:
        assert c[key]==old[key]
    assert c['output_directory']!=old['output_directory'] and not c['metadata_recovery']['old_numerical_worker_started']


def test_only_one_controller_subscript_name_is_changed_and_actual_API_supports_it():
    tree=ast.parse(Path(s.previous.__file__).read_text(encoding='utf-8'))
    old=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    new=s.patched_main_tree().body[0]
    class Undo(ast.NodeTransformer):
        def visit_Subscript(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='api' and isinstance(node.slice,ast.Constant) and node.slice.value=='has_headroom':
                node.slice=ast.Constant('enough')
            return node
    assert ast.dump(old,include_attributes=False)==ast.dump(Undo().visit(new),include_attributes=False)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    api=resource_api();assert 'has_headroom' in api and 'enough' not in api
    c=s.registration();settings=c['settings']
    snapshot=dict(physical_available_bytes=settings['minimum_available_memory_bytes'],commit_available_bytes=settings['minimum_commit_headroom_bytes'])
    assert api['has_headroom'](snapshot,settings)
    snapshot['commit_available_bytes']-=1
    assert not api['has_headroom'](snapshot,settings)
