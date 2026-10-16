"""Bind original paper recipes, full input identities and native metric semantics.

This is a material/protocol audit, never a training or benchmark result.
"""
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

AUDIT = 'projects/flow_matching_research/ara/adopted_baselines/trajectorynet_original_recipe_and_metric_audit.2026-10-16_v1.json'
PLAN = 'configs/reproducibility/fm_TrajectoryNet_EB_original_grid_full180_plan.v1.json'
ACQUISITION = 'configs/acquisition/fm_TrajectoryNet_author_checkpoint_materials.v1.json'
REPLAY = 'configs/reproducibility/fm_TrajectoryNet_author_fig8_full_checkpoint_replay.v1.json'
MODEL = 'configs/models/fm_TrajectoryNet_author_fig8_EB_leave3_replay.v1.json'
HELPER = 'src/iia_benchmark/models/trajectorynet_author_replay_v1.py'


def generate():
    import numpy as np
    from pypdf import PdfReader
    from sklearn.preprocessing import StandardScaler

    c, replay, profile = read(ROOT / ACQUISITION), read(ROOT / REPLAY), read(ROOT / MODEL)
    receipt = read(Path(c['output_directory']) / 'verified_acquisition.json')
    source = ROOT / c['source_root']
    for item in c['expected_original_blobs']:
        content = (source / item['path']).read_bytes()
        assert len(content) == item['size']
        assert hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == item['sha']
    assert len(c['expected_original_blobs']) == 63 and receipt['all_original_Git_blobs_verified']
    spec = importlib.util.spec_from_file_location('_trajectorynet_metadata_audit_v1', ROOT / HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    arguments = helper.logged_arguments((source / c['source_logged_arguments']).read_text())
    assert arguments == profile['source_parameters']
    checkpoint = helper.checkpoint_metadata(source / c['source_checkpoint'])

    paper = Path(receipt['transfers']['paper']['path'])
    assert sha(paper) == receipt['transfers']['paper']['sha256']
    pages = PdfReader(paper).pages
    appendix = pages[14].extract_text()
    compact = ''.join(appendix.split())
    assert 'E.2.SingleCellDatasets' in compact
    assert '{0,0.1,0.01}' in compact and '{0,0.001,0.0001}' in compact
    assert '{1.0,0.1,0.01}' in compact
    assert 'torch==1.3.1' in compact and 'torchdiffeq==0.0.1' in compact
    table_text = ''.join(pages[8].extract_text().split())
    assert 'Base+D+V0.9300.8060.8100.848' in table_text
    assert 'Base+E.0.7370.8960.8420.825' in table_text
    assert 'Base+D0.7590.7830.8110.784' in table_text
    otcfm = ROOT / 'papers/literature/flow_matching/pdfs/ot_cfm.pdf'
    ot_pages = PdfReader(otcfm).pages
    assert '1-Wasserstein' in ot_pages[9].extract_text()
    assert '0.848' in ot_pages[10].extract_text() and '0.825' in ot_pages[10].extract_text()

    data_path = Path(replay['original_data'])
    assert sha(data_path) == replay['original_data_sha256']
    with np.load(data_path, allow_pickle=False) as raw:
        pcs = raw['pcs']
        # The data label key is part of the original EBData implementation.
        labels = raw['sample_labels'] if 'sample_labels' in raw.files else raw['labels']
        assert pcs.shape == (16819, 100)
        scaled = StandardScaler().fit_transform(pcs)[:, :5].astype(np.float32)
    counts = {str(t): int(np.sum(labels == t)) for t in range(5)}
    assert counts == replay['full_population_counts']
    historical = np.load(replay['author_reference_trajectory'], mmap_mode='r', allow_pickle=False)
    assert historical.shape == (100, 3332, 5)
    assert np.array_equal(scaled[labels == 4], historical[0])
    assert np.array_equal(scaled[labels == 3], np.load(replay['matched_CFM_truth_reference'], allow_pickle=False))

    metric = source / 'TrajectoryNet/optimal_transport/emd.py'
    emd = next(n for n in ast.parse(metric.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'earth_mover_distance')
    assert ast.literal_eval(emd.args.defaults[-1]) == 'sqeuclidean'
    evaluation = source / 'TrajectoryNet/eval_utils.py'
    main = source / 'TrajectoryNet/main.py'
    train_tree = ast.parse(main.read_text())
    train = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    train_eval = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_eval')
    calls = [n for n in ast.walk(train) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'train_eval']
    assert len(calls) == 1 and not any(isinstance(n, ast.Return) for n in ast.walk(train_eval))
    best_assignments = [n for n in ast.walk(train) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'best_loss' for t in n.targets)]
    assert len(best_assignments) == 1 and ast.unparse(best_assignments[0].value) == "float('inf')"

    cases = []
    for density, velocity in itertools.product([0.0, 0.1, 0.01], [0.0, 0.001, 0.0001]):
        variant = 'Base' + ('+D' if density else '') + ('+V' if velocity else '')
        cases.append(dict(family=variant, density_weight=density, velocity_weight=velocity,
                          source_flags=dict(top_k_reg=density, vecint=velocity, use_magnitude=False),
                          status='source_coefficients_bound_native_training_and_full_replay_not_executed'))
    for energy in [1.0, 0.1, 0.01]:
        cases.append(dict(family='Base+E', energy_weight=energy,
                          squared_energy_source_candidate='sl2int',
                          status='pending_EB_Jacobian_coefficient_and_norm_convention_binding'))
    local_seeds = [42, 43, 44, 45, 46]
    jobs = [dict(recipe_index=i, leftout_timepoint=t, local_training_seed=seed, completed=False)
            for i in range(len(cases)) for t in [1, 2, 3] for seed in local_seeds]
    assert len(cases) == 12 and len(jobs) == 180
    pending = [
        'The selected density/velocity/energy coefficients for each published Table3 cell are not supplied.',
        'EB Base+E Jacobian coefficient and exact norm convention need explicit source binding; AppendixE.1 artificial Figure2 coefficients cannot be transplanted.',
        'Growth-prior fits for each leaveout and the exact Base+G recipe remain required.',
        'Mouse cortex raw/processed populations and original artificial-data full comparisons remain in scope.',
        'Original Table3 direction, selection protocol, and metric-generating artifact are not established by numeric coincidence or this snapshot.',
        'Original Torch1.3.1 environment equivalence and original training seeds are uncertified.',
    ]
    bindings = [dict(path=p, sha256=sha(ROOT / p)) for p in [ACQUISITION, REPLAY, MODEL, HELPER,
        'scripts/flow_matching/audit_trajectorynet_original_recipe_and_materials_v1.py',
        'projects/flow_matching_research/ara/paper_method_inventory.json',
        'papers/literature/flow_matching/pdfs/ot_cfm.pdf']]
    plan = dict(schema_version=1, status='source_bound_plan_not_a_running_training_queue',
        citation='https://arxiv.org/abs/2002.04461v2', paper_page=15, source_commit=c['source_commit'],
        source_model_configuration=MODEL, source_maximum_iterations=10000, source_training_batch_size=1000,
        source_solver='dopri5', source_atol=1e-5, source_rtol=1e-5,
        density_neighbours=5, density_hinge_distance=0.1,
        density_grid=[0.0, 0.1, 0.01], velocity_grid=[0.0, 0.001, 0.0001], energy_grid=[1.0, 0.1, 0.01],
        recipes=cases, local_training_seeds=local_seeds, author_training_seeds=None,
        leaveout_timepoints=[1, 2, 3], jobs=jobs, planned_models=180,
        density_velocity_planned_models=135, energy_models_pending_full_recipe_binding=45,
        completed_models=0, dispatched_models=0, automatic_dispatch_enabled=False,
        training_prerequisites=['actual full original checkpoint API replay and guard pass',
            'full original training loss/backward/optimizer and native full-population evaluation API validation',
            'unchanged 10000-step budget, original batch/solver/guards and explicit source checkpoint selection',
            'isolated correction and source-vs-corrected reporting for any proven training bug'],
        unresolved=pending, source_bindings=bindings, all_original_experiments_complete=False,
        boundary='Five local seeds are a declared new robustness protocol, not recovered author seeds. This plan includes the explicit AppendixE.2 grid; it is neither 180 completed experiments nor all 45-paper experiments. Unknown energy and growth choices stay pending.')
    table = {
        'Base': [.764, .811, .863, .813], 'Base+D': [.759, .783, .811, .784],
        'Base+V': [.816, .839, .865, .840], 'Base+D+V': [.930, .806, .810, .848],
        'Base+E': [.737, .896, .842, .825], 'Base+G': [.700, .913, .829, .814],
        'OT': [.791, .831, .841, .821], 'prev': [1.715, 1.400, .814, 1.309],
        'next': [1.400, .814, 1.694, 1.302], 'rand': [.872, 1.036, .998, .969]}
    audit = dict(schema_version=1, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_kind='full_original_material_and_metric_protocol_audit_not_model_performance',
        paper=dict(citation='https://arxiv.org/abs/2002.04461v2', local_path=str(paper),
                   sha256=sha(paper), pages=len(pages), table_page=9, grid_and_versions_page=15,
                   table3_columns=['leave1', 'leave2', 'leave3', 'reported_mean'], table3=table),
        source=dict(repository=c['source_repository'], commit=c['source_commit'], tree=c['source_tree'],
                    complete_original_files_verified=63, every_official_git_blob_verified=True),
        original_checkpoint=checkpoint,
        checkpoint_recipe=dict(variant='Base+D', density_weight=.1, velocity_weight=0.,
            selected_checkpoint_training_step_unknown=True, logged_iterations=10000, locally_trained_models=0),
        full_input_identity=dict(raw_file_sha256=sha(data_path), shape=[16819, 100], counts=counts,
            original_normalization='StandardScaler on all100PCs/all16819cells before first5PC truncation',
            historical_initial_frame_shape=[3332, 5], historical_initial_frame_all_coordinates_bitwise_equal=True,
            matched_CFM_truth_shape=[3665, 5], matched_CFM_truth_all_float32_coordinates_bitwise_equal=True,
            CFM_reference_model_leftout_timepoint=1, compared_TrajectoryNet_leftout_timepoint=3,
            CFM_reference_role='input_identity_only_no_CFM_LO1_model_score_used_as_LO3_comparator'),
        native_metric=dict(default_cost='sqeuclidean', output='square_root_of_exact_transport_objective',
            resulting_metric='W2', euclidean_keyword_result='sqrt(W1)_not_W1',
            original_evaluation_directions=['future_to_leftout', 'past_to_leftout'],
            full_prediction_population_counts=[3332, 3278], full_target_population_count=3665,
            original_target_dtype='float64', additional_matched_target_dtype='float32',
            fixed_additional_primary_direction='past_to_leftout', best_direction_selection_allowed=False),
        later_table=dict(paper='https://arxiv.org/abs/2302.00482', section_page=10, table_page=11,
            declared_metric='1-Wasserstein', TrajectoryNet_EB=.848, Reg_CNF_EB=.825,
            starred_values_are_reused=True, exact_numeric_matches_original_Table3=['Base+D+V_mean', 'Base+E_mean'],
            exact_historical_metric_generation_certified=False,
            standalone_Finlay_RNODE_image_repository_is_proof_of_EB_BaseE_reproduction=False),
        original_entrypoint=dict(unconditional_exit_precedes_evaluate_kantorovich_v2=True,
            replay_calls_unchanged_original_functions=True, author_source_files_modified=False),
        original_training_selection_issue=dict(source='TrajectoryNet/main.py:train/train_eval',
            caller_initial_best_loss='infinity', caller_best_loss_update_count=0,
            train_eval_returns_best_loss=False,
            consequence='For finite validation losses, current pinned source overwrites checkpt.pth at each scheduled validation, rather than retaining the lowest historical value.',
            applies_to_released_historical_checkpoint_without_further_evidence=False,
            corrected_training_variant_executed=False),
        paper_grid_plan=PLAN, unresolved=pending, source_bindings=bindings,
        current_runtime='full original author-checkpoint replay has a separately verified live controller; numerical completion must be read from its actual guarded terminal report',
        all_original_experiments_complete=False, paper_full_baseline_reproduced=False,
        original_scope=replay['original_scope'])
    exclusive(ROOT / PLAN, plan)
    exclusive(ROOT / AUDIT, audit)
    print(json.dumps(dict(audit=AUDIT, plan=PLAN, full_input_identity_verified=True,
        source_files_verified=63, planned_models=180, completed_models=0)))


if __name__ == '__main__':
    generate()
