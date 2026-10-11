"""Full2000epoch WT image comparison, all related native integration recipes."""
import ast
from itertools import product
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.wt_full_image_training_contract_v1 import validate_wt_job, WT_PARAMETERS


def native_solver_facts(source_receipts):
    """Read pinned AST only; never import author feature models or logging code."""
    verify_receipts(source_receipts)
    trees = {item['role']: ast.parse((ROOT / item['path']).read_text(encoding='utf-8'))
             for item in source_receipts}
    if set(trees) != {'main', 'infer_config', 'infer'}:
        raise ValueError('All original WT solver/default sources required')
    defaults = [ast.literal_eval(k.value) for n in ast.walk(trees['main'])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'add_argument' and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == '--step' for k in n.keywords if k.arg == 'default']
    infer_defaults = [ast.literal_eval(n.value) for n in trees['infer_config'].body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'step' for t in n.targets)]
    function = next(n for n in trees['infer'].body if isinstance(n, ast.FunctionDef) and n.name == 'step_test')
    steps = [ast.literal_eval(n.value) for n in function.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'steps' for t in n.targets)]
    categories = [ast.literal_eval(n.value) for n in function.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'mv_names' for t in n.targets)]
    generator = next(n for n in trees['infer'].body if isinstance(n, ast.FunctionDef) and n.name == 'generate_pdf')
    assignments = {t.id: ast.unparse(n.value) for n in ast.walk(generator) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    loops = [n for n in generator.body if isinstance(n, ast.For)]
    if (defaults != [50] or infer_defaults != [1] or steps != [[1, 10, 20, 50, 100, 200]]
            or categories[-1] != ['transistor'] or assignments['dt'] != '1.0 / step'
            or len(loops) != 1 or ast.unparse(loops[0].iter) != 'range(step)'
            or not any(isinstance(n, ast.Assign) and ast.unparse(n.value) == 'j * dt' for n in loops[0].body)
            or assignments['x_t'] != 'rf.euler(x_t, v_pred, dt)'
            or assignments['logp'] != '-0.5 * torch.mean(x_t ** 2, 1)'
            or assignments['LNWT'] != 'nn.LayerNorm(x_t[0].shape).to(c.device)'):
        raise ValueError('Pinned original native WT step scan/defaults differ')
    return dict(training_CLI_default_steps=defaults[0], inference_default_steps=infer_defaults[0],
        native_step_scan=steps[0], active_native_step_scan_categories=categories[-1],
        native_integration='left_grid_Euler', native_time='j*(1/steps)', interval=[0., 1.],
        comprehensive_all15category_scan_is_local_extension=True,
        original_target_solver_and_step_identity_certified=False)


def parameters(steps):
    return dict(epsilon=WT_PARAMETERS['epsilon'], integration_steps=steps,
        integration='related_native_left_grid_Euler', interval=[0., 1.],
        energy='positive_one_half_channel_SUM_endpoint_squared', top_fraction=.01)


def validate_queue(queue):
    evaluation = read(ROOT / queue['evaluation_model_configuration'])
    if (queue['solver_facts'] != native_solver_facts(evaluation['native_solver_source_receipts'])
            or evaluation['solver_facts'] != queue['solver_facts']
            or evaluation['target_heatmap'] != parameters(1)['energy']
            or evaluation['original_target_solver_and_step_identity_certified']
            or queue['original_target_solver_and_step_identity_certified']
            or queue['original_author_equivalence_certified']
            or queue['original_WT_best_feature_setting_reproduced'] or queue['strict_TAB_result']
            or queue['solver_selected_using_test_labels'] or queue['all_original_experiments_complete']):
        raise ValueError('Explicit original WT source and uncertified target boundary required')
    protocol = read(ROOT / queue['original_protocol'])
    index, row = next((i, r) for i, r in enumerate(protocol['Table2_same_backbone_FM_comparison'])
                      if r['method'] == 'WT-Flow')
    expected = set(product(range(1103, 1108), queue['solver_facts']['native_step_scan']))
    observed = {(x['seed'], x['score_parameters']['integration_steps']) for x in queue['cases']}
    if (len(queue['cases']) != 30 or observed != expected
            or len({x['id'] for x in queue['cases']}) != 30
            or len({x['output_directory'] for x in queue['cases']}) != 30):
        raise ValueError('All six native budgets and five full independent WT training seeds required')
    parent = read(ROOT / queue['parent_full_image_evaluation_queue'])
    if queue['metrics'] != parent['metrics'] or queue['verification'] != parent['verification']:
        raise ValueError('Frozen full-test metrics and unchanged replay tolerances required')
    for case in queue['cases']:
        steps = case['score_parameters']['integration_steps']
        if (isinstance(steps, bool) or not isinstance(steps, int)
                or case['score_parameters'] != parameters(steps)
                or case['paper_table_key'] != 'Table2_same_backbone_FM_comparison'
                or case['paper_row_index'] != index or case['paper_reported'] != row['paper_reported']
                or case['dataset'] != 'mvtec' or case['original_dataset_name'] != row['dataset']
                or case['variant'] != 'conditional' or case['checkpoint_epoch'] != 2000):
            raise ValueError('Full WT target table/image-domain/native sensitivity recipe differs')
        job = read(ROOT / case['training_configuration'])
        model = read(ROOT / job['model_configuration'])
        validate_wt_job(job, model, read(ROOT / job['optimizer_configuration']), read(ROOT / job['dataset_configuration']))
        if (job['seed'] != case['seed'] or model['native_source'] != evaluation['native_source']
                or case['checkpoint'] != job['output_directory'] + '/epoch_2000.pt'):
            raise ValueError('Full2000epoch original WT checkpoint/source identity differs')
    return True


def complete(case):
    path = ROOT / case['output_directory'] / 'fully_verified_score_result.json'
    if not path.exists():
        return False
    location = read(path)
    if sha(location['report']) != location['report_sha256'] or location.get('passed') is not True:
        raise ValueError('Complete immutable independent WT audit required')
    report = read(location['report'])
    required = ('passed', 'every_native_image_model_replayed', 'all_native_heatmaps_and_image_scores_equal',
                'original_full_state_dictionary_loaded_strictly', 'independent_all_category_metrics_passed',
                'native_performance_measured')
    if (any(report.get(k) is not True for k in required) or report['case'] != case
            or report['complete_original_test_images'] != 1725 or report['complete_original_categories'] != 15
            or report['full_test_velocity_calls'] != 1725 * case['score_parameters']['integration_steps']
            or report['original_author_equivalence_certified']
            or report['original_WT_best_feature_setting_reproduced'] or report['strict_TAB_result']
            or report['original_target_solver_and_step_identity_certified'] or report['solver_selected_using_test_labels']
            or report['all_original_experiments_complete']
            or sha(ROOT / case['checkpoint']) != report['full_original_checkpoint_sha256']
            or report['queue_sha256'] != sha(ROOT / report['queue_configuration'])):
        raise ValueError('Every native WT image/category/full checkpoint/solver boundary required')
    candidate_path = report['native_score_candidate']
    if sha(candidate_path) != report['native_score_candidate_sha256']:
        raise ValueError('Full native WT score candidate changed')
    candidate = read(candidate_path)
    if (candidate['case'] != case or candidate['complete_original_test_images'] != 1725
            or candidate['training_binding'] != report['training_binding']
            or candidate['original_checkpoint_sha256'] != report['full_original_checkpoint_sha256']):
        raise ValueError('Full native WT score/checkpoint binding differs')
    verify_receipts(candidate['output_receipts']); verify_receipts(candidate['native_truth_array_receipts'])
    return True
