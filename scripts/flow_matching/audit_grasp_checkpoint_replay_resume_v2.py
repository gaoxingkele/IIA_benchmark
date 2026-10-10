"""Finish all frozen GRASP recipes, reusing only verified complete full replays.

All six original full models are restored again. A preserved prior full recipe
may be reused only after its actual full arrays, rows, original score agreement,
metrics and executed forward counts are verified. Pending recipes use the
unchanged native scoring loop. No training seeds or original scope are dropped.
"""
import ast
import json
import math
from pathlib import Path

from scripts.flow_matching import audit_grasp_checkpoint_replay_v1 as base


def recipe_key(canonical, recipe):
    return canonical, recipe['source_samples'], recipe['flow_evaluations']


def validate_complete_inventory(config):
    registered = {record['canonical_id']: base.read(base.ROOT / record['result_path']) for record in config['full_results']}
    expected = {recipe_key(key, metric['recipe']) for key, result in registered.items() for metric in result['metrics']}
    if len(expected) != config['full_score_recipe_count']:
        raise ValueError('Preserve every original full score recipe')
    observed = set()
    for item in config['reusable_complete_recipes']:
        key = recipe_key(item['canonical_id'], item['recipe'])
        if key not in expected or key in observed:
            raise ValueError('No duplicate or unregistered prior replay recipe')
        observed.add(key)
        if base.sha(base.ROOT / item['proof_path']) != item['proof_sha256']:
            raise ValueError('Preserved prior full replay proof changed')
    if config['pending_full_recipe_count'] != len(expected - observed):
        raise ValueError('All remaining original full recipes must be executed')
    return registered


def verify_reusable_recipes(config):
    import numpy as np
    from scripts.flow_matching.grasp_protocol_runner import metric_records
    import importlib.util

    registered = validate_complete_inventory(config)
    queue = base.verify_registration(config)
    spec = importlib.util.spec_from_file_location('grasp_resume_original_metrics', base.ROOT / queue['native_metrics'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.basic_metricor()
    cache = {}
    for item in config['reusable_complete_recipes']:
        result = registered[item['canonical_id']]
        proof = base.read(base.ROOT / item['proof_path'])
        candidates = [record for record in proof['completed_recipes'] if record['recipe'] == item['recipe']]
        if len(candidates) != 1 or not candidates[0]['full_recipe_reinference_passed']:
            raise ValueError('Prior recipe must already have completed its whole native evaluation')
        record = candidates[0]
        original = next(metric for metric in result['metrics'] if metric['recipe'] == item['recipe'])
        entity = result['data_audit']
        if (base.sha(record['replay_score_path']) != record['replay_score_sha256']
                or base.sha(record['original_score_path']) != record['original_score_sha256']
                or record['original_score_path'] != original['scores_path']):
            raise ValueError('Full prior native score files changed')
        parameters = base.read(base.ROOT / result['job']['model_config'])['parameters']
        with np.load(record['replay_score_path'], allow_pickle=False) as actual, np.load(original['scores_path'], allow_pickle=False) as saved:
            base.equal_arrays(actual['labels'], saved['labels'], 'complete original labels')
            base.equal_arrays(actual['test_row_ids'], saved['test_row_ids'], 'complete test row identities')
            base.equal_arrays(actual['validation_raw_row_ids'], saved['validation_raw_row_ids'], 'complete validation identities')
            for tag in ('validation', 'test'):
                count = result['job'][tag + '_points']
                if len(actual[tag + '_scores']) != count or record['complete_' + tag + '_points'] != count:
                    raise ValueError('All prior full points required')
                base.compare_full_scores(actual[tag + '_scores'], saved[tag + '_scores'], config['score_tolerance'])
                window_count = count - parameters['window'] + 1
                scale = item['recipe']['source_samples'] * item['recipe']['flow_evaluations']
                calls = dict(count=math.ceil(window_count / parameters['batch_size']) * scale, windows=window_count * scale)
                if record['forward_evaluation_counts'][tag] != calls:
                    raise ValueError('Full actually executed prior original model forward budget required')
            metrics = metric_records(native, actual['labels'], actual['test_scores'], actual['validation_scores'])
            base.compare_metrics(metrics, original['metrics'], config['metric_tolerance'])
        cache[recipe_key(item['canonical_id'], item['recipe'])] = dict(record,
            prior_full_replay_proof_path=item['proof_path'], prior_full_replay_proof_sha256=item['proof_sha256'],
            reused_complete_full_recipe=True, pending_recipe_replaced_with_partial=False)
    return cache


def worker(config, config_path):
    cache = verify_reusable_recipes(config)
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'worker')
    changed = 0
    for node in ast.walk(target):
        if isinstance(node, ast.For) and ast.unparse(node.target) == 'original' and ast.unparse(node.iter) == "result['metrics']":
            node.body[0:0] = ast.parse('full_scores = {}').body
            # Insert after original recipe binding; all model restoration and
            # remaining native score loops remain exactly the original code.
            bridge = ast.parse("""
cached_key = recipe_key(registered['canonical_id'], recipe)
if cached_key in reusable_cache:
    job_records.append(reusable_cache[cached_key])
    write(target / (registered['canonical_id'] + '.json'), dict(canonical_id=registered['canonical_id'], execution_id=job['id'], completed_recipes=job_records))
    continue
""").body
            node.body[2:2] = bridge
            changed += 1
    if changed != 1:
        raise ValueError('Exactly one native per-recipe reuse boundary required')
    scope = dict(base.__dict__, reusable_cache=cache, recipe_key=recipe_key)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[target], type_ignores=[])), base.__file__, 'exec'), scope)
    scope['worker'](config, config_path)


def main():
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    replaced = 0
    for node in ast.walk(target):
        if isinstance(node, ast.Constant) and node.value == 'scripts.flow_matching.audit_grasp_checkpoint_replay_v1':
            node.value = 'scripts.flow_matching.audit_grasp_checkpoint_replay_resume_v2'
            replaced += 1
    if replaced != 1:
        raise ValueError('Exactly one controller child module binding required')
    scope = dict(base.__dict__, worker=worker)
    exec(compile(ast.Module(body=[target], type_ignores=[]), base.__file__, 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
