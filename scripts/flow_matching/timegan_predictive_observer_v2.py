"""Repair target observation without changing any native TimeGAN computation.

The untouched author predictor feeds X and T for inference, and retains Y_mb
in its caller frame. The v1 observer incorrectly indexes the absent Y feed.
This version changes precisely that observation expression in memory. Frozen
native training workers, sources and recovery registrations remain untouched.
"""
import ast
import copy
import inspect
from pathlib import Path
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha


def native_predictive_targets(feed, target_node, caller, predictions, np):
    if target_node in feed:
        raise ValueError('Untouched author inference must feed only X and T')
    if 'Y_mb' not in caller:
        raise ValueError('Original predictor caller target array is required')
    targets = np.asarray(caller['Y_mb'])
    if targets.shape != np.asarray(predictions).shape or not np.isfinite(targets).all():
        raise ValueError('Every original predictor window/step/target must match')
    return targets


def corrected_observer_ast(tree):
    variant = copy.deepcopy(tree)
    expected = ast.dump(ast.parse("np.asarray(feed[self.native_nodes['Y']])", mode='eval').body)
    replacement = ast.parse(
        "_native_predictive_targets(feed, self.native_nodes['Y'], "
        "inspect.currentframe().f_back.f_locals, result, np)", mode='eval').body
    matches = [0]

    class CorrectTarget(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) == expected:
                matches[0] += 1
                return ast.copy_location(copy.deepcopy(replacement), node)
            return self.generic_visit(node)

    variant = CorrectTarget().visit(variant)
    if matches[0] != 1:
        raise ValueError('Exactly one predictive target observation may change')
    return ast.fix_missing_locations(variant)


def corrected_native_run():
    from scripts.flow_matching import run_timegan_native_job_v1 as original
    tree = corrected_observer_ast(ast.parse(inspect.getsource(original.run)))
    namespace = dict(original.__dict__, _native_predictive_targets=native_predictive_targets)
    exec(compile(tree, original.__file__, 'exec'), namespace)
    return namespace['run']


def corrected_resume_compiler():
    """Retain the exact legacy resume AST and all stage/optimizer/RNG proofs."""
    from scripts.flow_matching import timegan_stage_resume_v2 as original
    tree = ast.parse(inspect.getsource(original.compile_observed_resume))
    compilations = 0
    bindings = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'compile'):
            expected = ast.dump(ast.parse('ast.fix_missing_locations(tree)', mode='eval').body)
            if ast.dump(node.args[0]) != expected:
                raise ValueError('Legacy full observer compiler unexpectedly changed')
            node.args[0] = ast.Call(func=ast.Name(id='_corrected_observer_ast', ctx=ast.Load()),
                args=[node.args[0]], keywords=[])
            compilations += 1
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'namespace' for t in node.targets)):
            if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != 'dict':
                raise ValueError('Legacy complete observer namespace unexpectedly changed')
            node.value.keywords.append(ast.keyword(arg='_native_predictive_targets',
                value=ast.Name(id='_native_predictive_targets', ctx=ast.Load())))
            bindings += 1
    if compilations != 1 or bindings != 1:
        raise ValueError('Only one target patch and one helper namespace binding permitted')
    namespace = dict(original.__dict__, _corrected_observer_ast=corrected_observer_ast,
        _native_predictive_targets=native_predictive_targets)
    exec(compile(ast.fix_missing_locations(tree), original.__file__, 'exec'), namespace)
    return namespace['compile_observed_resume']


def verify_runtime(config):
    for receipt in config['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen corrected observer source changed: ' + receipt['path'])
    from scripts.flow_matching.timegan_native_common_v1 import verify_queue
    queue = read(ROOT / config['original_queue'])
    verify_queue(queue)
    if (config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
            or len(queue['jobs']) != 45 or config['correction'] != 'predictive_observation_only'
            or config['resource_lock'] != queue['resource_lock'] or config['cpu_threads'] != queue['cpu_threads']):
        raise ValueError('Complete original research scope and TimeGAN family retained')
    for role in ('original', 'stock', 'energy'):
        route = config['roles'][role]
        if role == 'original':
            if route['settings'] != queue['settings'] or route['recovery_configuration'] is not None:
                raise ValueError('Original all45 native resource safeguards required')
        else:
            recovery = read(ROOT / route['recovery_configuration'])
            from scripts.flow_matching.timegan_stage_resume_v2 import binding
            binding(recovery)
            for pin in recovery.get('recovery_source_receipts', []):
                if sha(ROOT / pin['path']) != pin['sha256']:
                    raise ValueError('Original frozen full-stage recovery changed')
            expected = copy.deepcopy(queue['settings'])
            if 'retry_admission_commit_headroom_bytes' in recovery:
                from scripts.flow_matching.run_preserved_native_gan_retry_v1 import admission_settings
                expected = admission_settings(recovery, expected)
            if route['settings'] != expected or route['canonical_job_id'] != recovery['canonical_job_id']:
                raise ValueError('Exact same-seed full recovery and existing admission required')
    return queue


def worker(config, role, job_id=None, probe_only=False):
    queue = verify_runtime(config)
    if role == 'original':
        function = corrected_native_run()
        job = next(j for j in queue['jobs'] if j['id'] == job_id)
        queue_path = config['original_queue']
    else:
        recovery = read(ROOT / config['roles'][role]['recovery_configuration'])
        function, queue, job = corrected_resume_compiler()(recovery)
        queue_path = recovery['original_queue']
    if probe_only:
        return dict(passed=True, role=role, canonical_job_id=job['id'], full_original_jobs=45,
            observer_expression_changes=1, author_training_or_scoring_source_changed=False,
            tensorflow_imported='tensorflow' in sys.modules, numpy_imported='numpy' in sys.modules,
            training_executed=False, benchmark_performance=False)
    function(queue, job, queue_path)
    if role == 'original':
        from scripts.flow_matching.run_timegan_native_queue_v1 import verify_result
        verify_result(queue, job)
    else:
        from scripts.flow_matching.timegan_stage_resume_v2 import verify_recovery_result
        verify_recovery_result(recovery)
