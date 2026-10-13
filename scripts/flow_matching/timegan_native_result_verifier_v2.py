"""Retain the exact native result gate without importing host process monitoring.

TimeGAN: Yoon, Jarrett and van der Schaar, NeurIPS 2019; original source and
dataset citations remain in the frozen model/queue. No numerical dependency,
author algorithm, result, checkpoint, evaluator, or tolerance is modified.
"""
import ast
import copy
import inspect

from scripts.flow_matching import timegan_native_common_v1 as common
from scripts.flow_matching import timegan_predictive_observer_v2 as observer
from scripts.flow_matching.run_timegan_native_job_v1 import validate_evaluators


def verifier_ast():
    source = common.ROOT / 'scripts/flow_matching/run_timegan_native_queue_v1.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_result']
    if len(functions) != 1:
        raise ValueError('Exactly one complete original native result verifier required')
    tree.body = [functions[0]]
    return ast.fix_missing_locations(tree)


def verifier_namespace():
    namespace = dict(common.__dict__, validate_evaluators=validate_evaluators)
    exec(compile(verifier_ast(), __file__, 'exec'), namespace)
    return namespace


def verify_result(queue, job):
    return verifier_namespace()['verify_result'](queue, job)


def observer_worker_ast():
    tree = ast.parse(inspect.getsource(observer.worker))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.run_timegan_native_queue_v1':
            if len(node.names) != 1 or node.names[0].name != 'verify_result':
                raise ValueError('Only the host-dependent verifier import may be isolated')
            node.module = 'scripts.flow_matching.timegan_native_result_verifier_v2'
            replacements += 1
    if replacements != 1:
        raise ValueError('Exactly one verification import must change; all native work stays')
    return ast.fix_missing_locations(tree)


def observer_worker():
    namespace = dict(observer.__dict__)
    exec(compile(observer_worker_ast(), __file__, 'exec'), namespace)
    return namespace['worker']
