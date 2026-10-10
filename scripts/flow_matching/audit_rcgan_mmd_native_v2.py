"""Diagnose full native cache fidelity, reusing the complete144M-pair audit."""
import argparse
import ast
import copy
from pathlib import Path
from scripts.flow_matching import audit_rcgan_mmd_native_v1 as base


def run(config):
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    first = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'original_sqrt')
    last = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                and isinstance(n.targets[0].elts[0], ast.Name) and n.targets[0].elts[0].id == 'Z')
    function.body[first:last] = ast.parse("""
if sha(ROOT / config['completed_full_bandwidth_audit']) != config['completed_full_bandwidth_audit_sha256']:
    raise ValueError('Completed full original144M-pair audit changed')
median_evidence = read(ROOT / config['completed_full_bandwidth_audit'])
if median_evidence['full_original_pairs'] != 144000000 or not median_evidence['original_bandwidth_finite']:
    raise ValueError('Verified complete finite original bandwidth required')
bandwidth = median_evidence['original_bandwidth']
median_evidence = dict(median_evidence, reused_complete_original_pairwise_audit=True)
write(ROOT / config['output_directory'] / 'full_pairwise_bandwidth_audit.json', median_evidence)
_source_grams = {}
kernel_tree = ast.parse(Path(mmd.__file__).read_text(encoding='utf-8'))
kernel_function = next(n for n in kernel_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_mix_rbf_kernel')
kernel_function.body.insert(-1, ast.parse('_record_source_grams(X, XX, XY, YY)').body[0])
def record_source_grams(X, XX, XY, YY):
    _source_grams['fit' if int(X.shape[0]) == 996 else 'test'] = dict(XX=XX, XY=XY, YY=YY)
kernel_scope = dict(mmd.__dict__, _record_source_grams=record_source_grams)
exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel_function])), mmd.__file__, 'exec'), kernel_scope)
mmd._mix_rbf_kernel = kernel_scope['_mix_rbf_kernel']
""").body
    count = 0
    for node in ast.walk(function):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and [x.id for x in node.target.elts if isinstance(x, ast.Name)] == ['actual', 'expected']:
            parent_loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and node in n.body)
            position = parent_loop.body.index(node)
            parent_loop.body[position:position] = ast.parse("""
native_gram_values = session.run(_source_grams[tag], feed_dict=feed)
manual_gram_values = session.run(case['native_grams'], feed_dict=feed)
diagnostic = dict(phase=tag, full_points=len(eval_real[selection]),
    original_metrics=np.asarray(original_values).tolist(), cached_metrics=np.asarray(cached_values).tolist(),
    original_sigma_gradient=np.asarray(original_grad).tolist(), cached_sigma_gradient=np.asarray(cached_grad).tolist(),
    native_gram_maximum_errors={key:float(np.max(np.abs(native_gram_values[key]-manual_gram_values[key]))) for key in native_gram_values},
    native_gram_bitwise_equal={key:bool(np.array_equal(native_gram_values[key],manual_gram_values[key])) for key in native_gram_values},
    original_tolerance=config['numerical_tolerance'])
write(ROOT / config['output_directory'] / ('full_native_cache_numeric_diagnostic_' + tag + '.json'), diagnostic)
print(json.dumps(diagnostic), flush=True)
""").body
            count += 1
            break
    if count != 1:
        raise ValueError('Exactly one full numeric comparison boundary required')
    scope = dict(base.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function])), base.__file__, 'exec'), scope)
    return scope['audit'](config)


def main():
    base.audit = run
    base.main()


if __name__ == '__main__':
    main()
