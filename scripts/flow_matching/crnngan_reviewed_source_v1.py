"""Explicit paper-grounded optimizer/freezing/curriculum corrections in memory."""
import ast
import copy
from pathlib import Path


def functions(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    main, epoch = copy.deepcopy(definitions['main']), copy.deepcopy(definitions['run_epoch'])
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'run_epoch' and len(n.args) >= 6 and isinstance(n.args[4], ast.Attribute)
             and n.args[4].attr == 'opt_d' and isinstance(n.args[5], ast.Attribute) and n.args[5].attr == 'opt_g']
    if len(calls) != 1:
        raise ValueError('Exactly one original adversarial optimizer swap required')
    calls[0].args[4], calls[0].args[5] = calls[0].args[5], calls[0].args[4]
    loops = [n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == 'i' and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
             and n.iter.func.id == 'range']
    if len(loops) != 1:
        raise ValueError('Exactly one full original epoch loop required')
    conditions = [n for n in loops[0].body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and isinstance(n.test.left, ast.Name) and n.test.left.id == 'global_step']
    if len(conditions) != 1:
        raise ValueError('Exactly one stale curriculum condition required')
    conditions[0].test.left.id = 'i'
    branches = [n for n in ast.walk(epoch) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                and any(isinstance(k, ast.Name) and k.id == 'datasetlabel' for k in ast.walk(n.test))]
    if len(branches) != 1:
        raise ValueError('Exactly one original train-only freezing branch required')
    branches[0].body = ast.parse("""
if d_loss == 0.0 and g_loss == 0.0:
    print('Both G and D train loss are zero. Exiting.')
    break
elif d_loss < 0.7 * g_loss:
    op_d = tf.no_op()
elif g_loss < 0.7 * d_loss:
    op_g = tf.no_op()
""").body
    return ast.fix_missing_locations(main), ast.fix_missing_locations(epoch)


def instrumented_main(path):
    main, epoch = functions(path)
    graph_contexts = [n for n in ast.walk(main) if isinstance(n, ast.With)
                      and any(isinstance(k, ast.Attribute) and k.attr == 'as_default'
                              for item in n.items for k in ast.walk(item.context_expr))]
    if len(graph_contexts) != 1:
        raise ValueError('Exactly one complete original model graph context required')
    graph_contexts[0].body[0:0] = ast.parse('_native_graph_seed()').body
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'i')
    exits = [i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
             and isinstance(n.test, ast.Name) and n.test.id == 'do_exit']
    if len(exits) != 1:
        raise ValueError('Exactly one original post-evaluation exit branch required')
    loop.body[exits[0]:exits[0]] = ast.parse('_native_epoch_complete(i, locals())').body
    tests = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == 'run_epoch' and len(n.value.args) > 3
             and isinstance(n.value.args[3], ast.Str) and n.value.args[3].s == 'test']
    if len(tests) != 1:
        raise ValueError('Exactly one complete original final test evaluation required')
    for parent in ast.walk(main):
        if hasattr(parent, 'body') and isinstance(parent.body, list) and tests[0] in parent.body:
            parent.body[parent.body.index(tests[0]) + 1:parent.body.index(tests[0]) + 1] = ast.parse('_native_test_complete(locals())').body
            break
    module = ast.parse('')
    module.body = [epoch, main]
    return ast.fix_missing_locations(module)
