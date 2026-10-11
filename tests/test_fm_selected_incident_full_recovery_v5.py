import ast
import copy

import pytest

from scripts.flow_matching.run_readonly_tab_range_queue_v1 import readonly_main_ast
from scripts.flow_matching.run_selected_incident_full_recovery_v5 import selected_main_ast


def test_every_original_scientific_and_guard_statement_is_retained():
    source = __import__('pathlib').Path('scripts/flow_matching/run_resource_recovery.py').read_text(encoding='utf-8')
    expected = readonly_main_ast(source, 'shared-original-family-mutex')
    actual = selected_main_ast(source, 'shared-original-family-mutex')
    loop = next(n for n in ast.walk(actual) if isinstance(n, ast.For) and isinstance(n.iter, ast.ListComp))
    assert ast.unparse(loop.iter) == "[case for case in queue['jobs'] if case['job']['id'] == selected_job_id]"
    loop.iter = ast.parse("queue['jobs']", mode='eval').body
    assert ast.dump(expected) == ast.dump(actual)


def test_selection_retains_registry_and_dispatches_full_selected_case():
    source = '''def main():
    for case in queue['jobs']:
        seen.append(case)
        with waited_lock(ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock', update):
            pass
'''
    function = selected_main_ast(source, 'mutex')
    loop = next(n for n in ast.walk(function) if isinstance(n, ast.For))
    queue = dict(jobs=[dict(job=dict(id=str(i), epochs=3, width=512, seed=i)) for i in range(13)])
    preserved = copy.deepcopy(queue)
    selected = eval(compile(ast.Expression(loop.iter), '<selected-full-registry>', 'eval'),
                    dict(queue=queue, selected_job_id='7'))
    assert selected == [queue['jobs'][7]] and queue == preserved
    assert len(queue['jobs']) == 13 and selected[0]['job']['epochs'] == 3 and selected[0]['job']['width'] == 512


@pytest.mark.parametrize('source', [
    "def main():\n    for unrelated in []:\n        pass\n",
    "def main():\n    for case in queue['jobs']:\n        pass\n    for case in queue['jobs']:\n        pass\n",
])
def test_missing_or_duplicate_dispatch_loop_is_rejected(source):
    # Preserve the original lock transform while testing the loop cardinality.
    source += "    with waited_lock(ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock', update):\n        pass\n"
    with pytest.raises(ValueError):
        selected_main_ast(source, 'mutex')
