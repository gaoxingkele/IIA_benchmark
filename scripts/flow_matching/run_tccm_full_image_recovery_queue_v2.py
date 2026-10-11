"""Wait for both capacity report and terminal receipt before full TCCM dispatch."""
import ast
from pathlib import Path

from scripts.flow_matching import run_tccm_full_image_recovery_queue_v1 as original


def capacity_receipts_pending(config):
    output = original.ROOT / config['capacity_cases'][0]['output_directory']
    location = output / 'capacity_probe_result.json'
    if not location.exists():
        return True
    record = original.read(location)
    report = original.ROOT / record['path']
    return not report.exists() or not (report.parent / 'terminal.json').exists()


def ready_main_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(tree.body) != 1:
        raise ValueError('Exactly one original full TCCM recovery controller required')
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    if len(loops) != 1 or 'capacity_probe_result.json' not in ast.unparse(loops[0].test):
        raise ValueError('Exactly one original native capacity readiness wait required')
    loops[0].test = ast.parse('_capacity_receipts_pending(config)', mode='eval').body
    return ast.fix_missing_locations(tree)


def bind():
    path = Path(original.__file__)
    namespace = dict(original.__dict__, _capacity_receipts_pending=capacity_receipts_pending)
    exec(compile(ready_main_ast(path.read_text(encoding='utf-8')), str(path), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    bind()['main']()
