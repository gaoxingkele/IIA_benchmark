"""Receipts and path-only adaptation for full released GiFlow experiments."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def configured_loader(module, data_root):
    def loading(**kwargs):
        provided = kwargs.pop('root', None)
        if provided is not None and Path(provided).resolve() != Path(data_root).resolve():
            raise ValueError('Native caller disagrees with configured data root')
        return module.dataset_loading(root=str(data_root), **kwargs)
    return loading


def verify(queue, root=ROOT):
    for receipt in queue['source_receipts']:
        if sha(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen GiFlow resource changed: ' + receipt['path'])
    ids = [j['id'] for j in queue['jobs']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate experiment slots')
    for job in queue['jobs']:
        args = job['arguments']
        if args['training_epoch'] != 300 or args['batch_size'] != 128 or args['stride'] != 1 or args['window'] != 24:
            raise ValueError('Released full-training budget changed')
        if args['missing_type'] != 'point' or job['track'] not in ('released_mirror', 'reviewed_corrected'):
            raise ValueError('Unsupported native protocol registered as complete')


def path_only_loader(module, data_root):
    """Compile the original loader with exactly its two data-root expressions replaced."""
    source = Path(module.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dataset_loading')

    class Roots(ast.NodeTransformer):
        count = 0

        def visit_JoinedStr(self, node):
            if (len(node.values) == 2 and isinstance(node.values[0], ast.Constant)
                    and node.values[0].value == './data/' and isinstance(node.values[1], ast.FormattedValue)
                    and isinstance(node.values[1].value, ast.Name) and node.values[1].value.id == 'dataset_name'):
                self.count += 1
                return ast.copy_location(ast.Constant(str(data_root)), node)
            return self.generic_visit(node)

    patch = Roots()
    function = patch.visit(function)
    if patch.count != 2:
        raise ValueError('Original data-root expressions changed; adaptation needs review')
    unit = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(module))
    exec(compile(unit, str(source), 'exec'), namespace)
    return namespace['dataset_loading'], {'kind': 'data_root_only', 'expressions_replaced': 2,
                                          'source_sha256': sha(source), 'configured_root': str(data_root)}


def complete(job, root=ROOT):
    path = root / job['output_directory'] / 'result.json'
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'completed' or result.get('diagnostic') is not False or result.get('experiment_sha256') != fingerprint(job):
        raise ValueError('Invalid full GiFlow result binding')
    for item in result['artifacts']:
        if sha(root / item['path']) != item['sha256']:
            raise ValueError('GiFlow result artifact changed: ' + item['path'])
    return True
