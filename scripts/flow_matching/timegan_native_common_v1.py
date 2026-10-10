"""Python 3.7 compatible registration helpers for the untouched TimeGAN source."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def full_budget(model):
    p = model['parameters']
    if p != dict(seq_len=24, hidden_dim=24, num_layer=3, iterations=50000,
                 batch_size=128, metric_iteration=10):
        raise ValueError('Full published source defaults must remain frozen')
    return dict(E0=50000, GS=50000, G=100000, E=100000, D_maximum=50000,
                discriminative_repeats=10, discriminative_updates_per_repeat=2000,
                predictive_repeats=10, predictive_updates_per_repeat=5000)


def verify_counts(counts, model):
    expected = full_budget(model)
    if set(counts) != {'E0', 'E', 'D', 'G', 'GS'}:
        raise ValueError('All five original optimizers must be observed')
    if any(counts[k] != expected[k] for k in ('E0', 'GS', 'G', 'E')):
        raise ValueError('Full three-stage original training budget is incomplete')
    if not 0 <= counts['D'] <= expected['D_maximum']:
        raise ValueError('Original loss-gated discriminator budget differs')


def verify_source(model):
    source = ROOT / model['source_root']
    manifest = read(ROOT / model['source_manifest'])
    if manifest['commit'] != model['source_commit'] or len(manifest['files']) != 13:
        raise ValueError('Original complete thirteen-file source snapshot required')
    for item in manifest['files']:
        if sha(source / item['path']) != item['sha256']:
            raise ValueError('Untouched author source or raw data changed: ' + item['path'])
    return source, manifest


def constructor_prefix(module, function_name):
    """Compile original graph statements before Session, exclusively for capacity audits.

    The actual training worker calls the untouched original function in full.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    prefix = copy.deepcopy(original)
    cutoff = next(i for i, n in enumerate(prefix.body)
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sess' for t in n.targets))
    prefix.name = function_name + '_constructor_only'
    prefix.body = prefix.body[:cutoff] + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))]
    unit = ast.fix_missing_locations(ast.Module(body=[prefix]))
    namespace = dict(module.__dict__)
    exec(compile(unit, module.__file__, 'exec'), namespace)
    return namespace[prefix.name]


def verify_queue(queue):
    for item in queue['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen native registration changed: ' + item['path'])
    model = read(ROOT / queue['model_config'])
    full_budget(model)
    verify_source(model)
    expected = {(d, m, s) for d in ('stock', 'energy', 'sine')
                for m in ('gru', 'lstm', 'lstmLN') for s in (1103, 1104, 1105, 1106, 1107)}
    identities = [(j['dataset'], j['module'], j['seed']) for j in queue['jobs']]
    if len(identities) != 45 or set(identities) != expected or len({j['id'] for j in queue['jobs']}) != 45:
        raise ValueError('All original dataset/module choices and five independent seeds required')
    return model
