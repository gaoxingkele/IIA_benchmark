"""Preserve the complete author tree and repair only native runtime plumbing."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repair_entry(source, regular):
    """Keep all equations, solver settings and loop bodies; fix Path arithmetic."""
    tree = ast.parse(source)
    changes = []
    replacements = []
    if regular:
        for name in ('embedder_1000.pt', 'recovery_1000.pt', 'embedder_5000.pt', 'recovery_5000.pt'):
            old = 'path+"/' + name + '"'
            if source.count(old) != 1:
                raise ValueError('Pinned regular Path repair target differs: ' + name)
            replacements.append((old, 'path/"' + name + '"'))
        marker = 'args.save_dir = here / args.save_dir'
        if source.count(marker) != 1:
            raise ValueError('Pinned regular output directory target differs')
        replacements.append((marker, marker + '\n    os.makedirs(args.save_dir, exist_ok=True)'))
    for old, new in replacements:
        source = source.replace(old, new)
        changes.append(dict(before=old, after=new, scope='filesystem_only'))
    repaired = ast.parse(source)
    before_classes = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.ClassDef)}
    after_classes = {n.name: ast.dump(n, include_attributes=False) for n in repaired.body if isinstance(n, ast.ClassDef)}
    if before_classes != after_classes:
        raise ValueError('Original native network classes changed')
    # Reversing the enumerated filesystem repairs must restore the whole AST.
    restored = source
    for old, new in reversed(replacements):
        restored = restored.replace(new, old)
    if ast.dump(ast.parse(restored), include_attributes=False) != ast.dump(tree, include_attributes=False):
        raise ValueError('An unregistered algorithm change was introduced')
    return source, changes


def drop_unused_imports(source, modules):
    """Remove optional imports unused by the selected original stock task."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    drops = []
    for node in tree.body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name in modules:
            name = node.names[0].name
            bound = node.names[0].asname or name.split('.')[0]
            if any(isinstance(n, ast.Name) and n.id == bound for n in ast.walk(tree)):
                raise ValueError('Optional import is used in this source: ' + name)
            drops.append((node.lineno, node.end_lineno, name))
    if {x[2] for x in drops} != set(modules):
        raise ValueError('Expected optional import set differs')
    for start, end, _ in reversed(drops):
        lines[start - 1:end] = []
    result = ''.join(lines)
    expected = ast.Module(body=[n for n in tree.body if not (isinstance(n, ast.Import) and
        len(n.names) == 1 and n.names[0].name in modules)], type_ignores=[])
    if ast.dump(ast.parse(result), include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Non-import source changed')
    return result, [dict(module=x[2], scope='unused_optional_import_only') for x in drops]


def prepare(config, destination):
    source = ROOT / config['source_root']
    dest = Path(destination)
    if not dest.is_absolute() or not dest.resolve().is_relative_to(Path(config['artifact_root']).resolve()):
        raise ValueError('Runtime must stay in the registered F-drive artifact directory')
    if dest.exists():
        raise FileExistsError('Preserve existing runtime and all training attempts')
    receipt = json.loads((ROOT / config['acquisition_receipt']).read_text(encoding='utf-8'))
    assets = receipt['assets']
    if len(assets) != 245 or not receipt['all_registered_author_blobs_verified']:
        raise ValueError('The complete pinned author tree is required')
    for item in assets:
        path = source / item['preserved_relative_path']
        if digest(path) != item['sha256']:
            raise ValueError('Original author asset changed: ' + str(path))
    dest.mkdir(parents=True)
    copied = []
    for item in assets:
        rel = item['preserved_relative_path']
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, out)
        copied.append(dict(path=rel, original_sha256=item['sha256']))
    repairs = []
    for filename, regular in [('GTGAN_stocks.py', True), ('GTGAN_stocks_irregular.py', False)]:
        out = dest / filename
        text, changes = repair_entry(out.read_text(encoding='utf-8'), regular)
        text, imports = drop_unused_imports(text, {'gru_ode_bayes.data_utils'})
        out.write_text(text, encoding='utf-8', newline='')
        repairs.append(dict(path=filename, repaired_sha256=digest(out), changes=changes + imports,
                            all_original_network_classes_unchanged=True, full_ast_reversal_verified=True))
    out = dest / 'time_dataset.py'
    text, changes = drop_unused_imports(out.read_text(encoding='utf-8'), {'torchaudio', 'sktime.utils.load_data'})
    out.write_text(text, encoding='utf-8', newline='')
    repairs.append(dict(path='time_dataset.py', repaired_sha256=digest(out), changes=changes,
                        all_native_data_processing_unchanged=True))
    for item in assets:
        if digest(source / item['preserved_relative_path']) != item['sha256']:
            raise ValueError('Raw downloaded files must remain unchanged')
    proof = dict(source_commit=config['source_commit'], complete_copied_assets=copied, runtime_repairs=repairs,
                 original_source_unchanged=True, models_reduced=False, training_budget_reduced=False,
                 author_equivalence_certified=False, boundary=config['boundary'])
    (dest / 'runtime_preparation_receipt.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--destination', required=True)
    args = p.parse_args()
    c = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    proof = prepare(c, args.destination)
    print(dict(assets=len(proof['complete_copied_assets']), repaired_files=len(proof['runtime_repairs'])))


if __name__ == '__main__':
    main()
