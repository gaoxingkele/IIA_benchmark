"""Bind official S4 conjugate Cauchy fallback without editing downloaded source."""
import ast
import hashlib
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_function(source):
    tree=ast.parse(source)
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='_conj' for t in n.targets)]
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cauchy_naive']
    if len(assignments)!=1 or len(functions)!=1:
        raise ValueError('Official conjugate Cauchy implementation boundary changed')
    if [a.arg for a in functions[0].args.args]!=['v','z','w','conj']:
        raise ValueError('Official conjugate Cauchy signature changed')
    return ast.fix_missing_locations(ast.Module(body=assignments+functions,type_ignores=[]))


def install(module,settings,root):
    source=root/settings['official_cauchy_source']
    if digest(source)!=settings['official_cauchy_source_sha256']:
        raise ValueError('Frozen official Cauchy source changed')
    if digest(module.__file__)!=settings['ls4_s4_source_sha256']:
        raise ValueError('Frozen LS4 S4 source changed')
    if module.has_cauchy_extension or module.has_pykeops:
        raise ValueError('Registered correction requires the observed naive fallback backend')
    namespace=dict(torch=module.torch)
    exec(compile(official_function(source.read_text(encoding='utf-8')),str(source),'exec'),namespace)
    module.cauchy_naive=namespace['cauchy_naive']
    return dict(kind='official_S4_conjugate_symmetric_naive_Cauchy',
        official_source=settings['official_cauchy_source'],official_source_sha256=digest(source),
        ls4_source_sha256=digest(module.__file__),downloaded_original_source_unchanged=True,
        algorithmic_correction=True,author_equivalence_certified=False)
