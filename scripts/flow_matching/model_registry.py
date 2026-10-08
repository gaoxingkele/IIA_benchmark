"""Instantiate explicitly configured local implementations; no alias guessing."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def load_model_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def build_configured_method(config, overrides=None):
    if not isinstance(config,dict):config=load_model_config(config)
    entry=config.get('entrypoint')
    if not entry:
        raise ValueError('No callable registered: '+config.get('reproduction_status','unknown'))
    module,symbol=entry.split(':',1)
    if not module.startswith('iia_benchmark.models.'):
        raise ValueError('Expected an explicit local model entrypoint')
    implementation=getattr(importlib.import_module(module),symbol)
    if not callable(implementation):raise TypeError('Configured symbol is not callable')
    parameters={**config.get('parameters',{}),**(overrides or {})}
    return implementation(**parameters)


def verify_config(path, root=ROOT):
    config=load_model_config(path);entry=config.get('entrypoint')
    record={'config':Path(path).relative_to(root).as_posix(),'id':config['id'],
            'task':config.get('task'),'reproduction_status':config.get('reproduction_status'),
            'config_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if not entry:return {**record,'status':'explicitly_non_callable','boundary':'Source-only or pending adapter; not a runnable method.'}
    module,symbol=entry.split(':',1)
    source=root/'src'/Path(*module.split('.')).with_suffix('.py')
    if not source.is_file():raise ValueError('Missing implementation source: '+entry)
    tree=ast.parse(source.read_bytes())
    if not any(isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==symbol for n in tree.body):
        raise ValueError('Missing top-level implementation symbol: '+entry)
    if not config.get('citation') and not config.get('citations'):
        raise ValueError('Missing literature citation: '+record['id'])
    return {**record,'status':'local_symbol_verified','entrypoint':entry,
            'implementation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'boundary':'Source/citation verification, not training or paper equivalence acceptance.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    print(json.dumps([verify_config(path) for path in args.config],ensure_ascii=True,indent=2))


if __name__=='__main__':main()
