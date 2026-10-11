"""Disable CLI abbreviation so --worker enters the full frozen Py37 runtime."""
import ast
from pathlib import Path
from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v3 as original

ROOT,read,sha,verify=original.ROOT,original.read,original.sha,original.verify


def worker_safe_main_ast(source):
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main']
    if len(tree.body)!=1:raise ValueError('One complete direct native worker entry point required')
    matches=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='ArgumentParser':
            if any(keyword.arg=='allow_abbrev' for keyword in node.keywords):
                raise ValueError('Expected the original outer parser abbreviation setting')
            node.keywords.append(ast.keyword(arg='allow_abbrev',value=ast.parse('False',mode='eval').body));matches+=1
    if matches!=1:raise ValueError('Only the outer native-entry parser abbreviation may change')
    return ast.fix_missing_locations(tree)


def host_dispatch_ast(source):
    tree=original.host_dispatch_ast(source)
    matches=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v3':
            node.value='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v4';matches+=1
    if matches!=1:raise ValueError('Exactly one compatible native worker entry module required')
    return ast.fix_missing_locations(tree)


def main():
    source=Path(original.__file__)
    namespace=dict(original.__dict__,host_dispatch_ast=host_dispatch_ast)
    exec(compile(worker_safe_main_ast(source.read_text(encoding='utf-8')),str(source),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
