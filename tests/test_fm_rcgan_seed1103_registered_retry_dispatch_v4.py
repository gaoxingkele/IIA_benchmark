import argparse
import ast
from pathlib import Path
import pytest
from scripts.flow_matching import run_rcgan_seed1103_registered_retry_dispatch_v4 as entry


def test_main_only_disables_outer_parser_abbreviation():
    source=Path(entry.original.__file__).read_text(encoding='utf-8')
    expected=ast.parse(source)
    expected.body=[node for node in expected.body if isinstance(node,ast.FunctionDef) and node.name=='main']
    calls=[node for node in ast.walk(expected) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='ArgumentParser']
    assert len(calls)==1
    calls[0].keywords.append(ast.keyword(arg='allow_abbrev',value=ast.Constant(value=False)))
    assert ast.dump(expected)==ast.dump(entry.worker_safe_main_ast(source))


def test_exact_worker_flag_is_not_swallowed_as_worker_probe_abbreviation():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--worker-probe-only',action='store_true')
    args,remaining=parser.parse_known_args(['--runtime-config','frozen.json','--worker','--proof-binding','binding.json'])
    assert not args.worker_probe_only and remaining==['--worker','--proof-binding','binding.json']
    args,remaining=parser.parse_known_args(['--runtime-config','frozen.json','--worker','--worker-probe-only'])
    assert args.worker_probe_only and remaining==['--worker']


def test_host_dispatch_only_rebinds_the_corrected_worker_entry():
    source=Path(entry.original.previous.original.__file__).read_text(encoding='utf-8')
    expected=entry.original.host_dispatch_ast(source)
    nodes=[node for node in ast.walk(expected) if isinstance(node,ast.Constant)
        and node.value=='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v3']
    assert len(nodes)==1
    nodes[0].value='scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v4'
    assert ast.dump(expected)==ast.dump(entry.host_dispatch_ast(source))


@pytest.mark.parametrize('changed',['missing','already_changed'])
def test_unexpected_native_outer_parser_layout_is_rejected(changed):
    source=Path(entry.original.__file__).read_text(encoding='utf-8')
    if changed=='missing':source=source.replace('argparse.ArgumentParser(description=__doc__,add_help=False)','object()')
    else:source=source.replace('argparse.ArgumentParser(description=__doc__,add_help=False)','argparse.ArgumentParser(description=__doc__,add_help=False,allow_abbrev=False)')
    with pytest.raises(ValueError):entry.worker_safe_main_ast(source)
