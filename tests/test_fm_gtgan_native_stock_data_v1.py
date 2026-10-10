import ast
from pathlib import Path

import pytest

from scripts.flow_matching.audit_gtgan_native_stock_data_v1 import ROOT,original_loader_module


def test_complete_original_loader_scope_has_only_two_cpu_device_bindings():
    source=(ROOT/'data/public_datasets/flow_matching/gtgan_native_source_v1/original/time_dataset.py').read_text(encoding='utf-8')
    actual=original_loader_module(source)
    text=ast.unparse(actual)
    assert {n.name for n in actual.body}=={'normalize','to_tensor','load_data','save_data','TimeDataset_regular','TimeDataset_irregular'}
    assert text.count(".to('cpu')")==2 and '.cuda(' not in text
    assert 'manual_seed(56789)' in text and 'natural_cubic_spline_coeffs' in text
    assert 'range(len(norm_data) - seq_len + 1)' in text
    assert 'torch.randperm(norm_data.shape[0]' in text
    # Reversing only the two device substitutions yields exactly original AST.
    class Restore(ast.NodeTransformer):
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and len(n.args)==1 and isinstance(n.args[0],ast.Constant) and n.args[0].value=='cpu':
                n.func.attr='cuda';n.args=[]
            return n
    restored=Restore().visit(actual)
    original=ast.parse(source)
    names={n.name for n in restored.body}
    expected=ast.Module(body=[n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[])
    assert ast.dump(restored)==ast.dump(expected)


def test_missing_class_or_new_device_call_cannot_silently_change_audit_scope():
    source=(ROOT/'data/public_datasets/flow_matching/gtgan_native_source_v1/original/time_dataset.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError):original_loader_module(source.replace('class TimeDataset_regular','class Unrelated'))
    with pytest.raises(ValueError):original_loader_module(source.replace('.float().cuda()','.float().cuda(1)'))
