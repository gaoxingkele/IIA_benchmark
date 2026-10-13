"""Correct the post-audit metadata field to frozen observed_states schema.

The failed v1 audit and its immutable registration remain preserved. No original
native observation, model, solver, full-array checks or metric calculation changes.
"""
import ast
import inspect
from types import FunctionType
from scripts.flow_matching import audit_cfm_full80_observed_arrays_v1 as previous


def corrected_audit_ast():
    tree = ast.parse(inspect.getsource(previous.audit))
    class CorrectMetadata(ast.NodeTransformer):
        count = 0
        def visit_Subscript(self,node):
            if (isinstance(node.value,ast.Name) and node.value.id == 'source'
                    and isinstance(node.slice,ast.Constant) and node.slice.value == 'train_states'):
                self.count += 1
                node.slice.value = 'observed_states'
            return self.generic_visit(node)
    change = CorrectMetadata()
    tree = change.visit(tree)
    if change.count != 1:
        raise ValueError('Only the exact observed training-state metadata key may change')
    return ast.fix_missing_locations(tree)


values = dict(previous.__dict__)
exec(compile(corrected_audit_ast(),__file__,'exec'),values)
audit = values['audit']


def main():
    FunctionType(previous.main.__code__,dict(previous.__dict__,audit=audit))()


if __name__ == '__main__':
    main()
