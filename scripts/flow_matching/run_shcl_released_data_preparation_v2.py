"""Use the exact v1 guards; change only the fresh worker's reference-audit module."""
import ast
import inspect

from scripts.flow_matching import run_shcl_released_data_preparation_v1 as original


def controller_ast():
    tree=ast.parse(inspect.getsource(original.main)); changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.prepare_shcl_released_data_v1':
            node.value='scripts.flow_matching.prepare_shcl_released_data_v2'; changes+=1
    if changes!=1: raise ValueError('Exactly one fresh worker dispatch changes; all original caps remain')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(original.__dict__)
    exec(compile(controller_ast(),__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
