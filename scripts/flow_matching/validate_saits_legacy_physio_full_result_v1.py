"""Validate unchanged original PhysioNet jobs whose historical config lacks a data hash."""
import ast
from pathlib import Path

from scripts.flow_matching import validate_saits_full_result_v1 as original


def legacy_dataset_sha256(config,result,audit):
    if (config['method'] not in ('saits','brits') or config['phase']!='author_original_protocol'
            or config['id']!=f"saits_paper_{config['method']}_physio_seed{config['seed']}"
            or config['seed'] not in range(26,31) or result['bootstrap_group']!='patient'
            or len(audit['feature_names'])!=37 or audit['splits']['test']['shape'][1:]!=[48,37]
            or audit['splits']['test']['shape'][0]!=result['test_patients']
            or not audit['raw_archives'] or result['dataset_sha256']!=audit['dataset_sha256']
            or config.get('dataset_sha256',result['dataset_sha256'])!=result['dataset_sha256']):
        raise ValueError('Only the unchanged original full37feature48hour PhysioNet patient jobs are accepted')
    return result['dataset_sha256']


def validation_ast():
    source=Path(original.__file__)
    node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    class Binding(ast.NodeTransformer):
        replacements=0
        def visit_Subscript(self,n):
            if isinstance(n.value,ast.Name) and n.value.id=='config' and isinstance(n.slice,ast.Constant) and n.slice.value=='dataset_sha256':
                self.replacements+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='legacy_dataset_sha256',ctx=ast.Load()),
                    args=[ast.Name(id=v,ctx=ast.Load()) for v in ('config','result','audit')],keywords=[]),n)
            return self.generic_visit(n)
    binding=Binding();node=binding.visit(node)
    if binding.replacements!=1:raise ValueError('Only the one historical dataset-binding expression may change')
    return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))


def validate(config_path):
    namespace=dict(vars(original),legacy_dataset_sha256=legacy_dataset_sha256)
    exec(compile(validation_ast(),original.__file__,'exec'),namespace)
    return namespace['validate'](config_path)
