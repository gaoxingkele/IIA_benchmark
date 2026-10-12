import ast

from scripts.flow_matching import shcl_native_observer_v1 as observer


def test_every_original_solver_statement_loop_math_and_budget_is_exact_after_removing_observations():
    model=observer.native.read(observer.native.ROOT/'configs/models/fm_shcl_released_vae_native.v1.json')
    source=(observer.native.verify(model)/'solver.py').read_text(encoding='utf-8')
    tree=observer.solver_ast(source); removed=[]
    class Remove(ast.NodeTransformer):
        def visit_Expr(self,node):
            value=node.value
            if (isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute)
                    and isinstance(value.func.value,ast.Name) and value.func.value.id=='_observer'):
                removed.append(value.func.attr); return None
            return self.generic_visit(node)
    unchanged=Remove().visit(tree)
    assert len(removed)==11
    assert ast.dump(unchanged)==ast.dump(ast.parse(source))
    assert removed.count('before_forward')==4 and removed.count('after_score')==2
    assert removed.count('after_epoch')==removed.count('after_step')==removed.count('before_adjustment')==1


def test_publisher_pinned_environment_is_separate_and_only_NumPy_numerical_version_repaired():
    config=observer.native.read(observer.native.ROOT/'configs/acquisition/fm_shcl_author_environment.v1.json')
    assert config['root'].startswith('F:/aicoding/IIA_Data/environments/')
    assert len(config['wheels'])==14
    wheels={r['package']:r for r in config['wheels']}
    assert wheels['torch']['version']=='1.11.0+cu113'
    assert wheels['numpy']['version']=='1.26.4'
    for name in ('pandas','scikit-learn','scipy','tqdm'):
        assert wheels[name]['version']==config['declared_author_versions'][name]
    for row in config['wheels']:
        assert len(row['sha256'])==64
        assert row['url'].startswith(('https://download-r2.pytorch.org/','https://files.pythonhosted.org/'))
