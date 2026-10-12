"""Release residual temporaries after their final source use, without new math.

CrossAD source: decisionintelligence/CrossAD, frozen revision
ff0fb2d41291b76452898910312dd38ed3389bf3. Every original operation, batch,
attention output, weight, dropout draw and normalization is retained.
"""
import ast
import inspect
import textwrap
from types import MethodType


def residual_lifetime_ast(source,expected_uses):
    tree=ast.parse(textwrap.dedent(source))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Exactly the original residual layer forward function is required')
    function=tree.body[0]
    expected=ast.dump(ast.parse('x = x + self.dropout(x_)').body[0])
    uses=0;body=[]
    for node in function.body:
        body.append(node)
        if ast.dump(node)==expected:
            uses+=1;body.append(ast.parse('del x_').body[0])
    if uses!=expected_uses:
        raise ValueError('Every original residual temporary must have the frozen final-use site')
    function.body=body
    return ast.fix_missing_locations(tree)


def install_residual_lifetime_policy(model):
    changed=[]
    for name,module in model.named_modules():
        kind=type(module).__name__
        if kind not in ('EncoderLayer','DecoderLayer'):
            continue
        source_module=__import__(type(module).__module__,fromlist=[kind])
        if not type(module).__module__.endswith('EncDec'):
            raise ValueError('Only the frozen author encoder/decoder residual implementation may be bound')
        expected=1 if kind=='EncoderLayer' else 2
        tree=residual_lifetime_ast(inspect.getsource(type(module).forward),expected)
        namespace=dict(source_module.__dict__)
        exec(compile(tree,source_module.__file__,'exec'),namespace)
        module.forward=MethodType(namespace['forward'],module)
        changed.append(dict(module=name,kind=kind,released_final_use_references=expected))
    if not changed:
        raise ValueError('A full author model with its original residual layers is required')
    model.residual_reference_lifetime_observations_=changed
    return model


def build_lifetime_memory_model(runtime,parameters):
    from .fm_crossad_author_runtime import build_model
    from .fm_crossad_memory_v2 import install_activation_checkpointing
    model=build_model(runtime,parameters)
    install_residual_lifetime_policy(model)
    return install_activation_checkpointing(model)
