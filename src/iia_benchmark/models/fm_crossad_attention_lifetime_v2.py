"""Shorten only dead author attention-score references after source softmax.

Source: decisionintelligence/CrossAD ff0fb2d41291b76452898910312dd38ed3389bf3.
Original score computation, masks, softmax, dropout and products are retained.
This is a memory policy for original-code reproduction, not a new method.
"""
import ast
import inspect
import textwrap
from types import MethodType


def attention_lifetime_ast(source):
    tree=ast.parse(textwrap.dedent(source))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Require the original attention forward function')
    function=tree.body[0]
    expected=ast.dump(ast.parse('attn_weights = F.softmax(attn_scores, dim=-1)').body[0])
    matches=[i for i,n in enumerate(function.body) if ast.dump(n)==expected]
    if len(matches)!=1:raise ValueError('Require exactly the frozen source softmax site')
    last=matches[0]
    if any(isinstance(n,ast.Name) and n.id=='attn_scores' for tail in function.body[last+1:] for n in ast.walk(tail)):
        raise ValueError('A score reference used later cannot be released')
    function.body.insert(last+1,ast.parse('del attn_scores').body[0])
    return ast.fix_missing_locations(tree)


def install_attention_lifetime_policy(model):
    changed=[]
    for name,module in model.named_modules():
        if type(module).__name__!='ScaledDotProductAttention':continue
        if not type(module).__module__.endswith('Attention_Blocks'):
            raise ValueError('Only the frozen original author attention can be rebound')
        source_module=__import__(type(module).__module__,fromlist=['ScaledDotProductAttention'])
        tree=attention_lifetime_ast(inspect.getsource(type(module).forward))
        namespace=dict(source_module.__dict__)
        exec(compile(tree,source_module.__file__,'exec'),namespace)
        module.forward=MethodType(namespace['forward'],module)
        changed.append(dict(module=name,released_reference='attn_scores',after='original F.softmax'))
    if not changed:raise ValueError('Require original scaled dot-product attention modules')
    model.attention_reference_lifetime_observations_=changed
    return model


def build_attention_lifetime_memory_model(runtime,parameters):
    from .fm_crossad_author_runtime import build_model
    from .fm_crossad_memory_v2 import install_activation_checkpointing
    from .fm_crossad_reference_lifetime_v1 import install_residual_lifetime_policy
    model=build_model(runtime,parameters)
    install_residual_lifetime_policy(model)
    install_attention_lifetime_policy(model)
    return install_activation_checkpointing(model)
