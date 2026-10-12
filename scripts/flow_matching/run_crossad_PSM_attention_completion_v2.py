"""Use the same complete PSM native/post gates with a new worker version."""
import ast
import inspect
from scripts.flow_matching import run_crossad_PSM_lifetime_completion_v1 as previous


def main_ast():
    tree=ast.parse(inspect.getsource(previous.main));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='scripts.flow_matching.replay_crossad_residual_lifetime_v4':
            node.value='scripts.flow_matching.replay_crossad_attention_lifetime_v5';changed+=1
    if changed!=1:raise ValueError('Keep every original native and post-review gate unchanged')
    return tree


def main():
    namespace=dict(previous.__dict__)
    exec(compile(main_ast(),__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
