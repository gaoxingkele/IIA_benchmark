"""Isolated correction of the bundled CNF trajectory endpoint selection.

The original source is supplied by the caller;no source file is rewritten.
All likelihood,regularization,ODE and optimizer equations remain original.
"""
import ast
import inspect
import textwrap


def step_tree(source_text,correct=True):
    tree=ast.parse(textwrap.dedent(source_text))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CNFLitModule')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    changed=0
    class Endpoint(ast.NodeTransformer):
        def visit_Compare(self,n):
            nonlocal changed
            self.generic_visit(n)
            if (ast.dump(n.left)==ast.dump(ast.parse('self.hparams.leaveout_timepoint',mode='eval').body)
                    and len(n.ops)==1 and isinstance(n.ops[0],ast.Eq)
                    and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Name) and n.comparators[0].id=='t'):
                changed+=1
                if correct:n.comparators[0]=ast.parse('(len(even_ts)-1-t) if self.is_trajectory else t',mode='eval').body
            return n
    fn=Endpoint().visit(fn)
    if changed!=1:raise ValueError('Exactly one frozen original reversed-loop leaveout comparison is required')
    return ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))


def compile_step(source_text,namespace,correct=True):
    local=dict(namespace)
    exec(compile(step_tree(source_text,correct),'<isolated_original_CNF_leaveout>','exec'),local)
    return local['step']


def build_corrected_cnf(source_class,**parameters):
    if source_class.__name__!='CNFLitModule':raise ValueError('Use the frozen original CNF class only')
    source=inspect.getsource(source_class)
    fixed=compile_step(source,source_class.step.__globals__,True)
    private=type('LeaveoutCorrectedSourceCNF',(source_class,),dict(step=fixed,__module__=__name__))
    return private(**parameters)


def endpoint_mapping(timepoints):
    if type(timepoints) is not int or timepoints not in (4,5):raise ValueError('Complete four/five-timepoint trajectories required')
    return [dict(requested_leftout=left,original_omitted_observation_index=timepoints-1-left,
        corrected_omitted_observation_index=left,
        original_objective_includes_requested_leftout=(timepoints-1-left)!=left)
        for left in range(1,timepoints-1)]
