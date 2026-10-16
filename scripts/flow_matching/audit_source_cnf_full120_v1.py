"""Fresh complete CNF checkpoint replay with independent full W1 LP costs."""
import argparse
import ast
import inspect as pyinspect
from pathlib import Path
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import run_source_cnf_full120_v1 as execution
from scripts.flow_matching import audit_otcfm_EB_CITE_full150_replay_v1 as parent
from scripts.flow_matching import otcfm_EB_CITE_result_integrity_v1 as integrity
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.audit_source_cnf_full120_v1'
CONFIG='configs/reproducibility/fm_source_cnf_full120_checkpoint_replay.v1.json'


def registration():
    q=read(ROOT/execution.QUEUE);execution.verify(q)
    c=parent.registration()
    c.update(original_queue=execution.QUEUE,original_queue_sha256=sha(ROOT/execution.QUEUE),
        required_full_native_cases=120,required_case_ids=[j['id'] for j in q['jobs']],
        state_root='experiments/runs/fm_source_cnf_full120_checkpoint_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/source_cnf_full120_checkpoint_replay_v1',
        settings=q['settings'],original_scope=q['original_scope'],source_derived_plain_CNF=True,
        boundary=q['boundary']+' All actual trained parameters and AdamW moments,each registered decay,full source metric schema,all complete interior populations,original2e-6 metric tolerance and independent SciPy-cost exact W1 LPs are required.')
    paths={x['path'] for x in c['source_receipts']}|{execution.QUEUE,MODULE.replace('.','/')+'.py',execution.TEST}
    c['source_receipts']=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)]
    return c


def verify(c):
    if c!=registration():raise ValueError('Frozen full120 CNF checkpoint replay contract required')
    return read(ROOT/c['original_queue'])


def inspect(q,job):
    def validate(queue,condition,result,guard):
        integrity.validate_record(queue,condition,result,guard)
        if (result['original_source_native_execution'] or not result['source_derived_repaired_execution']
                or not result['endpoint_gradient_accumulation'] or not result['one_optimizer_update_per_full_training_batch']
                or result['full_input_preflight_binding']!=execution.prerequisite(q)
                or result['native_framework_lifecycle_binding']!=execution.lifecycle_binding(q)
                or result['source_CNF_parameter_track']!=job['protocol']):
            raise ValueError('Disclosed source CNF repairs and complete preflight binding required')
    f=FunctionType(integrity.inspect.__code__,dict(integrity.inspect.__globals__,
        execution=execution,validate_record=validate))
    return f(q,job)


def replay_tree():
    tree=ast.parse(pyinspect.getsource(parent.worker))
    changed=0
    class Decay(ast.NodeTransformer):
        def visit_Compare(self,node):
            nonlocal changed
            if ast.unparse(node)=='group[\'weight_decay\'] != 1e-05':
                changed+=1
                return ast.copy_location(ast.parse("group['weight_decay'] != job['scientific_protocol']['weight_decay']",mode='eval').body,node)
            return self.generic_visit(node)
    tree=Decay().visit(tree)
    if changed!=1:raise ValueError('Exactly the frozen source optimizer decay assertion may change')
    return ast.fix_missing_locations(tree)


def worker(c,q,job):
    namespace=dict(parent.worker.__globals__,inspect=inspect,native=execution,CONFIG=CONFIG)
    exec(compile(replay_tree(),'<full_CNF_original_all_population_replay>','exec'),namespace)
    namespace['worker'](c,q,job)


def controller(c,q):
    # Retain the complete tested generalized interior-timepoint worker and
    # controller. Only the original literal150 scope is rebound to120.
    tree=ast.parse(pyinspect.getsource(parent.controller))
    count=0
    class Scope(ast.NodeTransformer):
        def visit_Constant(self,node):
            nonlocal count
            if type(node.value) is int and node.value==150:
                count+=1;return ast.copy_location(ast.Constant(120),node)
            return node
    tree=Scope().visit(tree)
    if count!=2:raise ValueError('Exactly the controller count and terminal loop scope may change')
    namespace=dict(parent.controller.__globals__,MODULE=MODULE)
    exec(compile(ast.fix_missing_locations(tree),'<full120_CNF_replay_controller>','exec'),namespace)
    namespace['controller'](c,q)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--job-id');args=parser.parse_args()
    if args.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);q=verify(c)
    if args.job_id:worker(c,q,next(j for j in q['jobs'] if j['id']==args.job_id))
    else:controller(c,q)


if __name__=='__main__':main()
