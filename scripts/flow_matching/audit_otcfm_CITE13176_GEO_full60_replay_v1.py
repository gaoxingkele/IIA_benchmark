"""All60 full native CITE checkpoint replays via the unchanged150-case worker."""
import argparse
import ast
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import audit_otcfm_EB_CITE_full150_replay_v1 as base
from scripts.flow_matching import otcfm_EB_CITE_result_integrity_v1 as integrity
from scripts.flow_matching import run_otcfm_CITE13176_GEO_native60_v1 as native

ROOT,read,sha,exclusive=base.ROOT,base.read,base.sha,base.exclusive
MODULE='scripts.flow_matching.audit_otcfm_CITE13176_GEO_full60_replay_v1'
CONFIG='configs/reproducibility/fm_otcfm_CITE13176_GEO_full60_checkpoint_replay.v1.json'
TEST='tests/test_fm_otcfm_CITE13176_GEO_full60_replay_v1.py'


def validate_record(q,j,r,g):
    if j['dataset']!=native.DATASET or j['data']['cells']!=29394 or j['data']['full_raw_RNA_features']!=58676:
        raise ValueError('Entire correct raw-backed GEO CITE cohort required')
    integrity.validate_record(q,j,r,g)


def inspect(q,j):
    ns=dict(vars(integrity),execution=native,validate_record=validate_record)
    return FunctionType(integrity.inspect.__code__,ns)(q,j)


def registration():
    q=read(ROOT/native.QUEUE);native.verify(q);old=read(ROOT/base.CONFIG)
    paths={r['path'] for r in old['source_receipts']}
    paths.update([native.QUEUE,native.PLAN,MODULE.replace('.','/')+'.py',TEST,base.CONFIG])
    return dict(old,original_queue=native.QUEUE,original_queue_sha256=sha(ROOT/native.QUEUE),
        required_full_native_cases=60,required_case_ids=[j['id'] for j in q['jobs']],python=q['python'],settings=q['settings'],
        state_root='experiments/runs/fm_otcfm_CITE13176_GEO_full60_checkpoint_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_CITE13176_GEO_full60_checkpoint_replay_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],GEO_CITE_complete_metric_count=36,
        preserved_prior_singlecell_cases=360,original_scope=q['original_scope'],
        boundary='All60new raw-backed GEO CITE full models:all29394cells,complete36source metrics,two full interior distributions. '
            'Unchanged original150worker bytecode with privately bound source/inspector and exact native2e-6tolerance. '
            'All source checkpoints,optimizer moments,total/selected counters,full cost matrices and OPTIMAL/perturbation bounds required. '
            'No extra fitting,subsampling,retry,prior360replacement,historicalKaggle/fullTAB certification.')


def verify(c):
    if c!=registration():raise ValueError('Exact all60full native/data/guard contracts required')
    return read(ROOT/c['original_queue'])


def namespace():
    return dict(vars(base),MODULE=MODULE,CONFIG=CONFIG,native=native,inspect=inspect,verify=verify)


def worker(c,q,j):return FunctionType(base.worker.__code__,namespace())(c,q,j)


def controller(c,q):
    node=next(n for n in ast.parse(Path(base.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class Scope(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value) is int and n.value==150:return ast.copy_location(ast.Constant(60),n)
            if isinstance(n.value,str):return ast.copy_location(ast.Constant(n.value.replace('150','60')),n)
            return n
    ns=namespace();exec(compile(ast.fix_missing_locations(ast.Module(body=[Scope().visit(node)],type_ignores=[])),str(base.__file__),'exec'),ns)
    return ns['controller'](c,q)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job-id');a=parser.parse_args()
    c=read(ROOT/CONFIG);q=verify(c)
    if a.job_id:worker(c,q,next(j for j in q['jobs'] if j['id']==a.job_id))
    else:controller(c,q)


if __name__=='__main__':main()
