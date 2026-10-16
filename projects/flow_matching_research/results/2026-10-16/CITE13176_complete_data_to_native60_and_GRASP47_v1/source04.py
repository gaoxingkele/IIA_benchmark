"""Isolated metadata-only resource API repair; retain the full frozen V1 worker."""
import ast
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import prepare_CITE13176_GEO_raw_backed_QC_v1 as previous

ROOT,read,sha=previous.ROOT,previous.read,previous.sha
MODULE='scripts.flow_matching.prepare_CITE13176_GEO_raw_backed_QC_v2'
CONFIG='configs/data/fm_CITE13176_GEO_raw_backed_QC_pca5.v2.json'
TEST='tests/test_fm_CITE13176_GEO_raw_backed_QC_v2.py'


def registration():
    import psutil
    old=read(ROOT/previous.CONFIG);previous.verify(old)
    base=ROOT/old['state_root'];handle=read(base/'launch.json')
    try:
        p=psutil.Process(handle['pid'])
        if p.create_time()==handle['create_time'] and p.is_running():raise ValueError('Original controller remains live;do not replace it')
    except psutil.NoSuchProcess:pass
    if (Path(old['output_directory']).exists() or (base/'worker.stdout.log').exists()
            or (base/'worker.stderr.log').exists() or (base/'guard_receipt.json').exists()):
        raise ValueError('This recovery is only for proven pre-numerical controller failure')
    failure=base/'controller.stderr.log'
    if "KeyError: 'enough'" not in failure.read_text(encoding='utf-8'):
        raise ValueError('Exact preserved controller API-name failure required')
    pins={p['path'] for p in old['source_receipts']}
    pins.update([previous.CONFIG,MODULE.replace('.','/')+'.py',TEST,str(failure),str(base/'launch.json')])
    return dict(old,output_directory='F:/aicoding/IIA_Data/experiments/CITE13176_GEO_raw_backed_QC_pca5_v2',
        state_root='experiments/runs/fm_CITE13176_GEO_raw_backed_QC_pca5_v2',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        metadata_recovery=dict(previous_configuration=previous.CONFIG,previous_failed_handle=handle,
            old_numerical_worker_started=False,old_failure_preserved=True,
            change="private controller AST substitutes resource API key enough with actual has_headroom;full worker bytecode unchanged"))


def verify(config):
    if config!=registration():raise ValueError('Exact isolated pre-numerical metadata repair and complete original data/guard contract required')


def namespace():
    ns=dict(vars(previous),MODULE=MODULE,CONFIG=CONFIG,TEST=TEST,registration=registration,verify=verify)
    ns['worker']=FunctionType(previous.worker.__code__,ns,previous.worker.__name__,previous.worker.__defaults__)
    return ns


def patched_main_tree():
    tree=ast.parse(Path(previous.__file__).read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    changes=[]
    class ResourceAPIName(ast.NodeTransformer):
        def visit_Subscript(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='api' and isinstance(node.slice,ast.Constant) and node.slice.value=='enough':
                changes.append('has_headroom');node.slice=ast.copy_location(ast.Constant('has_headroom'),node.slice)
            return node
    result=ResourceAPIName().visit(fn)
    if changes!=['has_headroom']:raise ValueError('Exactly one controller-only resource API-name correction required')
    return ast.fix_missing_locations(ast.Module(body=[result],type_ignores=[]))


def main():
    ns=namespace();exec(compile(patched_main_tree(),str(previous.__file__),'exec'),ns)
    return ns['main']()


if __name__=='__main__':main()
