"""Request the existing finite data window only when original headroom is ready."""
import ast
import inspect
from pathlib import Path
from scripts.flow_matching import reserve_CITE_full_preprocessing_window_v1 as original

ROOT, read, sha, exclusive = original.ROOT, original.read, original.sha, original.exclusive
MODULE='scripts.flow_matching.reserve_CITE_headroom_ready_window_v2'
CONFIG='configs/runtime/fm_CITE13176_GEO_headroom_ready_data_window.v2.json'
TEST='tests/test_fm_CITE_headroom_ready_window_v2.py'
PREVIOUS_RECEIPT='experiments/runs/fm_CITE13176_GEO_nonpreemptive_data_window_v1/window_receipt.json'


def registration():
    c=original.registration();r=read(ROOT/PREVIOUS_RECEIPT)
    if (not r['all_metadata_locks_released'] or r['numerical_children_signalled'] or r['original_guards_changed']
            or r['last_state']['state']!='release_no_original_headroom_after_grace'
            or r['last_state']['actual_worker_live'] or not r['last_state']['actual_producer_live']):
        raise ValueError('Prior finite metadata window must have released without starting or changing a numeric worker')
    import psutil
    old=read(ROOT/original.CONFIG);p=old['preparation_handle']
    try:
        actor=psutil.Process(r['last_state']['pid'])
        if actor.is_running() and actor.create_time()==r['last_state']['create_time']:
            raise ValueError('Previous scheduling window is still live;never duplicate owners')
    except psutil.NoSuchProcess:pass
    if c['preparation_handle']!=p:raise ValueError('Continue only the same existing full preprocessing controller')
    c.update(state_root='experiments/runs/fm_CITE13176_GEO_headroom_ready_data_window_v2',
        require_original_headroom_before_reserving_any_family_lock=True,
        preserved_predecessor_receipt=PREVIOUS_RECEIPT,preserved_predecessor_receipt_sha256=sha(ROOT/PREVIOUS_RECEIPT),
        original_configuration=original.CONFIG,original_configuration_sha256=sha(ROOT/original.CONFIG),
        numerical_preprocessing_attempt_repeated=False)
    paths=[original.CONFIG,PREVIOUS_RECEIPT,MODULE.replace('.','/')+'.py',TEST]
    c['source_receipts']+=[dict(path=p,sha256=sha(ROOT/p)) for p in paths]
    c['boundary']+=' New isolated finite scheduling entry first measures the unchanged original headroom before requesting either lock. Prior window never admitted preprocessing;no numerical attempt is repeated. Original preprocessing controller,all worker bytecode,6x5second admission,900/900second deadlines,120second grace and all guards remain unchanged.'
    return c


def verify(c):
    if c!=registration():raise ValueError('Same live full preprocessing owner and all original finite admission conditions required')


def headroom_gated_reserve(exclusive_lock,paths,api,settings):
    if not api['has_headroom'](api['resource_snapshot'](),settings):return None
    return original.try_reserve(exclusive_lock,paths)


def main_tree():
    tree=ast.parse(inspect.getsource(original.main));count=0
    class Scope(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='try_reserve':
                count+=1;node.func.id='headroom_gated_reserve'
                node.args.extend([ast.Name(id='api',ctx=ast.Load()),
                    ast.Subscript(value=ast.Name(id='c',ctx=ast.Load()),slice=ast.Constant(value='original_preparation_settings'),ctx=ast.Load())])
            return node
        def visit_Constant(self,node):
            if node.value=='waiting_both_families_naturally_idle_no_lock_owned':
                node.value='waiting_original_headroom_and_both_families_idle_no_lock_owned'
            return node
    tree=Scope().visit(tree)
    if count!=1:raise ValueError('Only the original metadata acquisition call may change')
    return ast.fix_missing_locations(tree)


def main():
    namespace=dict(original.__dict__,MODULE=MODULE,CONFIG=CONFIG,TEST=TEST,registration=registration,
        verify=verify,headroom_gated_reserve=headroom_gated_reserve)
    exec(compile(main_tree(),__file__,'exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
