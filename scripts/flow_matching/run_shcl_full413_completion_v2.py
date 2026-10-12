"""Preserve all413 native replay tasks and require shared GPU admission for tests."""
import ast
import inspect
from types import FunctionType

from scripts.flow_matching import run_shcl_full413_completion_v1 as original
from scripts.flow_matching import run_cached_resource_monitor_controller_v2 as handoff
from scripts.flow_matching.combined_readonly_gpu_slot_v1 import try_combined_slot, verify_gpu_mutex


class CombinedHandle:
    def __init__(self, handles): self.handles=handles
    def close(self):
        for handle in reversed(self.handles): handle.close()


def main_ast():
    tree=ast.parse(inspect.getsource(original.main))
    function=tree.body[0]; matches=0
    for index,node in enumerate(function.body):
        if isinstance(node,ast.FunctionDef) and node.name=='ready_slot':
            replacement=ast.parse('''def ready_slot(job):
    lane_api=dict(api,gpu_snapshot=gpu_snapshot)
    while True:
        handles, observation=try_combined_slot(lane_api,config,fixture=job['mode']=='train')
        if handles is not None:
            return CombinedHandle(handles)
        update(dict(observation,shared_slot_owned=False))
        time.sleep(10)
''').body[0]
            function.body[index]=replacement;matches+=1
    if matches!=1:raise ValueError('Only original full replay readiness gains shared GPU admission')
    return ast.fix_missing_locations(tree)


def safe_phase(config,state,descendants):
    old=config['handoff']['previous_owner']
    return (state.get('pid')==old['pid'] and state.get('create_time')==old['create_time']
        and state.get('active_pid') is None and state.get('state')=='waiting_for_full_original_training_and_test_results'
        and state.get('required_count')==413 and state.get('verified_count')==0 and not state.get('failures')
        and state.get('configuration_sha256')==config['handoff']['previous_configuration_sha256']
        and all(handoff.allowed_console_child(p['name'],p['exe']) for p in descendants))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    args=parser.parse_args();native=original.replay.native;config=native.read(native.ROOT/args.config)
    original.verify(config);verify_gpu_mutex(config)
    base=native.ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    stop=FunctionType(handoff.safe_handoff.__code__,dict(handoff.__dict__,safe_phase=safe_phase))
    stop(config,base,lambda values:original.write(base/'idle_handoff_wait.json',values))
    namespace=dict(original.__dict__,try_combined_slot=try_combined_slot,CombinedHandle=CombinedHandle)
    exec(compile(main_ast(),__file__,'exec'),namespace);namespace['main']()


if __name__=='__main__':main()
