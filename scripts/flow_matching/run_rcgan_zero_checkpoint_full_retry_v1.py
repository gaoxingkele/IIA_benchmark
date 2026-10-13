"""Run the complete original seed1107 budget with stronger40GiB admission."""
import argparse
import ast
import copy
import inspect
import sys

from scripts.flow_matching import run_rcgan_native_original_queue_v2 as original
from scripts.flow_matching.gan_zero_checkpoint_retry_v1 import (
    verify_zero_checkpoint_incident, stronger_slot_ast, stronger_admission)
from scripts.flow_matching.rcgan_seed1105_resume_binding_v1 import cached_resource_api


def verify(config):
    for pin in config['source_receipts']:
        if original.sha(original.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Immutable full500 zero-checkpoint retry registration changed')
    old = original.read(original.ROOT / config['original_queue'])
    queue = original.read(original.ROOT / config['queue'])
    expected = copy.deepcopy(old)
    expected['state_root'] = config['state_root']
    selected = next(job for job in expected['jobs'] if job['id'] == config['zero_checkpoint_retry']['canonical_job']['id'])
    if selected != config['zero_checkpoint_retry']['canonical_job']:
        raise ValueError('Exact original canonical seed and full data are required')
    selected['output_directory'] = config['zero_checkpoint_retry']['execution_output_directory']
    if expected != queue:
        raise ValueError('Only selected output and controller metadata may change in all-five registry')
    original.verify_registration(queue)
    verify_zero_checkpoint_incident(config['zero_checkpoint_retry'])
    stronger_admission(config, queue['settings'])
    return queue


def main_ast():
    tree = ast.parse(inspect.getsource(original.main))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'job'
        and ast.dump(node.iter) == ast.dump(ast.parse("queue['jobs']", mode='eval').body)]
    if len(loops) != 1:
        raise ValueError('Exactly one original complete-job dispatch loop required')
    loops[0].iter = ast.parse("[job for job in queue['jobs'] if job['id'] == _selected_job_id]", mode='eval').body
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.run_cfm_ts_low_memory_queue_v2':
            if len(node.names) != 1 or node.names[0].name != 'resource_api':
                raise ValueError('Only host resource telemetry may be rebound')
            node.module = 'scripts.flow_matching.rcgan_seed1105_resume_binding_v1'
            node.names[0].name = 'cached_resource_api'
            node.names[0].asname = 'resource_api'
    return stronger_slot_ast(ast.fix_missing_locations(tree), '_retry_config', "queue['settings']")


def bind(config):
    queue = verify(config)
    namespace = dict(original.__dict__, _retry_config=config,
        _selected_job_id=config['zero_checkpoint_retry']['canonical_job']['id'],
        _stronger_admission=stronger_admission)
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    return namespace['main'], queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = original.read(original.ROOT / args.config)
    function, queue = bind(config)
    if args.probe_only:
        print(dict(required_original_jobs=len(queue['jobs']), original_epochs_each=500,
            selected_same_seed_full_retry=config['zero_checkpoint_retry']['canonical_job']['id'],
            credited_prior_epochs=0, new_independent_seed_slots=0,
            native_training_executed=False, all_original_experiments_complete=False))
        return
    sys.argv = ['original-full500-native-controller', '--queue', config['queue']]
    function()


if __name__ == '__main__':
    main()
