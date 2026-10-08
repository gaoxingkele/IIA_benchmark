"""Freeze orchestration-only bindings without changing experiment identities."""
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import FUNCTIONS, ROOT, digest


def main():
    controllers = []
    definitions = [
        ('crossad', 'run_crossad_case_gated_queue.py', ['native_proof','main'], 'fm_crossad_complete_queue.v7.json', 'fm_crossad_author_v2', '--queue', 'run_crossad_case_gated_queue_v2.py', True),
        ('preflight', 'preflight_crossad_complete.py', ['main'], 'fm_crossad_complete_queue.v4.json', 'fm_crossad_complete_preflight_v4', '--queue', 'scripts.flow_matching.preflight_crossad_complete_v2', True),
        ('recovery', 'run_resource_recovery.py', ['main'], 'fm_resource_recovery_queue.v3.json', 'fm_resource_recovery_20261009_v2', '--queue', 'scripts.flow_matching.run_resource_recovery_v2', True),
        ('range_main', 'run_tab_ranges_sparse_queue.py', ['main'], 'fm_tab_range_execution.sparse.v2.json', 'fm_tsad_tab_range_sparse_v2', '--config', 'scripts.flow_matching.run_tab_ranges_sparse_queue', False),
        ('range_reflow', 'run_tab_ranges_sparse_queue.py', ['main'], 'fm_reflow_tab_range.sparse.v2.json', 'fm_reflow_tab_range_sparse_v2', '--config', 'scripts.flow_matching.run_tab_ranges_sparse_queue', False),
        ('range_industrial', 'run_tab_ranges_sparse_queue.py', ['main'], 'fm_industrial_tab_range.sparse.v2.json', 'fm_industrial_tab_range_sparse_v2', '--config', 'scripts.flow_matching.run_tab_ranges_sparse_queue', False),
        ('range_moment', 'run_tab_ranges_sparse_queue.py', ['main'], 'fm_moment_pretrained_tab_range.sparse.v2.json', 'fm_moment_pretrained_tab_range_sparse_v2', '--config', 'scripts.flow_matching.run_tab_ranges_sparse_queue', False)]
    for name, source, functions, queue, output, argument, token, scheduler in definitions:
        path = 'configs/experiments/'+queue
        controllers.append(dict(name=name, source_path='scripts/flow_matching/'+source, source_functions=functions,
            queue_path=path, queue_sha256=digest(ROOT/path), status_path='experiments/runs/'+output+'/status.json',
            queue_argument=argument, old_worker_token=token, resource_scheduler_binding=scheduler))
    sources = set(FUNCTIONS) | {c['source_path'] for c in controllers} | {
        'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/register_light_controllers.py',
        'scripts/flow_matching/transition_light_controllers.py'}
    config = {'schema_version':1, 'controllers':controllers,
              'source_receipts':[{'path':p, 'sha256':digest(ROOT/p)} for p in sorted(sources)],
              'transition_archive':'experiments/runs/fm_light_controller_transition_20261009',
              'boundary':'Exact pinned original orchestration definitions compiled without unused preprocessing imports. Original model/evaluator child programs, queues, job identities, artifacts, budgets and resource gates remain unchanged.'}
    path = ROOT/'configs/runtime/fm_light_controllers.v1.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==config, 'Runtime already frozen'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'controllers':len(controllers),'source_files':len(sources),'config_sha256':digest(path)}))


if __name__=='__main__':
    main()
