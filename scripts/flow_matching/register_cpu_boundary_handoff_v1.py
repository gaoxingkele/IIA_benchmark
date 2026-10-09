"""Pin four unchanged queues and live predecessor identities for safe handoff."""
import json
from pathlib import Path
import psutil
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def main():
    path = ROOT / 'configs/runtime/fm_cpu_boundary_handoff.v1.json'
    if path.exists():
        raise FileExistsError('Preserve registered boundary plan')
    specifications = [
        ('strict_cpu', 'configs/experiments/fm_strict_baseline_queue.v1.json',
         'scripts/flow_matching/run_tsad_queue.py', 46056, 'tsad'),
        ('entropic_cpu', 'configs/experiments/fm_tsad_entropic_repair_queue.v1.json',
         'scripts/flow_matching/run_tsad_queue.py', 26368, 'tsad'),
        ('pi_cpu', 'configs/experiments/fm_pi_transformer_author_queue.v1.json',
         'scripts/flow_matching/run_pi_transformer_author_queue.py', 22516, 'pi'),
        ('moment_cpu', 'configs/experiments/fm_moment_pretrained_queue.v1.json',
         'scripts/flow_matching/run_moment_pretrained_queue.py', 35944, 'tsad'),
    ]
    old = json.loads((ROOT / 'configs/runtime/fm_light_controllers.v1.json').read_text(encoding='utf-8'))
    receipts = {r['path']: r for r in old['source_receipts']}
    for source in ['scripts/flow_matching/run_light_controller.py',
                   'scripts/flow_matching/transition_light_controllers.py',
                   'scripts/flow_matching/transition_light_controllers_v2.py',
                   'scripts/flow_matching/transition_cpu_controllers_at_boundary_v1.py',
                   'scripts/flow_matching/register_cpu_boundary_handoff_v1.py',
                   'tests/test_fm_cpu_boundary_handoff_v1.py']:
        receipts[source] = {'path': source, 'sha256': digest(ROOT / source)}
    bindings = []
    for name, queue_path, source_path, pid, kind in specifications:
        queue = json.loads((ROOT / queue_path).read_text(encoding='utf-8'))
        parent = psutil.Process(pid); command = parent.cmdline()
        if '--queue' not in command or not any(Path(a).resolve() == (ROOT / source_path).resolve()
            for a in command if not a.startswith('-')):
            raise ValueError('Live predecessor is not the registered source')
        if Path(command[command.index('--queue') + 1]).resolve() != (ROOT / queue_path).resolve():
            raise ValueError('Live predecessor queue identity differs')
        if kind == 'pi':
            status = queue['settings']['output_root'] + '/status.json'
        elif name == 'moment_cpu':
            status = queue['state_root'] + '/moment_pretrained_status.json'
        else:
            status = queue['state_root'] + '/' + queue['lane'] + '_status.json'
        state = json.loads((ROOT / status).read_text(encoding='utf-8'))
        if state.get('pid') != pid:
            raise ValueError('Status belongs to a different live predecessor')
        bindings.append({'name': name, 'source_path': source_path, 'source_functions': ['main'],
            'queue_path': queue_path, 'queue_sha256': digest(ROOT / queue_path),
            'status_path': status, 'queue_argument': '--queue',
            'old_worker_token': Path(source_path).name, 'resource_scheduler_binding': False,
            'completion_kind': kind, 'registered_jobs': len(queue['jobs']),
            'transition_root': 'experiments/runs/fm_cpu_boundary_handoff_v1/' + name,
            'predecessor': {'pid': pid, 'create_time': parent.create_time(), 'command': command,
                            'observed_private_bytes': parent.memory_info().private}})
        receipts[source_path] = {'path': source_path, 'sha256': digest(ROOT / source_path)}
        receipts[queue_path] = {'path': queue_path, 'sha256': digest(ROOT / queue_path)}
    config = {'schema_version': 1, 'controllers': bindings, 'source_receipts': list(receipts.values()),
        'boundary': 'Only exact observed scheduler processes may be replaced after their model children exit and full output verifies. Same queues, full data, jobs, model workers, trained seeds, budgets and result verifiers. Observation timeouts never imply completion.'}
    for binding in bindings:
        load_controller(ROOT, config, binding['name'])
    path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'runtime': path.relative_to(ROOT).as_posix(),
        'registered_jobs': {b['name']: b['registered_jobs'] for b in bindings},
        'idle_parent_private_bytes_to_release': sum(b['predecessor']['observed_private_bytes'] for b in bindings)}))


if __name__ == '__main__':
    main()
