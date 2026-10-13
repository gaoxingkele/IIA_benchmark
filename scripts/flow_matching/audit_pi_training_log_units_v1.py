"""Audit Pi log timing units without changing a live/frozen scientific worker.

Only consecutive logging boundaries within one full epoch measure100 updates.
Epoch starts can include validation/cache work, so exclude those intervals.
This is monitoring evidence, never a benchmark or a completed training result.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_counter_source(source):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AnomalyDetection']
    if len(classes) != 1:
        raise ValueError('Expected one pinned AnomalyDetection class')
    functions = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'train']
    if len(functions) != 1:
        raise ValueError('Expected one full author training function')
    fn = functions[0]
    stores = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'iter_count' and isinstance(n.ctx, ast.Store)]
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'iter_count' for t in n.targets)]
    if len(stores) != 2 or len(assignments) != 2 or any(not isinstance(n.value, ast.Constant) or n.value.value != 1 for n in assignments):
        raise ValueError('Only the exact unchanged constant-one timing counter is supported')
    match = ast.dump(ast.parse('(i + 1) % 100 == 0', mode='eval').body)
    branches = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.dump(n.test) == match]
    if len(branches) != 1:
        raise ValueError('Exactly the original100-update logging boundary is required')
    statements = branches[0].body
    expected_speed = ast.dump(ast.parse('(time.time() - time_now) / iter_count', mode='eval').body)
    speeds = [n for n in statements if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'speed' for t in n.targets)]
    if len(speeds) != 1 or ast.dump(speeds[0].value) != expected_speed:
        raise ValueError('Original logged speed expression changed')
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple) and isinstance(n.target.elts[0], ast.Name)
                and n.target.elts[0].id == 'i')
    position = loop.body.index(branches[0])
    steps = [i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
             and n.value.func.attr == 'step']
    if len(steps) != 1 or steps[0] <= position:
        raise ValueError('Logging must occur before the current optimizer update')
    return dict(logging_interval_updates=100, denominator_constant=1,
                counter_never_incremented=True, logging_before_current_optimizer_step=True)


def timing_summary(stderr, stdout, steps_per_epoch, required_epochs):
    if steps_per_epoch < 200 or required_epochs < 1:
        raise ValueError('At least two interior logging boundaries in a full epoch required')
    matches = re.findall(r'speed:\s*([0-9.]+)s/iter; left time:\s*([0-9.]+)s', stderr)
    if len(matches) > (steps_per_epoch // 100) * required_epochs:
        raise ValueError('Duplicate or extra logged epochs cannot pass the frozen full budget')
    rows = []
    boundaries = steps_per_epoch // 100
    for ordinal, (duration, claimed) in enumerate(matches):
        epoch, block = divmod(ordinal, boundaries)
        completed_before_current_update = epoch * steps_per_epoch + (block + 1) * 100 - 1
        remaining = required_epochs * steps_per_epoch - completed_before_current_update
        duration, claimed = float(duration), float(claimed)
        # The first block in each epoch spans cache/validation/tail work as well.
        valid = block > 0
        rows.append(dict(epoch_one_based=epoch + 1, logging_block_one_based=block + 1,
                         original_logged_seconds_per_iter=duration,
                         original_logged_left_seconds=claimed,
                         interval_is100_optimizer_updates_without_epoch_boundary=valid,
                         corrected_seconds_per_update=duration / 100 if valid else None,
                         corrected_remaining_optimizer_only_seconds=(duration / 100) * remaining if valid else None,
                         remaining_registered_optimizer_updates=remaining,
                         completed_optimizer_updates_lower_bound=completed_before_current_update,
                         exclusion_reason=None if valid else 'Epoch-start interval includes validation, final epoch tail or initial preparation.'))
    finished_epochs = [int(value) for value in re.findall(r'Epoch:\s*(\d+), Steps:\s*' + str(steps_per_epoch) + r'\s*\|', stdout)]
    if finished_epochs != list(range(1, len(finished_epochs) + 1)) or len(finished_epochs) > required_epochs:
        raise ValueError('Epoch completion log must preserve the full ordered original budget')
    return dict(required_epochs=required_epochs, steps_per_epoch=steps_per_epoch,
                required_optimizer_updates=required_epochs * steps_per_epoch,
                observed_logging_boundaries=len(rows), completed_epoch_logs=finished_epochs,
                completed_optimizer_updates_lower_bound=max([len(finished_epochs) * steps_per_epoch]
                                                           + [r['completed_optimizer_updates_lower_bound'] for r in rows]),
                last_observed_logging_boundary=rows[-1] if rows else None,
                latest_valid_interior_interval=next((r for r in reversed(rows) if r['interval_is100_optimizer_updates_without_epoch_boundary']), None),
                interval_records=rows, validation_and_scoring_time_estimated=False,
                estimated_total_job_finish_time_certified=False, full_training_completion_certified=False)


def audit(config):
    import psutil
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Pinned original source/log snapshot changed: ' + pin['path'])
    expected = config['scientific_worker']
    process = psutil.Process(expected['pid'])
    if abs(process.create_time() - expected['create_time']) >= .001 or process.cmdline() != expected['command']:
        raise ValueError('The actual original scientific handle must remain identical')
    source = (ROOT / config['author_training_source']).read_text(encoding='utf-8')
    source_proof = verify_counter_source(source)
    proof = timing_summary((ROOT / config['frozen_stderr']).read_text(encoding='utf-8'),
                           (ROOT / config['frozen_stdout']).read_text(encoding='utf-8'),
                           config['steps_per_epoch'], config['required_epochs'])
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_id=config['canonical_id'],
                execution_id=config['execution_id'], original_scope=config['original_scope'],
                actual_worker=dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(),
                                   cpu_seconds=sum(process.cpu_times()[:2])), source_timing_proof=source_proof,
                **proof, training_source_modified=False, training_budget_changed=False,
                worker_stopped_or_restarted=False, additional_optimizer_updates=0,
                benchmark_performance_result=False, author_equivalence_certified=False,
                all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    report = audit(config)
    with (ROOT / config['output_report']).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: report[k] for k in ['completed_optimizer_updates_lower_bound', 'latest_valid_interior_interval']}))


if __name__ == '__main__':
    main()
