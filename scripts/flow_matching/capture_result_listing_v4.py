"""Capture current verified metrics without rewriting legacy reports or live jobs."""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def readonly_function(source):
    """Keep the frozen snapshot calculation; omit its fixed-path publication tail."""
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'snapshot')
    indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'report' for t in node.targets)]
    if len(indices) != 1:
        raise ValueError('Snapshot report boundary changed')
    end = indices[0]+1
    tail = function.body[end:]
    if not tail or not isinstance(tail[0], ast.Assign) or not any(
            isinstance(t, ast.Name) and t.id == 'target' for t in tail[0].targets):
        raise ValueError('Fixed-path publication boundary changed')
    function.body = function.body[:end] + [ast.Return(value=ast.Name(id='report', ctx=ast.Load()))]
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def canonical_grasp(queue_path, recovery_paths):
    from scripts.flow_matching import grasp_protocol_runner as worker
    from scripts.flow_matching import summarize_grasp_protocol_v3 as summarizer
    from scripts.flow_matching.native_exact_recovery import verify
    queue = read(ROOT/queue_path)
    worker.verify(queue, ROOT)
    candidates = []
    for path in recovery_paths:
        recovery = read(ROOT/path)
        verify(recovery, ROOT)
        candidates.extend(recovery['jobs'])
    merged = dict(queue, jobs=[])
    resolutions = []
    for job in queue['jobs']:
        selected = job
        if not worker.complete(job, ROOT):
            for candidate in candidates:
                if candidate['recovery_original_job_id'] != job['id']:
                    continue
                normalized = dict(candidate)
                normalized.pop('recovery_original_job_id')
                normalized.update(id=job['id'], output_directory=job['output_directory'])
                if normalized != job:
                    raise ValueError('Native recovery changed canonical experiment')
                if worker.complete(candidate, ROOT):
                    selected = candidate
                    resolutions.append(dict(original_id=job['id'], execution_id=candidate['id']))
                    break
        merged['jobs'].append(selected)
    capture = summarizer.summarize(merged, ROOT)
    capture.update(queue=queue_path, queue_sha256=sha(ROOT/queue_path), exact_recovery_resolutions=resolutions)
    rows = []
    for group in capture['seed_aggregates']:
        recipes = next(j['score_recipes'] for j in queue['jobs']
                       if j['profile'] == group['profile'] and j['dataset'] == group['dataset'])
        for recipe in recipes:
            values = next((m['ten_seed_metrics'] for m in group['metrics'] or [] if m['recipe'] == recipe), {})
            for metric in ('AP', 'ROC', 'Best_F1', 'precision', 'recall', 'f1'):
                rows.append(dict(algorithm='GRASP/'+group['profile'], dataset=group['dataset'],
                    task='anomaly_detection', score_recipe=json.dumps(recipe, sort_keys=True),
                    protocol='paper_test_label_oracle_no_PA' if metric in ('AP','ROC','Best_F1')
                    else 'validation_only_1_percent_no_PA_control', metric=metric,
                    required_repeats=group['required_seeds'],
                    **values.get(metric, dict(n=0, mean=None, std=None, se=None, ci95_low=None, ci95_high=None)),
                    paper_mean=None, author_equivalence_certified=False,
                    captured_utc=capture['captured_utc'], source='grasp_execution_audit.json',
                    comparison='local_full_budget_source_fidelity_not_certified'))
    return capture, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT/args.config)
    for receipt in config['source_receipts']:
        if sha(ROOT/receipt['path']) != receipt['sha256']:
            raise ValueError('Listing source changed: '+receipt['path'])
    target = ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Keep previous captures immutable')
    target.mkdir(parents=True)
    from scripts.flow_matching import summarize_tsad_execution as frozen
    namespace = dict(vars(frozen))
    exec(compile(readonly_function(Path(frozen.__file__).read_text(encoding='utf-8')), frozen.__file__, 'exec'), namespace)
    snapshot = namespace['snapshot'](ROOT)
    write(target/'tsad_source_snapshot.json', snapshot)
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'scripts.flow_matching.export_complete_metrics',
                    '--config', config['export_config']], cwd=ROOT, check=True)
    cfm = config['cfm_output_directory']
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'scripts.flow_matching.summarize_cfm_ts_canonical_v2',
                    '--queue', config['cfm_queue'], '--output', cfm,
                    *[v for path in config['cfm_recovery_queues'] for v in ('--recovery-queue', path)]], cwd=ROOT, check=True)
    grasp, rows = canonical_grasp(config['grasp_queue'], config['grasp_recovery_queues'])
    write(target/'grasp_execution_audit.json', grasp)
    csv_write(target/'grasp_algorithm_dataset_metrics.csv', rows)
    write(target/'capture_receipt.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        source_receipts=config['source_receipts'], tsad_counts=snapshot['new_job_counts'],
        grasp_completed_cohorts=grasp['completed_cohorts'],
        all_original_experiments_complete=False, existing_reports_and_jobs_unchanged=True))
    print(json.dumps(dict(target=config['output_directory'], tsad=snapshot['new_job_counts'],
                         grasp_completed_cohorts=grasp['completed_cohorts'])))


if __name__ == '__main__':
    main()
