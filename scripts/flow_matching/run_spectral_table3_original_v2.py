"""Use every original optimizer assignment, including the author's logger scope."""
import argparse
import ast
from pathlib import Path

from scripts.flow_matching import run_spectral_table3_original_v1 as released

ROOT, read, sha, verify = released.ROOT, released.read, released.sha, released.verify


def optimizer_assignments(entry):
    tree = ast.parse(Path(entry).read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    names = {'muon_params', 'adamw_params', 'param_groups', 'optimizer'}
    selected = []

    def visit(statements):
        for node in statements:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in names:
                    selected.append(node)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                visit(node.body)
            # Do not silently extract conditional optimizers or nested functions.

    visit(main.body)
    actual_names = [n.targets[0].id for n in selected]
    if actual_names not in [['optimizer'], ['muon_params', 'adamw_params', 'param_groups', 'optimizer']]:
        raise ValueError('Complete original optimizer assignments required in source order')
    return selected


def original_optimizer(entry, model, args, torch):
    selected = optimizer_assignments(entry)
    namespace = dict(model=model, args=args, torch=torch)
    if selected[0].targets[0].id == 'muon_params':
        from models.spectral_flow.muon import SingleDeviceMuonWithAuxAdam
        namespace['SingleDeviceMuonWithAuxAdam'] = SingleDeviceMuonWithAuxAdam
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(entry), 'exec'), namespace)
    return namespace['optimizer']


# Only the diagnostic extractor changes. Full train/eval entries still execute
# byte-identically through the frozen v1 bootstrap and original generator loops.
released.original_optimizer = original_optimizer
data_audit, preflight, run_job = released.data_audit, released.preflight, released.run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id')
    parser.add_argument('--audit-data')
    parser.add_argument('--preflight-output')
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    verify(queue)
    if args.audit_data:
        data_audit(queue, ROOT / args.audit_data)
    else:
        job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
        if args.preflight_output:
            preflight(queue, job, ROOT / args.preflight_output)
        else:
            run_job(queue, job)


if __name__ == '__main__':
    main()
