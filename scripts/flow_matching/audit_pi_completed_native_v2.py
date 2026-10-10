"""Keep old Pi audit immutable; verify current full budgets and saved scores."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

from scripts.flow_matching import audit_pi_transformer_author_execution as original
from scripts.flow_matching.giflow_native_protocol import ROOT, sha


def readonly_audit(source):
    tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    boundaries=[i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(x,ast.Name) and x.id=='report' for x in n.targets)]
    if len(boundaries)!=1:
        raise ValueError('Original Pi report boundary changed')
    end=boundaries[0]+1
    if not isinstance(function.body[end],ast.Assign) or not any(
            isinstance(x,ast.Name) and x.id=='target' for x in function.body[end].targets):
        raise ValueError('Original Pi fixed-path publication boundary changed')
    function.body=function.body[:end]+[ast.Return(ast.Name('report',ast.Load()))]
    return ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))


def full_budget(job,result,epochs):
    args=job['author_config'];window=args['data']['win_size'];batch=args['data']['batch_size']
    train_points=job['data_audit']['expected_shapes']['train'][0]
    windows=train_points-window+1;steps=math.ceil(windows/batch)
    if result['native_train_points']!=train_points or result['training_windows']!=windows or result['steps_per_epoch']!=steps:
        raise ValueError('Pi full training sample or batch coverage differs')
    if result['epochs']!=epochs or not epochs or len(epochs)>args['training']['n_epochs']:
        raise ValueError('Pi persisted original epoch budget differs')
    if [e['epoch'] for e in epochs]!=list(range(1,len(epochs)+1)) or any(e['steps']!=steps for e in epochs):
        raise ValueError('Pi complete epoch step budget differs')
    if len(epochs)<args['training']['n_epochs'] and not epochs[-1]['early_stop']:
        raise ValueError('Pi short training lacks original early-stop evidence')
    return dict(training_windows=windows,steps_per_epoch=steps,epochs_completed=len(epochs),
                optimizer_minibatches=len(epochs)*steps,maximum_original_epochs=args['training']['n_epochs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text())
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen Pi audit source changed')
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Keep previous Pi audit snapshots immutable')
    queue=json.loads((ROOT/config['queue']).read_text());jobs={j['id']:j for j in queue['jobs']}
    # All arrays are needed by the existing complete calibration audit. Bound
    # the registered small audit rather than partially loading its score set.
    for job in jobs.values():
        output=ROOT/job['output_directory']
        if (output/'result.json').exists():
            with zipfile.ZipFile(output/'checkpoints/author_scores.npz') as archive:
                if sum(x.file_size for x in archive.infolist())>config['maximum_uncompressed_score_bytes_per_job']:
                    raise ValueError('Complete Pi array audit requires separate bounded admission')
    namespace=dict(vars(original))
    exec(compile(readonly_audit(Path(original.__file__).read_text(encoding='utf-8')),original.__file__,'exec'),namespace)
    report=namespace['audit']()
    budgets=[]
    for record in report['completed_records']:
        job=jobs[record['id']];output=ROOT/job['output_directory']
        result=json.loads((output/'result.json').read_text())
        epochs=json.loads((output/'checkpoints/epochs.json').read_text())
        budgets.append(dict(id=job['id'],seed=job['seed'],**full_budget(job,result,epochs),
            result_sha256=sha(output/'result.json'),independent_saved_score_metric_recalculation_passed=True,
            independent_trained_model_reinference_passed=False))
    report.update(full_budget_audits=budgets,source_receipts=config['source_receipts']+
        [dict(path=args.config,sha256=sha(ROOT/args.config))],audit_boundary='All full native budgets and complete saved-score calibration/PA/pointwise metrics verified. Trained model re-inference remains incomplete; neither branch is strict TAB or certified journal equivalence.',all_original_experiments_complete=False)
    target.mkdir(parents=True)
    (target/'execution_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed_full_jobs=len(budgets),counts=report['counts'],independent_saved_score_metrics_passed=len(budgets),independent_model_reinference_passed=False)))


if __name__=='__main__':main()
