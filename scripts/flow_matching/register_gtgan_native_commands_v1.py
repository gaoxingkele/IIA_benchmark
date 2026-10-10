"""Freeze complete released GT-GAN stock recipes without importing or rewriting models."""
import argparse
import ast
import copy
from pathlib import Path

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write


def native_arguments(source,entry,arguments):
    tree=ast.parse((source/'ctfp_tools.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_arguments')
    solvers=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOLVERS' for t in n.targets))
    namespace=dict(argparse=argparse)
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy([solvers,function]),type_ignores=[])),str(source/'ctfp_tools.py'),'exec'),namespace)
    namespace['parser']=namespace['parse_arguments']()
    main=next(n for n in ast.parse((source/entry).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    additions=[n for n in main.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='add_argument'
        and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='parser']
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(additions),type_ignores=[])),str(source/entry),'exec'),namespace)
    return vars(namespace['parser'].parse_args(arguments))


def recipes():
    regular=['--dims','32-64-64-32','--train','--atol','1e-2','--rtol','1e-3','--activation','exp',
        '--max-steps','8500','--last_activation_r','softplus','--last_activation_d','identity',
        '--reconstruction','0.01','--kinetic-energy','0.01','--jacobian-norm2','0.05','--directional-penalty','0.01']
    irregular=['--train','--data','stock','--dims','32-64-64-32','--atol','1e-2','--rtol','1e-3',
        '--first_epoch','10000','--max-steps','2000','--activation','exp','--last_activation_r','softplus',
        '--last_activation_d','identity','--reconstruction','0','--kinetic-energy','0.05','--jacobian-norm2','0.01',
        '--directional-penalty','0.05']
    yield 0,'GTGAN_stocks.py',regular,'Exact original README regular-stock training command'
    for rate in [.3,.5,.7]:
        yield rate,'GTGAN_stocks_irregular.py',irregular+['--missing_value',str(rate)],(
            'Exact original README70percent stock training command' if rate==.7 else
            'Declared30/50percent mask setting of original70percent command; source-supported, historical table-row equivalence unproved')


def main():
    acquisition=read(ROOT/'configs/acquisition/fm_gtgan_author_archive.v1.json')
    source=ROOT/acquisition['asset_root'];cases=[]
    for rate,entry,args,boundary in recipes():
        parsed=native_arguments(source,entry,args)
        assert parsed['first_epoch']==10000 and parsed['batch_size']==128 and parsed['seq_len']==24 and parsed['max_steps_metric']==10
        cases.append(dict(id='gtgan_original_stock_missing_'+str(rate).replace('.','p'),missing_rate=rate,
            source_entry=(Path(acquisition['asset_root'])/entry).as_posix(),source_entry_sha256=sha(source/entry),
            original_seed=7777,author_arguments=args,parsed_original_arguments=parsed,
            embedding_training_steps=10000,joint_training_steps=parsed['max_steps'],
            native_evaluator_repeats=10,boundary=boundary,full_native_training_complete=False))
    config=dict(id='fm_gtgan_native_original_stock_v1',method='GT-GAN',task='irregular_to_regular_time_series_generation',
        source_commit=acquisition['commit'],source_root=acquisition['asset_root'],dataset_config='configs/datasets/fm_gtgan_original_stock.v1.json',
        cases=cases,implementation='Original GTGAN_stocks.py:main / GTGAN_stocks_irregular.py:main',
        reproduction_status='complete_original_source_acquired_native_data_audit_running_full_training_and_evaluation_pending',
        tests=['tests/test_fm_pinned_author_archive_v1.py','tests/test_fm_gtgan_native_stock_data_v1.py','tests/test_fm_gtgan_native_commands_v1.py'],
        citations=acquisition['citations'],historical_environment='data/public_datasets/flow_matching/gtgan_native_source_v1/original/environment.yaml',
        data_audit_config='configs/reproducibility/fm_gtgan_native_stock_data_audit.2026-10-11_v1.json',
        author_equivalence_certified=False,performance_result=False,
        boundary='Supporting adopted original baseline of Spectral Mean FlowTable4. Exact full native regular and70percent recipes frozen;30/50percent flags explicitly extend the same released irregular command. Native source hardcodes seed7777. Data audit with compatible environment is separate from full generator/model training and original TensorFlow evaluator. Original generator uses complete overlapping windows with no heldout generator split; no strict leaderboard result.')
    path=ROOT/'configs/models/fm_gtgan_native_original_stock.v1.json'
    if path.exists():raise FileExistsError(path)
    write(path,config)
    stock=source/'datasets/stock_data.csv'
    data=dict(id='gtgan_original_stock',task='time_series_generation',raw_file=stock.relative_to(ROOT).as_posix(),raw_sha256=sha(stock),
        source_repository=acquisition['repository'],source_commit=acquisition['commit'],raw_storage_preserved=True,
        sequence_length=24,variables=6,preprocessing='Original full raw-row reversal,global per-variable MinMax,all stride-one windows,whole-row removal with seed56789 for30/50/70percent; no replacement raw payload',
        native_loader='data/public_datasets/flow_matching/gtgan_native_source_v1/original/time_dataset.py',
        audit_config='configs/reproducibility/fm_gtgan_native_stock_data_audit.2026-10-11_v1.json',citations=acquisition['citations'])
    path=ROOT/'configs/datasets/fm_gtgan_original_stock.v1.json'
    if path.exists():raise FileExistsError(path)
    write(path,data);print(dict(original_stock_cases=len(cases),embedding_steps_each=10000,original_seed=7777,native_training_complete=False))


if __name__=='__main__':main()
