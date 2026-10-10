"""Exact full LS4 attempts with separately disclosed official Cauchy correction."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from scripts.flow_matching.giflow_native_protocol import ROOT,sha
from scripts.flow_matching.ls4_cauchy_fallback_repair_v1 import install

WORKERS={'fred_nn5':'scripts.flow_matching.run_ls4_original_v1',
         'solar_temperature':'scripts.flow_matching.run_ls4_monash_extended_v2'}


def worker(queue):
    return importlib.import_module(WORKERS[queue['canonical_family']])


def verify(queue):
    module=worker(queue);module.verify(queue)
    path=ROOT/queue['original_canonical_queue']
    if sha(path)!=queue['original_canonical_queue_sha256']:
        raise ValueError('LS4 original canonical queue changed')
    original=json.loads(path.read_text(encoding='utf-8'))
    if len(original['jobs'])!=len(queue['jobs']) or original['settings']!=queue['settings']:
        raise ValueError('LS4 original full scope or memory gates changed')
    fields=['dataset','protocol','seed','samples','length','train_samples','test_samples',
            'optim','model','sigma','source_config','budget']
    for a,b in zip(original['jobs'],queue['jobs']):
        old=json.loads((ROOT/a['model_config']).read_text(encoding='utf-8'))
        new=json.loads((ROOT/b['model_config']).read_text(encoding='utf-8'))
        if a['id']!=b['id'] or any(old[f]!=new[f] for f in fields):
            raise ValueError('LS4 correction changed algorithms, seeds, data or full budgets')
        if b['original_output_directory']!=a['output_directory'] or new['author_equivalence_certified']:
            raise ValueError('LS4 corrected attempt original binding or fidelity changed')


def verify_result(queue,job):
    result=worker(queue).verify_result(queue,job)
    correction=result.get('backend_correction',{})
    expected=json.loads((ROOT/queue['cauchy_correction_config']).read_text())
    if not correction.get('algorithmic_correction') or correction.get('official_source_sha256')!=expected['official_cauchy_source_sha256']:
        raise ValueError('LS4 corrected result lacks actual corrected backend receipt')
    if result.get('original_canonical_queue_sha256')!=queue['original_canonical_queue_sha256']:
        raise ValueError('LS4 corrected result lost canonical seed binding')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True)
    args,_=parser.parse_known_args();queue=json.loads((ROOT/args.queue).read_text());verify(queue)
    module=worker(queue);original_import=module.source_imports;original_write=module.write;receipt={}
    correction=json.loads((ROOT/queue['cauchy_correction_config']).read_text())
    def source_imports(target):
        value=original_import(target)
        s4=importlib.import_module('models.s4')
        receipt.update(install(s4,correction,ROOT))
        return value
    def write(path,value):
        value=dict(value,backend_correction=dict(receipt),
            original_canonical_queue=queue['original_canonical_queue'],
            original_canonical_queue_sha256=queue['original_canonical_queue_sha256'],
            correction_boundary='Official conjugate Cauchy fallback replaces defective source naive function; raw source unchanged; full model/data/seeds/budgets unchanged; historical author environment equivalence not certified.')
        if 'source_models_and_evaluators_unchanged' in value:
            value.update(source_models_and_evaluators_unchanged=False,downloaded_source_files_unchanged=True)
        return original_write(path,value)
    module.source_imports=source_imports;module.write=write
    module.main()


if __name__=='__main__':main()
