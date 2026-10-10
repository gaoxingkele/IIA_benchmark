"""Execute both complete original GT-GAN stock loaders without touching raw caches."""
import argparse
import ast
import copy
import importlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,write,capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def original_loader_module(source):
    tree=ast.parse(source)
    names={'normalize','to_tensor','load_data','save_data','TimeDataset_regular','TimeDataset_irregular'}
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('Whole original loader/function scope changed')
    class CPU(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
                if node.args or node.keywords:raise ValueError('Unrecognized native device binding')
                self.count+=1;node.func.attr='to';node.args=[ast.Constant(value='cpu')]
            return node
    mapper=CPU();module=mapper.visit(ast.Module(body=nodes,type_ignores=[]))
    if mapper.count!=2:raise ValueError('Only original two preprocessing CUDA bindings may move to CPU')
    return ast.fix_missing_locations(module)


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Original author material changed: '+item['path'])


def worker(config):
    locked=read(ROOT/config['environment_lock'])
    if sys.version.split()[0]!=locked['python'].split()[0]:raise ValueError('Frozen compatible native Python environment required')
    for package,version in locked['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Frozen compatible native package changed: '+package)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    verify(config)
    source=ROOT/config['source_root'];sys.path.insert(0,str(source))
    cde=importlib.import_module('controldiffeq')
    target=ROOT/config['output_directory'];artifacts=Path(config['artifact_root'])
    if target.exists() or artifacts.exists():raise FileExistsError('Keep every original-loader attempt')
    target.mkdir(parents=True);artifacts.mkdir(parents=True)
    namespace=dict(np=np,torch=torch,os=os,here=artifacts,controldiffeq=cde)
    exec(compile(original_loader_module((source/'time_dataset.py').read_text(encoding='utf-8')),str(source/'time_dataset.py'),'exec'),namespace)
    raw=ROOT/config['raw_stock_csv'];before=sha(raw);data=np.loadtxt(raw,delimiter=',',skiprows=1)[::-1]
    normalized=(data-data.min(0))/(np.ptp(data,axis=0)+1e-7)
    length=24;windows=len(data)-length+1;records=[]
    for missing in config['missing_rates']:
        torch.manual_seed(7777);np.random.seed(7777)
        if missing==0:
            loader=namespace['TimeDataset_regular'](raw,length)
            samples=np.asarray(loader.samples);truth=samples[:,:,:-1]
        else:
            loader=namespace['TimeDataset_irregular'](raw,length,'stock',missing)
            samples=np.asarray(loader.samples);truth=np.asarray(loader.original_sample)
        assert samples.shape==(windows,length,7) and truth.shape==(windows,length,6)
        indices=samples[:,:,-1].astype(np.int64)
        assert np.array_equal(np.sort(indices[:,0]),np.arange(windows))
        assert np.array_equal(indices,indices[:,0,None]+np.arange(length)[None,:])
        assert np.array_equal(truth,normalized[indices])
        removed=torch.randperm(len(data),generator=torch.Generator().manual_seed(56789))[:int(len(data)*missing)].sort().values.numpy()
        row_mask=np.zeros(len(data),dtype=bool);row_mask[removed]=True
        expected_mask=np.broadcast_to(row_mask[indices][...,None],truth.shape)
        assert np.array_equal(np.isnan(samples[:,:,:-1]),expected_mask)
        assert np.array_equal(samples[:,:,:-1][~expected_mask],truth[~expected_mask])
        coeffs=[]
        if missing:
            for coefficient in loader.train_coeffs:
                assert coefficient.shape==(windows,length-1,6) and torch.isfinite(coefficient).all()
                coeffs.append(coefficient.detach().numpy())
            np.savez_compressed(artifacts/('coefficients_'+str(missing)+'.npz'),**{str(i):c for i,c in enumerate(coeffs)})
        output=artifacts/('full_stock_'+str(missing)+'.npz')
        np.savez_compressed(output,samples=samples,truth=truth,raw_row_ids=indices,removed_raw_row_ids=removed)
        record=dict(missing_rate=missing,raw_rows=len(data),variables=6,sequence_length=length,complete_windows=windows,
            removed_raw_rows=len(removed),missing_window_values=int(expected_mask.sum()),
            full_original_truth_and_time_row_identity=True,original_seed56789_whole_row_mask_exact=True,
            all_observed_values_exact=True,full_original_cubic_coefficients_finite=bool(missing),
            spline_coefficient_shapes=[list(c.shape) for c in coeffs],path=str(output),sha256=sha(output),
            native_original_loader_executed=True,performance_result=False)
        records.append(record);write(target/'progress.json',dict(completed_profiles=len(records),required_profiles=4))
    if sha(raw)!=before or torch.cuda.is_initialized():raise ValueError('Raw preservation and CPU-only audit required')
    verify(config)
    report=dict(passed=True,diagnostic_only=True,performance_result=False,records=records,
        original_author_loader_normalization_masking_and_shuffle_preserved=True,
        original_generator_heldout_split_present=False,original_random_seed=7777,original_row_mask_seed=56789,
        cuda_context_initialized=False,raw_csv_unchanged=True,environment_lock=config['environment_lock'],
        environment_equivalence_certified=False,all_original_experiments_complete=False,boundary=config['boundary'])
    write(target/'original_stock_loader_audit.json',report)
    write(target/'validation.json',dict(source_receipts=config['source_receipts'],outputs={p.name:sha(p) for p in target.glob('*.json')}))
    print(dict(full_original_loader_profiles=4,complete_windows_per_profile=windows,raw_rows=len(data),passed=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    config=read(ROOT/args.config);verify(config)
    if args.worker:worker(config);return
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lock=api['exclusive_lock'](base/'controller.lock')
    if lock is None:raise RuntimeError('Native stock audit already has live controller')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[str(ROOT/config['python']),'-X','utf8','-u','-m','scripts.flow_matching.audit_gtgan_native_stock_data_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                    PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,config,api,update)
        proof=ROOT/config['output_directory']/'original_stock_loader_audit.json';passed=code==0 and proof.exists() and read(proof)['passed']
        write(base/'receipt.json',dict(exit_code=code,full_native_loader_audit_passed=passed,**usage))
        update(dict(active_pid=None,state='completed' if passed else 'failed_preserved'))
        if not passed:raise RuntimeError('Original loader failed; preserve full partial attempt')
    finally:lock.close()


if __name__=='__main__':main()
