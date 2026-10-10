"""Observe byte-identical long-series author entries; keep the Windows main module safe."""
from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

from scripts.flow_matching.spectral_table2_bootstrap_v1 import execute_original_entry, sha, write


def direct_generator_frame(entry, torch_root):
    frame = inspect.currentframe().f_back
    own = str(Path(__file__).resolve())
    while frame is not None:
        path = Path(frame.f_code.co_filename).resolve()
        if path == entry:
            return frame
        if str(path) != own and not path.is_relative_to(torch_root):
            return None
        frame = frame.f_back
    return None


def verify_training_coverage(state, contract):
    budget = contract['epochs']
    expected = contract['batches_per_epoch']
    if state['optimizer_updates'] != budget * expected:
        raise ValueError('Full original generator update budget not completed')
    if state['epoch_updates'] != {str(i): expected for i in range(budget)}:
        raise ValueError('Full original epoch/batch coverage differs')
    if len(state['evaluations']) != contract['periodic_evaluations']:
        raise ValueError('Full original periodic evaluations missing')


def check_evaluator_repeats(repeats, scores):
    import numpy as np
    for key, output, spread in [('classification','clf_score_mean','clf_score_std'),
                                ('predictive','predictive_score_mean','predictive_score_std'),
                                ('marginal','marginal_score_mean',None)]:
        values = repeats[key]
        if len(values) != 10 or not all(math.isfinite(v) for v in values):
            raise ValueError('All ten full original evaluator repeats required')
        if not math.isclose(float(scores[output]),float(np.mean(values)),rel_tol=1e-6,abs_tol=1e-7):
            raise ValueError('Independent original evaluator mean differs')
        if spread and not math.isclose(float(scores[spread]),float(np.std(values)),rel_tol=1e-6,abs_tol=1e-7):
            raise ValueError('Independent original evaluator spread differs')


def configure_modules(contract):
    source = Path(contract['source_root'])
    sys.path.insert(0,str(source))
    os.chdir(contract['workspace'])
    import torch
    from utils.utils import seed_everything
    from utils.utils_args import parse_args_uncond_spectral, parse_args_uncond
    old = sys.argv
    sys.argv = ['author',*contract['arguments']]
    try:
        args = (parse_args_uncond_spectral if contract['method']=='spectral_flow' else parse_args_uncond)()
    finally:
        sys.argv = old
    seed_everything(args.seed if contract['stage']=='training' else args.test_seed)
    args.device = 'cuda'
    return torch,args


def run_stage(contract_path):
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
    entry = Path(contract['entry']).resolve()
    if sha(entry)!=contract['entry_sha256']:
        raise ValueError('Original entry differs')
    torch,args = configure_modules(contract)
    if not torch.cuda.is_available():
        raise ValueError('Full original CUDA stage required')
    state=dict(status='running',stage=contract['stage'],entry_sha256=sha(entry),
        contract_sha256=sha(contract_path),optimizer_updates=0,epoch_updates={},evaluations=[],
        diagnostic_only=False,full_original_budget=True)
    receipt=Path(contract['receipt']);write(receipt,state)
    metrics=importlib.import_module('metrics')
    long_metrics=importlib.import_module('metrics.metrics_long_range')
    originals=[];holder={};active_repeats=None
    data_module=importlib.import_module('utils.utils_data')
    original_loader=data_module.gen_dataloader
    @functools.wraps(original_loader)
    def audited_loader(settings):
        import numpy as np
        generator=torch.Generator().set_state(torch.get_rng_state())
        with np.load(contract['normalized_arrays'],allow_pickle=False) as payload:
            normalized=payload['normalized']
        order=torch.randperm(len(normalized),generator=generator).numpy()
        boundary=int(len(normalized)*.8)
        loaders=original_loader(settings)
        for loader,ids in zip(loaders,(order[:boundary],order[boundary:])):
            actual=loader.dataset.tensors[0].numpy()
            if (not np.allclose(actual,normalized[ids],rtol=2e-6,atol=2e-6)
                    or loader.num_workers!=contract['num_workers']
                    or loader.batch_size!=contract['batch_size'] or loader.drop_last):
                raise ValueError('Original full normalization/split/loader differs')
        if not torch.equal(torch.get_rng_state(),generator.get_state()):
            raise ValueError('Original full data loader RNG consumption differs')
        state['data_audit']=dict(train_ids=order[:boundary].tolist(),test_ids=order[boundary:].tolist(),
            full_normalized_rows_verified=len(normalized),normalization='original full-trajectory mean/std before series split',
            num_workers=settings.num_workers,actual_seq_len=settings.seq_len,split_overlap=0)
        write(receipt,state)
        return loaders
    data_module.gen_dataloader=audited_loader
    originals.append((data_module,'gen_dataloader',original_loader))
    def observe_repeated(name,key):
        original=getattr(long_metrics,name)
        @functools.wraps(original)
        def measured(*a,**kw):
            value=original(*a,**kw)
            number=float(value['marginal_loss']) if key=='marginal' else float(value)
            if not math.isfinite(number):raise ValueError('Nonfinite full original evaluation')
            active_repeats[key].append(number)
            return value
        setattr(long_metrics,name,measured);originals.append((long_metrics,name,original))
    for name,key in [('compute_classification_score','classification'),
                     ('compute_predictive_score','predictive'),('compute_test_metrics','marginal')]:
        observe_repeated(name,key)
    original_metric=metrics.evaluate_model_uncond
    @functools.wraps(original_metric)
    def evaluated(real,generated,settings):
        nonlocal active_repeats
        import numpy as np
        if (tuple(real.shape)!=tuple(contract['test_shape']) or generated.shape!=real.shape
                or not np.isfinite(real).all() or not np.isfinite(generated).all()):
            raise ValueError('Full original test arrays missing, reduced, or nonfinite')
        active_repeats={k:[] for k in ('classification','predictive','marginal')}
        index=len(state['evaluations'])
        rng_path=Path(contract['array_root'])/('evaluation_'+str(index)+'_rng.pt')
        rng_path.parent.mkdir(parents=True,exist_ok=True)
        if rng_path.exists():raise FileExistsError('Preserve original evaluator RNG snapshot')
        torch.save(dict(python=__import__('random').getstate(),numpy=np.random.get_state(),
            torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),arguments=vars(settings)),rng_path)
        scores=original_metric(real,generated,settings)
        check_evaluator_repeats(active_repeats,scores)
        artifact=Path(contract['array_root'])/('evaluation_'+str(index)+'.npz')
        artifact.parent.mkdir(parents=True,exist_ok=True)
        if artifact.exists():raise FileExistsError('Preserve original evaluation arrays')
        np.savez_compressed(artifact,real=real,generated=generated)
        state['evaluations'].append(dict(scores={k:float(v) for k,v in scores.items()},
            repeats=active_repeats,arrays=str(artifact),arrays_sha256=sha(artifact),
            evaluator_rng=str(rng_path),evaluator_rng_sha256=sha(rng_path),
            shape=list(real.shape),generator_updates=state['optimizer_updates']))
        write(receipt,state)
        return scores
    metrics.evaluate_model_uncond=evaluated
    originals.append((metrics,'evaluate_model_uncond',original_metric))
    optimizer_class=(importlib.import_module('models.spectral_flow.muon').SingleDeviceMuonWithAuxAdam
                     if contract['method']=='spectral_flow' else torch.optim.AdamW)
    original_step=optimizer_class.step
    @functools.wraps(original_step)
    def stepped(optimizer,*a,**kw):
        frame=direct_generator_frame(entry,Path(torch.__file__).resolve().parent)
        result=original_step(optimizer,*a,**kw)
        if frame is not None:
            epoch=int(frame.f_locals['epoch'])
            state['optimizer_updates']+=1
            state['epoch_updates'][str(epoch)]=state['epoch_updates'].get(str(epoch),0)+1
            holder.update(frame=frame,optimizer=optimizer)
            if not bool(torch.isfinite(frame.f_locals['loss']).all()):
                raise ValueError('Nonfinite full original generator loss')
            write(receipt,state)
        return result
    optimizer_class.step=stepped
    originals.append((optimizer_class,'step',original_step))
    begin=time.perf_counter();namespace={}
    try:
        # The exact original main/eval source executes once. No __main__ replacement,
        # no rewritten optimizer/training loop, no budget or worker override.
        execute_original_entry(entry,contract['arguments'],namespace)
        if contract['stage']=='training':
            verify_training_coverage(state,contract)
            local=holder['frame'].f_locals
            checkpoint=dict(epoch=int(local['epoch']),optimizer_updates=state['optimizer_updates'],
                model=local['model'].state_dict(),optimizer=holder['optimizer'].state_dict(),
                arguments=vars(local['args']),python_rng=__import__('random').getstate(),
                numpy_rng=__import__('numpy').random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),full_budget_completed=True)
            torch.save(checkpoint,contract['final_checkpoint'])
            state['final_checkpoint_sha256']=sha(contract['final_checkpoint'])
            best=Path(local['args'].log_dir)/'ckpt.pth'
            if not best.is_file():raise ValueError('Original marginal-selected checkpoint missing')
            state.update(selected_checkpoint=str(best),selected_checkpoint_sha256=sha(best))
        elif state['optimizer_updates'] or len(state['evaluations'])!=1:
            raise ValueError('Full original standalone evaluation required')
        state.update(status='completed',seconds=time.perf_counter()-begin)
        write(receipt,state)
    except BaseException as error:
        state.update(status='failed_or_partial_preserved',error=repr(error),seconds=time.perf_counter()-begin)
        write(receipt,state);raise
    finally:
        for module,name,original in reversed(originals):setattr(module,name,original)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',required=True)
    run_stage(parser.parse_args().contract)
