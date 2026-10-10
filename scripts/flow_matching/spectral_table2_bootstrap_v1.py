"""Windows-safe execution of byte-identical original training and evaluation entries.

Only stdlib is imported at module scope. Multiprocessing retains this guarded
outer module as __main__; the unguarded author VQ entry executes once in a
separate namespace. Source optimizer calls are observed, never replaced by a
different optimizer or training loop.
"""
import argparse
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def execute_original_entry(path, arguments, namespace=None):
    """Do not set sys.modules['__main__'] or use runpy on top-level author code."""
    original_main = sys.modules.get('__main__')
    old_argv = sys.argv
    namespace = namespace if namespace is not None else {}
    namespace.update(__name__='__main__', __file__=str(Path(path).resolve()), __builtins__=__builtins__)
    sys.argv = [str(path), *arguments]
    try:
        try:
            exec(compile(Path(path).read_bytes(), str(Path(path).resolve()), 'exec'), namespace)
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv = old_argv
        if sys.modules.get('__main__') is not original_main:
            raise RuntimeError('Original source changed outer multiprocessing entry identity')
    return namespace


def direct_entry_frame(entry, torch_root):
    frame = inspect.currentframe().f_back
    own = str(Path(__file__).resolve())
    while frame is not None:
        filename = str(Path(frame.f_code.co_filename).resolve())
        if filename == str(entry):
            return frame
        if filename != own and not Path(filename).is_relative_to(torch_root):
            return None  # Nested TS2Vec/evaluator optimizer is not a generator update.
        frame = frame.f_back
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    entry = Path(contract['entry']).resolve()
    if sha(entry) != contract['entry_sha256']:
        raise ValueError('Original author entry differs')
    receipt = Path(contract['receipt'])
    if receipt.exists():
        raise FileExistsError('Keep completed or partial stage receipts')
    os.chdir(contract['workspace']); sys.path.insert(0, contract['workspace'])
    import torch
    torch_root = Path(torch.__file__).resolve().parent
    state = dict(stage=contract['stage'], contract_sha256=sha(contract_path), entry_sha256=sha(entry),
        optimizer_updates=0, required_optimizer_updates=contract['optimizer_updates'],
        status='running', metrics=[], data_loaders=[], author_entry_bytes_unchanged=True,
        multiprocessing_outer_entry_preserved=True, benchmark_smoke=False)
    frame_holder = {}; originals = []
    def persist():
        write(receipt.with_name(receipt.stem + '.progress.json'), state)
    def patch_optimizer(cls):
        original = cls.step; originals.append((cls, original))
        @functools.wraps(original)
        def observed(optimizer, *a, **kw):
            frame = direct_entry_frame(entry, torch_root)
            result = original(optimizer, *a, **kw)
            if frame is not None:
                state['optimizer_updates'] += 1
                frame_holder['frame'] = frame; frame_holder['optimizer'] = optimizer
                if state['optimizer_updates'] % 500 == 0:
                    persist()
            return result
        cls.step = observed
    patch_optimizer(torch.optim.AdamW); patch_optimizer(torch.optim.Adam)
    original_loader_init = torch.utils.data.DataLoader.__init__
    def observed_loader_init(loader, *a, **kw):
        original_loader_init(loader, *a, **kw)
        caller = inspect.currentframe().f_back
        source_file = Path(caller.f_code.co_filename).resolve()
        if source_file.is_relative_to(Path(contract['workspace']).resolve()) and (
                source_file.name == 'dataset_VQ.py' or source_file == entry):
            state['data_loaders'].append(dict(source=source_file.name, num_workers=loader.num_workers,
                batch_size=loader.batch_size, drop_last=loader.drop_last, samples=len(loader.dataset),
                sampler=type(loader.sampler).__name__))
            persist()
    torch.utils.data.DataLoader.__init__ = observed_loader_init
    # Wrap functions before the author's from-import statements bind them.
    def patch_metric(module_name, function_name, metric):
        import importlib
        module = importlib.import_module(module_name)
        original = getattr(module, function_name)
        @functools.wraps(original)
        def observed(*a, **kw):
            begin = time.perf_counter(); result = original(*a, **kw)
            number = float(result)
            if not math.isfinite(number):
                raise ValueError('Nonfinite original evaluator result')
            state['metrics'].append(dict(metric=metric, value=number, seconds=time.perf_counter()-begin,
                generator_optimizer_updates=state['optimizer_updates'], reference_samples=len(a[0]),
                generated_samples=len(a[1])))
            persist(); return result
        setattr(module, function_name, observed)
    patch_metric('metrics.discriminative_metrics','discriminative_score_metrics','discriminative')
    patch_metric('metrics.predictive_metrics','predictive_score_metrics2','predictive')
    patch_metric('metrics.context_fid','context_fid','context_fid')
    start = time.perf_counter(); namespace = {}
    persist()
    try:
        execute_original_entry(entry, contract['arguments'], namespace)
        if state['optimizer_updates'] != contract['optimizer_updates']:
            raise ValueError('Full source training update budget was not completed')
        live = frame_holder['frame'].f_locals if frame_holder else namespace
        if contract['optimizer_updates']:
            snapshot = dict(optimizer_updates=state['optimizer_updates'], entry_sha256=sha(entry),
                contract_sha256=sha(contract_path), optimizer=frame_holder['optimizer'].state_dict(),
                models={key:live[key].state_dict() for key in contract['model_keys']},
                final_iteration=int(live['nb_iter']), arguments=vars(live['args']),
                selection={k:float(live[k]) for k in ('best_ds','best_iter','best_iter_test') if k in live})
            torch.save(snapshot, contract['final_checkpoint'])
            state['final_checkpoint_sha256'] = sha(contract['final_checkpoint'])
        elif contract['stage'] == 'evaluation':
            for metric in ('context_fid','discriminative','predictive'):
                if len([r for r in state['metrics'] if r['metric']==metric]) != 5:
                    raise ValueError('All five original metric repeats are required')
        state.update(status='completed', seconds=time.perf_counter()-start,
            completed_utc=datetime.now(timezone.utc).isoformat())
        write(receipt, state)
    except BaseException as error:
        state.update(status='failed_or_partial_preserved', error=repr(error), seconds=time.perf_counter()-start)
        write(receipt, state); raise
    finally:
        for cls, original in originals:
            cls.step = original
        torch.utils.data.DataLoader.__init__ = original_loader_init


if __name__ == '__main__':
    main()
