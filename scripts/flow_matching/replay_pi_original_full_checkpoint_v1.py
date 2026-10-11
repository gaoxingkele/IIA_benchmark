"""Replay every original Pi score with independently reconstructed sampler RNG."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

from scripts.flow_matching.audit_pi_original_recovery_completion_v1 import ROOT, read, digest, binding, same_numbers
from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import resource_api, capped_wait


def rng_digest(state):
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def restore_evaluation_rng(detector, job, epochs, cache):
    import torch
    from torch.utils.data import DataLoader
    # Original dataset/collation is deterministic. Consuming index-only loaders
    # preserves every sampler/base-seed draw without repeating optimization.
    def consume(count, shuffle):
        loader = DataLoader(range(count),batch_size=detector.batch_size,shuffle=shuffle,num_workers=0)
        return sum(len(batch) for batch in loader)
    before = rng_digest(torch.get_rng_state())
    if before != cache['rng_before_sha256']:
        raise ValueError('Original initialization/DFA entry RNG cannot be reconstructed')
    count = len(detector.train_loader.dataset)
    if consume(count, True) != count:
        raise ValueError('Original complete DFA sampler coverage differs')
    indices = torch.randperm(count)[:max(100,count//10)]
    if hashlib.sha256(indices.numpy().tobytes()).hexdigest() != cache['selection_index_sha256']:
        raise ValueError('Original DFA permutation hash differs')
    after = rng_digest(torch.get_rng_state())
    if after != cache['rng_after_sha256']:
        raise ValueError('Original DFA exit RNG cannot be reconstructed')
    if any(isinstance(m,torch.nn.Dropout) and m.p != 0 for m in detector.model.modules()):
        raise ValueError('Nonzero training dropout requires separately preserved original RNG evidence')
    for epoch in epochs:
        if consume(count,True) != count or consume(len(detector.test_loader.dataset),False) != len(detector.test_loader.dataset):
            raise ValueError('All original train/test-validation sampler draws required')
    return dict(original_DFA_entry_rng_sha256=before,original_DFA_exit_rng_sha256=after,
                original_DFA_selection_index_sha256=cache['selection_index_sha256'],
                reconstructed_post_training_rng_sha256=rng_digest(torch.get_rng_state()),
                fully_consumed_original_training_epochs=len(epochs),original_train_windows=count,
                original_test_validation_windows=len(detector.test_loader.dataset),
                original_batch_size=detector.batch_size,optimizer_updates_repeated=0)


def compare_complete_streams(saved, actual, tolerance):
    import numpy as np
    if set(saved) != set(actual):
        raise ValueError('All original full saved streams required')
    diagnostics = {}
    exact = {'labels','raw_predictions','pa_predictions','threshold','med_E','iqr_E','med_P','iqr_P'}
    for key, expected in saved.items():
        observed = actual[key]
        if observed.shape != expected.shape or not np.isfinite(observed).all():
            raise ValueError('Full finite stream geometry differs: '+key)
        agrees = np.array_equal(observed,expected) if key in exact else np.allclose(observed,expected,**tolerance)
        if not agrees:
            raise ValueError('Full original Pi checkpoint stream differs: '+key)
        diagnostics[key] = dict(shape=list(expected.shape),bitwise_equal=bool(np.array_equal(observed,expected)),
                               maximum_absolute_error=float(np.max(np.abs(observed-expected))))
    return diagnostics


def worker(config, config_path, rng_only=False):
    import numpy as np
    import torch
    import yaml
    case, job, result = binding(config)
    proof = read(config['independent_completion'])
    if not proof['passed'] or proof['original_id'] != job['id'] or proof['result_sha256'] != digest(proof['result_path']):
        raise ValueError('Full original completion must independently pass first')
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    if torch.__version__ != result['torch_version']:
        raise ValueError('Original full native PyTorch environment required')
    source = ROOT/case['job']['output_directory']
    target = ROOT/config['artifact_directory']
    if not rng_only:
        target.mkdir(parents=True,exist_ok=False)
        checkpoint = target/'checkpoints'
        checkpoint.mkdir()
        for name in (job['dataset']+'_checkpoint.pth',job['dataset']+'_global_hurst.pt'):
            shutil.copyfile(source/'checkpoints'/name,checkpoint/name)
            if digest(checkpoint/name) != digest(source/'checkpoints'/name):
                raise ValueError('Keep every full original selected weight')
    effective = yaml.safe_load((source/'effective_author_config.yaml').read_text())
    if not rng_only:
        effective['general']['model_dir'] = str(checkpoint)
        effective['general']['dfa_cache'] = str(target/'unused_dfa_cache.dat')
    path = target/'effective_replay_config.yaml' if not rng_only else source/'effective_author_config.yaml'
    if not rng_only:
        path.write_text(yaml.safe_dump(effective),encoding='utf-8')
    sys.path.insert(0,str(ROOT/job['runtime']))
    from main import AnomalyDetection
    from utils.utils import set_seed
    set_seed(job['seed'])
    detector = AnomalyDetection(str(path),job['dataset'])
    if detector.batch_size != job['author_config']['data']['batch_size'] or detector.n_epochs != job['author_config']['training']['n_epochs']:
        raise ValueError('Full original batch and training recipe must be preserved')
    if sum(p.numel() for p in detector.model.parameters()) != result['parameter_count']:
        raise ValueError('Full original model architecture required')
    if len(detector.model.encoder.attn_layers) != result['active_attention_layers']:
        raise ValueError('Original active source layers required')
    cache = read(ROOT/case['settings']['dfa_cache_root']/(case['job']['id']+'.json'))
    rng = restore_evaluation_rng(detector,job,result['epochs'],cache)
    weights = torch.load(source/'checkpoints'/(job['dataset']+'_checkpoint.pth'),map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Full finite original checkpoint required')
    detector.model.load_state_dict(weights,strict=True)
    if rng_only:
        out = ROOT/config['rng_probe_output']
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as handle:
            json.dump(dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,
                original_id=job['id'],recovery_id=case['job']['id'],result_sha256=digest(source/'result.json'),
                original_checkpoint_sha256=digest(source/'checkpoints'/(job['dataset']+'_checkpoint.pth')),
                original_full_checkpoint_loaded_strictly=True,full_parameter_count=result['parameter_count'],
                independent_full_model_reinference_passed=False,rng=rng,additional_training_seed_slots=0,
                boundary='Initialization and all original sampler RNG hashes independently reconstructed; full trained checkpoint loaded strictly. No forward score re-inference claimed.'),handle,indent=2)
            handle.write('\n')
        print(json.dumps(dict(rng_probe_passed=True,full_checkpoint_loaded_strictly=True,rng=rng)))
        return
    with torch.no_grad():
        returned = detector.test()
    with np.load(source/'checkpoints/author_scores.npz',allow_pickle=False) as archive:
        saved = {k:archive[k] for k in archive.files}
    with np.load(checkpoint/'author_scores.npz',allow_pickle=False) as archive:
        actual = {k:archive[k] for k in archive.files}
    streams = compare_complete_streams(saved,actual,config['score_tolerance'])
    same_numbers(dict(zip(('accuracy','precision','recall','f1'),returned)),
                 {k:result['metrics'][k] for k in ('accuracy','precision','recall','f1')})
    if torch.cuda.is_initialized():
        raise ValueError('Full native CPU replay must not initialize CUDA')
    final = dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,original_id=job['id'],
        recovery_id=case['job']['id'],original_checkpoint_sha256=digest(source/'checkpoints'/(job['dataset']+'_checkpoint.pth')),
        original_result_sha256=digest(source/'result.json'),original_full_checkpoint_loaded_strictly=True,
        full_parameter_count=result['parameter_count'],full_native_batch_size=detector.batch_size,cpu_threads=torch.get_num_threads(),
        full_original_training_score_points=len(saved['train_energy']),full_original_test_score_points=len(saved['labels']),
        dropped_original_test_tail_points=result['dropped_test_tail_points'],all_original_score_streams_replayed=streams,
        rng=rng,full_native_checkpoint_reinference_passed=True,completed_original_training_seed_slots=1,
        additional_training_seed_slots=0,strict_TAB_result=False,author_equivalence_certified=False,
        config=config_path,config_sha256=digest(config_path),boundary=config['boundary'])
    (target/'full_original_checkpoint_replay.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(full_original_checkpoint_replay_passed=True,original_id=job['id'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--rng-probe-only',action='store_true')
    args = parser.parse_args(); config = read(args.config)
    binding(config)
    if args.worker or args.rng_probe_only:
        return worker(config,args.config,args.rng_probe_only)
    api = resource_api(); base = ROOT/config['state_root']; base.mkdir(parents=True,exist_ok=True)
    owner = api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('This original full Pi replay controller already runs')
    if (base/'receipt.json').exists():
        owner.close()
        raise FileExistsError('Keep completed or failed full replay attempts immutable')
    state = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):
        state.update(values)
        (base/'status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child = subprocess.Popen([config['python'],'-X','utf8','-u','-m',
                    'scripts.flow_matching.replay_pi_original_full_checkpoint_v1','--config',args.config,'--worker'],cwd=ROOT,
                    stdout=out,stderr=err,env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                    PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                    OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid,active_create_time=handle.create_time()))
                code,usage = capped_wait(child,config,api,update)
        final = ROOT/config['artifact_directory']/'full_original_checkpoint_replay.json'
        passed = code == 0 and final.exists() and read(final)['full_native_checkpoint_reinference_passed']
        receipt = dict(exit_code=code,full_native_checkpoint_reinference_passed=passed,additional_training_seed_slots=0,**usage)
        (base/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,**receipt))
        if not passed:
            raise RuntimeError('Full original Pi replay failed; preserve all evidence')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
