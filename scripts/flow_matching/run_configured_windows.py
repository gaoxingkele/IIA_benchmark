"""Run frozen window train/validation/test arrays through local model configs.

This saves raw scores for later protocol-specific evaluation. It never selects
a threshold with test labels or emits a leaderboard result. Input windows must
already be grouped and preprocessed under an independently audited recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.flow_matching.model_registry import ROOT, build_configured_method, load_model_config
from scripts.literature.import_flow_matching_archives import contained


def run(root, config):
    model_path=contained(root,config['model_config'])
    input_path=contained(root,config['input_npz'])
    for path,expected in [(model_path,config['model_config_sha256']),(input_path,config['input_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen input checksum mismatch: '+path.name)
    model_config=load_model_config(model_path)
    input_audit=config.get('input_audit',{})
    if config.get('purpose')!='preflight' and not input_audit.get('grouped_splits_verified'):
        raise ValueError('Formal inputs require a separate grouped split audit')
    frozen=json.dumps(config,sort_keys=True).encode()
    fingerprint=hashlib.sha256(frozen).hexdigest()
    output=contained(root,config['output_directory']);marker=output/'run.json'
    if output.exists():
        if marker.is_file():
            previous=load_model_config(marker)
            if previous['experiment_sha256']==fingerprint:
                score_path=output/'raw_window_scores.npz'
                if not score_path.is_file() or hashlib.sha256(score_path.read_bytes()).hexdigest()!=previous['scores_sha256']:
                    raise ValueError('Existing score artifact is missing or changed; run preserved')
                return previous
        raise FileExistsError('Existing run preserved; choose a new output directory')
    with np.load(input_path,allow_pickle=False) as payload:
        arrays={name:payload[config.get('array_keys',{}).get(name,name)].copy()
                for name in ('train','validation','test')}
    if any(a.ndim!=3 or not len(a) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Expected nonempty finite [windows,length,features] arrays')
    if len({a.shape[1:] for a in arrays.values()})!=1:raise ValueError('Split feature/window shapes differ')
    model=build_configured_method(model_config,config.get('parameter_overrides'))
    if not callable(getattr(model,'fit',None)) or not callable(getattr(model,'score',None)):
        raise ValueError('This implementation is a component, not a fit/score window detector')
    started=time.monotonic();model.fit(arrays['train']);elapsed=time.monotonic()-started
    scores={name:np.asarray(model.score(values)) for name,values in arrays.items()}
    for name,score in scores.items():
        if score.shape not in {(len(arrays[name]),),(len(arrays[name]),arrays[name].shape[1])} or not np.isfinite(score).all():
            raise ValueError('Unaligned/nonfinite raw score array: '+name)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'raw_window_scores.npz',**scores)
    result={'id':config['id'],'experiment_sha256':fingerprint,'model_config':config['model_config'],
            'model_config_sha256':config['model_config_sha256'],'input_sha256':config['input_sha256'],
            'parameter_overrides':config.get('parameter_overrides',{}),'input_audit':input_audit,
            'score_shapes':{k:list(v.shape) for k,v in scores.items()},'training_seconds':elapsed,
            'status':'preflight_completed' if config.get('purpose')=='preflight' else 'raw_scores_generated_pending_evaluation',
            'scores_sha256':hashlib.sha256((output/'raw_window_scores.npz').read_bytes()).hexdigest(),
            'boundary':'Raw window scores only. No threshold/metric/leaderboard; no paper reproduction claim.'}
    marker.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(ROOT,load_model_config(args.experiment)),indent=2))


if __name__=='__main__':main()
