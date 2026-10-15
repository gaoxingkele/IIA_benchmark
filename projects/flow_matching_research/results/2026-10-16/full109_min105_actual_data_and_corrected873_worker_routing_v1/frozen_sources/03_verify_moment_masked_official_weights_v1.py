"""Guarded real Small/Base/Large weight check of native short-context masks.

This diagnostic is not an original dataset benchmark or a native source replay.
"""
from __future__ import annotations
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from scripts.flow_matching import prepare_original_full_entity_tsad_v2 as data

ROOT,read,sha,write=data.ROOT,data.read,data.sha,data.write
MODULE='scripts.flow_matching.verify_moment_masked_official_weights_v1'
CONFIG='configs/reproducibility/fm_moment_masked_all3_official_weight_checks.2026-10-16_v1.json'
TEST='tests/test_fm_moment_masked_official_weight_registration_v1.py'
REPORT='all3_official_weights_native_mask_checks.json'
POLICY='configs/runtime/fm_tsad_constant_fit_cpu_controls.v2.json'
SIZES=('small','base','large')
LENGTHS=(100,105,107,234,239,445,464,498,509,511)


def model_path(size):
    return 'configs/models/fm_moment_pretrained_masked_context_v1/fm_moment_pretrained_'+size+'__release512__native_masked_short_context_v1.json'


def registration():
    policy=read(ROOT/POLICY);models=[]
    pins={MODULE.replace('.','/')+'.py',TEST,POLICY,
        'tests/test_fm_moment_masked_context_v1.py',
        'src/iia_benchmark/models/fm_moment_masked_context_v1.py',
        'src/iia_benchmark/models/fm_moment_pretrained.py',
        'scripts/flow_matching/run_moment_masked_context_job_v1.py',
        'scripts/flow_matching/run_tsad_experiment.py',
        'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/resume_anomaly_transformer_full500_metadata_v1.py'}
    for size in SIZES:
        path=model_path(size);c=read(ROOT/path);p=c['parameters']
        if (p['window']!=512 or p['batch_size']!=1 or p['epochs']!=0 or p['seed']!=42
                or c['entrypoint']!='iia_benchmark.models.fm_moment_masked_context_v1:MomentMaskedContextWindow'):
            raise ValueError('Official original512 zero-shot model required')
        models.append(dict(size=size,model_config=path,model_config_sha256=sha(ROOT/path)))
        pins.add(path)
        source=Path(p['source_root'])
        pins.update(str(source/f) for f in ['momentfm/models/moment.py','momentfm/utils/masking.py','momentfm/models/layers/revin.py','momentfm/models/layers/embed.py'])
    return dict(schema_version=1,models=models,context=512,lengths=list(LENGTHS),channels=3,
        fill_values=[0.,10000.],seed=42,CPU_threads=2,python=sys.executable,
        environment=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','torch','safetensors']}),
        settings=policy['settings'],resource_lock=policy['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_masked_all3_official_weight_checks_v1',
        state_root='experiments/runs/fm_moment_masked_all3_official_weight_checks_v1',
        citation='https://arxiv.org/abs/2402.03885',
        boundary='Actual full release weights on deterministic synthetic protocol probes,not benchmark performance. '
            'All3sizes,all10short lengths,allreal targets,fixed512architecture,fill0vs10000bit equality, '
            'original full512 exact score equality and official masked RevIN/patch/attention behavior. '
            'No training,source experiment completion,author equivalence,or native real-dataset replay claim. '
            'Original16/32/16GiB CPU and10/12GiB worker caps remain;no CUDA substitution.')


def verify(c):
    if c!=registration():raise ValueError('All3official sizes/all10lengths/original resource policy required')


def worker(c):
    verify(c)
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','torch','safetensors']})
    if actual!=c['environment']:raise ValueError('Frozen real-weight diagnostic environment required')
    import gc
    import numpy as np
    import torch
    from scripts.flow_matching.model_registry import build_configured_method
    from iia_benchmark.models.fm_moment_masked_context_v1 import MaskedContextWindows,reconstruct_masked_context
    from iia_benchmark.models.fm_moment_pretrained import reconstruct_windows
    output=Path(c['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(c['CPU_threads']);rows=[]
    for item in c['models']:
        config=read(ROOT/item['model_config']);method=build_configured_method(config,dict(device='cpu'))
        method.fit(np.empty((0,512,c['channels']),dtype=np.float32))
        if method.model.seq_len!=512:raise ValueError('Official model architecture changed')
        long=np.random.default_rng(c['seed']).normal(size=(1,512,c['channels'])).astype(np.float32)
        reference=reconstruct_windows(method.model,long,1,'cpu','minimal_patch');native=method.score(long)
        if reference.tobytes()!=native.tobytes():raise ValueError('Original long512 reconstruction changed')
        cases=[]
        for length in c['lengths']:
            values=np.random.default_rng(c['seed']+length).normal(size=(length,c['channels'])).astype(np.float32)
            results=[]
            for fill in c['fill_values']:
                windows=MaskedContextWindows(values,fill_value=fill)
                scores=reconstruct_masked_context(method.model,windows,1,'cpu')
                if scores.shape!=(1,length) or not np.isfinite(scores).all():raise ValueError('Every real point must be scored')
                results.append(scores)
            if results[0].dtype!=results[1].dtype or results[0].tobytes()!=results[1].tobytes():
                raise ValueError('Masked storage fill leaked into real scores')
            cases.append(dict(length=length,real_targets=length,masked_fill_points=512-length,
                score_shape=list(results[0].shape),all_scores_finite=True,fill0_vs10000_byte_equal=True,
                partial_patch_real_points_scored=length%8))
        rows.append(dict(size=item['size'],checkpoint_load_audit=method.checkpoint_load_audit_,
            original_full512_scores_byte_equal=True,cases=cases,model_context=method.model.seq_len,
            fine_tuning_epochs=0,real_dataset_experiments_complete=0))
        del method;gc.collect()
    verify(c)
    write(output/REPORT,dict(passed=True,configuration_sha256=sha(ROOT/CONFIG),models=rows,
        required_sizes=3,short_context_cases=30,mask_fill_comparisons=30,original512_comparisons=3,
        environment=actual,neural_training_performed=False,benchmark_performance=False,
        native_real_dataset_replay_certified=False,all_original_experiments_complete=False))


def controller(c):
    from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as guard
    node=next(n for n in ast.parse(Path(guard.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    names={'closing_all1949_anomaly_sources':'verifying_all3_official_MOMENT_masked_contexts',
        'all1949_anomaly_cache_closure_passed':'all3_official_MOMENT_masked_context_checks_passed'}
    class StateNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in names:return ast.copy_location(ast.Constant(names[n.value]),n)
            return n
    ns=dict(guard.controller.__globals__,MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[StateNames().visit(node)],type_ignores=[])),str(guard.__file__),'exec'),ns)
    return ns['controller'](c)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,sizes=3,short_context_cases=30)));return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:print(json.dumps(dict(passed=True,sizes=3,short_cases=30,torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules)))
    elif a.worker:worker(c)
    else:controller(c)


if __name__=='__main__':main()
