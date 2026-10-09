"""Record released-vs-paper parameter counts without altering author models."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write
from scripts.flow_matching.run_spectral_original_job import verify


def parameter_audit(model,paper_count):
    named=list(model.named_parameters())
    total=sum(p.numel() for _,p in named)
    positions=[{'name':n,'shape':list(p.shape),'parameters':p.numel(),'trainable':p.requires_grad}
               for n,p in named if n in ('model.pos_enc.pe','model.pos_dec.pe')]
    return {'all_parameters':total,'trainable_parameters':sum(p.numel() for _,p in named if p.requires_grad),
        'paper_reported_parameters':paper_count,'learned_position_parameters':positions,
        'difference':total-paper_count if paper_count is not None else None,
        'difference_matches_position_parameter_count':total-paper_count==sum(p['parameters'] for p in positions) if paper_count is not None else None,
        'paper_count_explanation':'Difference matches learned positional embeddings; this numerical identity is an inference, not confirmation of the authors counting convention.',
        'author_model_altered':False}


def main():
    previous='configs/experiments/fm_spectral_mean_flow_original_queue.v1.json'
    target=ROOT/'configs/experiments/fm_spectral_mean_flow_original_queue.v2.json'
    if target.exists():raise FileExistsError('Preserve parameter-audited queue')
    source=read(ROOT/previous);policy=verify(source);queue=deepcopy(source)
    from iia_benchmark.models.fm_spectral_author import build_author_model
    records={}
    for job in queue['jobs']:
        key=job['model_config']
        if key not in records:
            config=read(ROOT/key);parameters=config['parameters']
            model=build_author_model(ROOT/parameters['source_root'],parameters['model_target'],parameters['parameters'])
            audit=parameter_audit(model,job['expected_parameters'])
            if audit['difference'] not in (None,0) and not audit['difference_matches_position_parameter_count']:
                raise ValueError('Unexplained released-model count mismatch')
            records[key]=audit
            del model
        job['paper_reported_parameters']=records[key]['paper_reported_parameters']
        job['expected_parameters']=records[key]['all_parameters']
    queue['schema_version']=2
    queue['preflight_failure_predecessor']={'queue':previous,'sha256':sha(ROOT/previous),'source_commit':'b2428d5',
        'reason':'Published Diffusion-TS parameter counts omit a quantity matching its two learned positional embeddings; source unchanged, counts separate.'}
    receipt='configs/reproducibility/spectral_mean_flow_parameter_audit.v2.json'
    write(ROOT/receipt,{'author_models_unchanged':True,'models':records,'preflight_is_not_benchmark_performance':True})
    for path in [receipt,'scripts/flow_matching/register_spectral_parameter_audit_v2.py','tests/test_fm_spectral_parameter_audit_v2.py']:
        queue['source_receipts'].append({'path':path,'sha256':sha(ROOT/path)})
    write(target,queue)
    print(json.dumps({'registered_jobs':len(queue['jobs']),'model_parameter_audits':len(records),'queue':target.relative_to(ROOT).as_posix()}))


if __name__=='__main__':main()
