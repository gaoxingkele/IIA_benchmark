"""Verify frozen original image recipe; never execute or certify training."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path


def read(root, path):
    return json.loads((root/path).read_text(encoding='utf-8'))


def validate_protocol(config):
    training=config['original_training']
    required=dict(model_per_dataset=1,category_conditioning=True,epochs=2000,
        batch_size=32,gradient_accumulation=1,optimizer='StableAdamW',amsgrad=True,
        learning_rate=1e-4,weight_decay=1e-4,betas=[.9,.99],epsilon=1e-6,
        update_RMS_clip_threshold=1.0,gradient_norm_clipping=False,
        mixed_precision='bfloat16',TF32=True,distributed_data_parallel=True)
    if any(training.get(k)!=v for k,v in required.items()):
        raise ValueError('Original Appendix C.1 training budget or optimizer settings differ')
    score=config['original_score']
    if score['paths']!=10 or score['times']!=10 or score['aggregation']!='minimum' or score['time_weight']!='t^2':
        raise ValueError('Original Eq.4/default K10 T10 scoring differs')
    grid=list(range(2,21,2))
    expected=[(5,n) for n in grid]+[(n,5) for n in grid]+[(n,n) for n in grid]
    compute=config['Table6_compute']
    if [(r['K'],r['T']) for r in compute]!=expected:
        raise ValueError('All thirty exact original Table6 rows must be retained in order')
    for row in compute:
        if row['forward_evaluations_per_image']!=row['K']*row['T']:
            raise ValueError('Native KT compute budget differs')
    epochs=config['Table9_epoch_checkpoints']
    if [r['checkpoint_epoch'] for r in epochs]!=list(range(100,2001,100)):
        raise ValueError('All twenty original full-duration checkpoints are required')
    for dataset in ['MVTec-AD','VisA']:
        rows=[r for r in config['Table7_path_aggregation'] if r['dataset']==dataset]
        if [(r['aggregation'],r['percentile']) for r in rows]!=[('average',None),('percentile',.3),('percentile',.2),('percentile',.1),('minimum',None)]:
            raise ValueError('All original path aggregation choices required on both datasets')
        conditional=[r for r in config['Table8_conditional_mean'] if r['dataset']==dataset]
        if [r['category_conditioning'] for r in conditional]!=[False,True]:
            raise ValueError('Conditional and pooled training are separate obligations')
    if config['training_exceptions']['TCCM']['epochs']!=50:
        raise ValueError('Native TCCM comparison uses fifty epochs, not default2000')
    if config['original_shared_checkpoint']!=['Flow Mismatching','D-Flow','ReconFlow']:
        raise ValueError('Original inference-only comparisons share one trained FM checkpoint')
    if not config['unresolved_fidelity'] or config['full_original_training_executed'] or config['strict_TAB_equivalent'] or config['journal_reproduction_certified']:
        raise ValueError('Unexecuted equation registration cannot certify original training or TAB results')
    for key in ['Table6_compute','Table7_path_aggregation','Table8_conditional_mean','Table9_epoch_checkpoints','Table2_same_backbone_FM_comparison']:
        for row in config[key]:
            required_metrics=set(config['metrics']) | ({'FPS'} if key=='Table6_compute' else set())
            if set(row['paper_reported'])!=required_metrics:
                raise ValueError('Original metric names/units must not be discarded')
            if any(not isinstance(v,(int,float)) or not 0<=v<=100 for k,v in row['paper_reported'].items() if k!='FPS'):
                raise ValueError('Paper percentage transcriptions must remain within [0,100]')
    return sum(len(row['paper_reported']) for key in ['Table6_compute','Table7_path_aggregation','Table8_conditional_mean','Table9_epoch_checkpoints','Table2_same_backbone_FM_comparison'] for row in config[key])


def audit(root, path):
    config=read(root,path)
    values=validate_protocol(config)
    receipts=[config['paper']]+config['source_receipts']
    for receipt in receipts:
        actual=hashlib.sha256((root/receipt['path']).read_bytes()).hexdigest()
        if actual!=receipt['sha256']:
            raise ValueError('Frozen paper/code dependency changed: '+receipt['path'])
    model=read(root,'configs/models/fm_flow_mismatching_original.v1.json')
    if model['score']['paths']!=10 or model['score']['times']!=10 or model['benchmark_ready']:
        raise ValueError('New original configuration must not inherit development8/8 or readiness')
    comparisons=read(root,'configs/models/fm_flow_mismatching_appendix_c_comparisons.v1.json')
    resolved=[]
    for method in comparisons['methods']:
        for field in ['objective','corruption','normalization','heatmap']:
            if method.get(field):
                module,name=method[field].split(':')
                if not callable(getattr(importlib.import_module(module),name)):
                    raise ValueError('Registered equation function is not callable')
                resolved.append(method[field])
        if method['id']=='D_Flow_FlowGrad_image_domain' and (method['callable'] is not None or 'missing' not in method['reproduction_status']):
            raise ValueError('Unavailable original FlowGrad must remain explicitly missing')
    return dict(passed=True,paper_sha256=config['paper']['sha256'],verified_source_dependencies=len(receipts),
        original_compute_rows=30,original_epoch_checkpoints=20,original_aggregator_rows=10,
        original_conditional_mean_rows=4,original_same_backbone_comparison_rows=6,
        paper_reported_metric_values=values,verified_equation_callables=len(resolved),
        full_original_training_executed=False,new_local_measured_performance_values=0,
        all_original_experiments_complete=False,strict_TAB_equivalent=False,
        boundary='Frozen recipe and equation callable verification; paper reported metrics do not constitute local benchmark performance.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/experiments/fm_flow_mismatching_original_protocol.v1.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    report=audit(root,args.config)
    output=root/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
