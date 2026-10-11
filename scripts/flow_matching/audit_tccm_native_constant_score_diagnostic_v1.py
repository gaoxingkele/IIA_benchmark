"""Whole native MVTec class-imbalance diagnostic; no trained model performance."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha,verify_receipts
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    queue=read(ROOT/args.queue);verify_receipts(queue['source_receipts'])
    case=queue['cases'][0]
    job=read(ROOT/case['training_configuration']);verify_receipts(job['source_receipts'])
    dataset=read(ROOT/job['dataset_configuration'])
    verify_receipts(dataset['source_receipts']);verify_receipts(dataset['array_receipts'])
    import numpy as np
    from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve
    derived=next(x['path'] for x in dataset['source_receipts'] if Path(x['path']).name=='full_derived_image_preprocessing.json')
    names=read(ROOT/derived)['original_categories']
    arrays={x['field']:np.load(x['path'],mmap_mode='r',allow_pickle=False)
        for x in dataset['array_receipts'] if x['split']=='test' and x['field']!='images'}
    if (len(names)!=15 or arrays['labels'].shape!=(1725,) or arrays['masks'].shape!=(1725,128,128)
            or arrays['categories'].shape!=(1725,)):
        raise ValueError('Whole original1725/15 native test truth required')
    rows=[]
    for index,name in enumerate(names):
        subset=arrays['categories']==index
        labels=arrays['labels'][subset];masks=arrays['masks'][subset]
        if not np.isin(labels,(0,1)).all() or not np.isin(masks,(0,1)).all():
            raise ValueError('Native binary image labels and segmentation truth required')
        ip=int(labels.sum());it=len(labels);pp=int(masks.sum());pt=masks.size
        if not 0<ip<it or not 0<pp<pt:
            raise ValueError('Complete native positive/negative units required')
        metrics={}
        for prefix,positives,total in [('image_',ip,it),('pixel_',pp,pt)]:
            # Two exact weighted score groups represent every truth unit at a
            # constant score. No sampling, model inference or synthetic benchmark.
            truth=np.array([0,1]);prediction=np.array([0.,0.]);weights=np.array([total-positives,positives])
            precision,recall,_=precision_recall_curve(truth,prediction,sample_weight=weights)
            values=dict(AUROC=float(roc_auc_score(truth,prediction,sample_weight=weights)),
                AP=float(average_precision_score(truth,prediction,sample_weight=weights)),
                F1_max=float((2*precision*recall/np.maximum(precision+recall,1e-100)).max()))
            expected=dict(AUROC=.5,AP=positives/total,F1_max=2*positives/(total+positives))
            if any(abs(values[k]-v)>1e-12 for k,v in expected.items()):
                raise ValueError('Independent weighted constant-score metric differs from exact full-population formula')
            metrics.update({prefix+k:v for k,v in values.items()})
        # Constant score has PRO=FPR along the sole tied linear segment.
        metrics['pixel_AUPRO']=queue['metrics']['aupro_fpr_limit']/2
        rows.append(dict(category=name,category_index=index,full_test_images=it,
            native_anomalous_images=ip,native_normal_images=it-ip,
            complete_native_pixels=pt,native_positive_pixels=pp,metrics=metrics))
    if sum(x['full_test_images'] for x in rows)!=1725:
        raise ValueError('All original native categories required')
    means={k:float(np.mean([row['metrics'][k] for row in rows])) for k in rows[0]['metrics']}
    total=1725;positive=sum(row['native_anomalous_images'] for row in rows)
    paper=case['paper_reported']
    report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        queue_configuration=args.queue,queue_sha256=sha(ROOT/args.queue),
        dataset_configuration=job['dataset_configuration'],dataset_configuration_sha256=sha(ROOT/job['dataset_configuration']),
        complete_original_test_images=total,complete_original_categories=15,per_category=rows,
        category_macro_mean=means,category_macro_percent={k:100*v for k,v in means.items()},
        pooled_image_anomaly_fraction=positive/total,pooled_all_positive_image_F1_max=2*positive/(total+positive),
        paper_TCCM_comparison={k:dict(paper_percent=v,constant_diagnostic_percent=100*means[k],
            difference_percentage_points=v-100*means[k]) for k,v in paper.items()},
        mathematical_score='Exact contraction v(y,1)=-y gives Eq20 zero at every pixel of every normal/anomalous image.',
        independent_verification='Every native label/mask count; sklearn exact weighted tied-score population curves agree with closed-form AUROC/AP/F1.',
        AUPRO_boundary='Constant-score AUPRO=FPRlimit/2 only under registered exact-tie linear interpolation; target-author implementation unknown.',
        counts_and_full_data_hashes_verified=True,trained_model_forward_calls=0,
        native_benchmark_performance=False,measured_model_seed_slots_added=0,
        original_author_equivalence_certified=False,all_original_experiments_complete=False,
        boundary='Ground-truth and mathematical failure-mode diagnostic, not a trained TCCM result. Category macro and pooled values are different aggregation scopes. High oracle F1 alone is not proof of good ranking.')
    write_new(ROOT/args.output,report)
    print(dict(native_test_images=1725,categories=15,constant_category_macro_image_F1_percent=100*means['image_F1_max'],
        paper_TCCM_image_F1_percent=paper['image_F1_max'],pooled_constant_F1_percent=200*positive/(total+positive),trained_model_results=0))


if __name__=='__main__':
    main()
