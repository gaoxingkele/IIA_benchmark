"""Cross-check mathematical diagnostic against earlier whole-test region audit."""
import argparse
from datetime import datetime,timezone

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_v1 import write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    diagnostic=read(ROOT/args.diagnostic)
    original='projects/flow_matching_research/results/2026-10-11/flow_mismatching_full_image_evaluation_registration_v1/complete_original_test_region_readiness_audit.json'
    # This independent audit was completed before this diagnostic was proposed.
    previous=read(ROOT/original)
    native=next(row for row in previous['records'] if row['dataset']=='MVTec-AD')
    if (diagnostic['complete_original_test_images']!=native['complete_original_test_images']
            or diagnostic['complete_original_categories']!=native['complete_original_categories']
            or len(diagnostic['per_category'])!=15
            or diagnostic['native_benchmark_performance'] or diagnostic['trained_model_forward_calls']!=0):
        raise ValueError('Complete native diagnostic scope and no trained-model performance required')
    all_values=[]
    for actual,expected in zip(diagnostic['per_category'],native['per_category']):
        pairs={'category':'category','category_index':'category_index','full_test_images':'full_test_images',
            'native_anomalous_images':'original_anomalous_image_labels','native_normal_images':'original_normal_image_labels',
            'native_positive_pixels':'complete_resized128_anomalous_pixels'}
        if any(actual[a]!=expected[b] for a,b in pairs.items()):
            raise ValueError('Earlier full native count audit differs')
        if actual['complete_native_pixels']!=expected['complete_resized128_anomalous_pixels']+expected['complete_resized128_background_pixels']:
            raise ValueError('Every full native positive/background pixel required')
        values={}
        for prefix,p,n in [('image_',expected['original_anomalous_image_labels'],expected['full_test_images']),
                ('pixel_',expected['complete_resized128_anomalous_pixels'],actual['complete_native_pixels'])]:
            values.update({prefix+'AUROC':.5,prefix+'AP':p/n,prefix+'F1_max':2*p/(n+p)})
        values['pixel_AUPRO']=.15
        if set(actual['metrics'])!=set(values) or any(abs(actual['metrics'][k]-v)>1e-12 for k,v in values.items()):
            raise ValueError('Complete native constant-score analytical metrics differ')
        all_values.append(values)
    means={k:sum(row[k] for row in all_values)/15 for k in all_values[0]}
    if any(abs(means[k]-diagnostic['category_macro_mean'][k])>1e-12 for k in means):
        raise ValueError('Unweighted15category macro diagnostic differs')
    positives=sum(x['original_anomalous_image_labels'] for x in native['per_category'])
    if abs(diagnostic['pooled_all_positive_image_F1_max']-2*positives/(1725+positives))>1e-12:
        raise ValueError('Pooled whole-native F1 differs')
    protocol=read(ROOT/'configs/experiments/fm_flow_mismatching_original_protocol.v1.json')
    paper=next(row['paper_reported'] for row in protocol['Table2_same_backbone_FM_comparison'] if row['method']=='TCCM')
    for k,v in paper.items():
        item=diagnostic['paper_TCCM_comparison'][k]
        if (item['paper_percent']!=v or abs(item['constant_diagnostic_percent']-means[k]*100)>1e-10
                or abs(item['difference_percentage_points']-(v-means[k]*100))>1e-10):
            raise ValueError('Original printed TCCM and same-scope diagnostic comparison differs')
    write_new(ROOT/args.output,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic=dict(path=args.diagnostic,sha256=sha(ROOT/args.diagnostic)),
        preexisting_independent_native_count_audit=dict(path=original,sha256=sha(ROOT/original)),
        all1725_original_test_image_counts_and28262400_pixel_counts_equal=True,
        all15_native_categories_and_all7_analytical_metrics_verified=True,
        macro_and_pooled_scopes_kept_separate=True,native_benchmark_performance=False,
        new_measured_seed_slots=0,all_original_experiments_complete=False))
    print(dict(independent_prior_counts_and_all7_metrics_verified=True,model_performance=False))


if __name__=='__main__':
    main()
