"""Explain original Pi PA inflation from every saved native anomaly event."""
import argparse
import csv
from datetime import datetime, timezone
import json

from scripts.flow_matching.audit_pi_original_recovery_completion_v1 import ROOT,read,digest,interval_pa


def main():
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(args.config)
    for item in config['source_receipts']:
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Original diagnostic source changed')
    proof=read(config['independent_completion'])
    if not proof['passed'] or proof['strict_TAB_result'] or proof['result_sha256']!=digest(proof['result_path']):
        raise ValueError('Full original source audit required; this is not strict TAB')
    result=read(proof['result_path'])
    score=next(a for a in result['artifacts'] if a['path'].endswith('/author_scores.npz'))
    if digest(score['path'])!=score['sha256']:
        raise ValueError('Keep exact original full saved score arrays')
    with np.load(ROOT/score['path'],allow_pickle=False) as data:
        labels=data['labels'];raw=data['raw_predictions'];pa=data['pa_predictions']
    if not np.array_equal(interval_pa(labels,raw),pa):
        raise ValueError('Original exact PA must independently agree')
    edges=np.diff(np.r_[0,labels,0]);events=[]
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        hits=np.flatnonzero(raw[start:end]);length=int(end-start)
        events.append(dict(event_id=len(events),start_point=int(start),end_point_exclusive=int(end),length_points=length,
            raw_true_positive_points=int(raw[start:end].sum()),author_PA_true_positive_points=int(pa[start:end].sum()),
            detected_by_original_threshold=bool(len(hits)),first_raw_detection_offset=int(hits[0]) if len(hits) else '',
            first_raw_detection_fraction=float(hits[0]/length) if len(hits) else '',
            additional_true_positive_points_from_PA=int(pa[start:end].sum()-raw[start:end].sum())))
    counts=proof['confusion_no_PA'];adjusted=proof['confusion_author_PA'];metrics=proof['independently_recomputed_metrics']
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),original_id=proof['original_id'],seed=proof['seed'],
        source_receipts=config['source_receipts'],original_score_sha256=score['sha256'],total_scored_points=len(labels),
        original_dropped_tail_points=proof['dropped_test_tail_points'],anomaly_points=int(labels.sum()),
        anomaly_prevalence=float(labels.mean()),anomaly_events=len(events),
        events_with_at_least_one_raw_hit=sum(e['detected_by_original_threshold'] for e in events),
        event_length_min=min(e['length_points'] for e in events),event_length_median=float(np.median([e['length_points'] for e in events])),
        event_length_max=max(e['length_points'] for e in events),raw_confusion=counts,author_PA_confusion=adjusted,
        additional_true_positive_points_from_PA=adjusted['tp']-counts['tp'],
        true_positive_point_expansion_factor=adjusted['tp']/counts['tp'],
        pointwise_f1=metrics['pointwise_no_PA_same_author_threshold']['f1'],author_PA_f1=metrics['f1'],
        F1_increase_percentage_points=100*(metrics['f1']-metrics['pointwise_no_PA_same_author_threshold']['f1']),
        strict_TAB_result=False,test_used_for_early_stopping=True,test_scores_used_for_calibration=True,
        author_equivalence_certified=False,independent_full_model_reinference_passed=False,
        independent_training_seeds_in_this_diagnostic=1,uncertainty_boundary='Single original training seed and one pooled PSM series; event rows are descriptive, not independent training replicates or group-held-out generalization.',
        boundary='Original same-threshold predictions compared before and after test-label PA. Every original covered anomaly event included. PA adds true-positive labels throughout hit events, not new observed detections. Tail41 retained as explicitly uncovered original points. No claim about unseen methods, all seeds, strict TAB, or original journal equivalence.')
    target=ROOT/config['output_directory'];target.mkdir(parents=True,exist_ok=False)
    (target/'all_original_event_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (target/'all_original_anomaly_events.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
    print(json.dumps(report))


if __name__=='__main__':main()
