from scripts.flow_matching.publish_algorithm_dataset_catalog import strict_metric_rows


def test_missing_metrics_are_blank_and_zero_is_measured():
    record=dict(algorithm_config='x',dataset='d',protocol='validation_only',completed_seeds=0,required_seeds=5,
                VUS_ROC_n=None)
    missing=strict_metric_rows([record],'source.csv','frozen-time')
    assert all(r['mean'] is None and r['n']==0 for r in missing)
    record.update(completed_seeds=2,f1_mean=0.,f1_std=0.,VUS_ROC_n=1,VUS_ROC_mean=.6)
    measured={r['metric']:r for r in strict_metric_rows([record],'source.csv','frozen-time')}
    assert measured['f1']['mean']==0. and measured['f1']['n']==2
    assert measured['VUS_ROC']['n']==1 and measured['VUS_ROC']['mean']==.6


def test_protocol_and_capture_identity_are_retained():
    records=[dict(algorithm_config='same',dataset='d',protocol=p,completed_seeds=1,required_seeds=5,f1_mean=.2)
             for p in ['validation_only','test_oracle']]
    rows=strict_metric_rows(records,'snapshot/source.csv','2026-10-09T02:24:49+00:00')
    assert {r['protocol'] for r in rows}=={'validation_only','test_oracle'}
    assert all(r['source']=='snapshot/source.csv' and r['captured_utc']=='2026-10-09T02:24:49+00:00'
               and r['author_equivalence_certified'] is False for r in rows)
