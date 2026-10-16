import io
import zipfile
import numpy as np
import pytest
from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as s


def npy(value):
    stream=io.BytesIO();np.save(stream,value,allow_pickle=False);stream.seek(0);return stream


def test_full_native_array_comparison_rejects_any_tail_value_shape_dtype_or_trailing_change():
    a=np.arange(16387,dtype=np.float32)
    assert s.complete_array(npy(a),npy(a),len(a))['all_score_payload_bytes_identical']
    for b in [a[:-1],a.astype(np.float64),np.append(a[:-1],np.float32(-1))]:
        with pytest.raises(ValueError):s.complete_array(npy(a),npy(b),len(a))
    b=npy(a);b.seek(0,2);b.write(b'bad');b.seek(0)
    with pytest.raises(ValueError):s.complete_array(npy(a),b,len(a))


def test_every_label_and_source_row_is_checked_including_last_row():
    def archive(rows):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as f:
            f.writestr('labels.npy',npy(np.array([0,1,0,1],dtype=np.uint8)).getvalue())
            f.writestr('source_rows.npy',npy(np.array(rows,dtype=np.int64)).getvalue())
        stream.seek(0);return zipfile.ZipFile(stream)
    with archive([10,11,12,13]) as f:assert s.labels_and_rows(f,4,10)['positive_labels']==2
    with archive([10,11,12,15]) as f,pytest.raises(ValueError):s.labels_and_rows(f,4,10)


def record(seed,value):
    return dict(seed=seed,raw_sha256='fixed',native_points=dict(train=10,validation=4,test=4),
        label_and_source_row_review=dict(labels_payload_sha256='same',source_rows_payload_sha256='same'),
        native_validation_control=dict(strict={'1':dict(micro=dict(f1=value,precision=value,recall=value,auroc=value,average_precision=value),macro_f1=value)}))


def test_partial_seed_cohorts_never_produce_means_and_complete_cohorts_preserve_paired_seeds():
    jobs=[];verified={}
    for row in [None,1]:
        for index,seed in enumerate(s.SEEDS):
            identity=str(row)+'_'+str(seed)
            jobs.append(dict(dataset='D',mode='fresh_train',seed=seed,reconstructed_ablation=row is not None,
                ablation_row=row,canonical_CPU_condition_reference=identity))
            verified[identity]=record(seed,.5+index*.01+(0 if row is None else .03))
    missing=verified.pop('1_1107');summary=s.cohorts(jobs,verified)
    assert summary['complete_primary_cohorts']==1
    assert 'metrics' not in summary['primary_cohorts']['D/paper_reconstruction_row1']
    assert not summary['paired_seed_ablation_differences']
    verified['1_1107']=missing;summary=s.cohorts(jobs,verified)
    assert summary['complete_primary_cohorts']==2
    delta=summary['paired_seed_ablation_differences']['D/paper_reconstruction_row1_minus_main']
    assert delta['validation_false_alarm_percent/1/micro/f1']['mean']==pytest.approx(.03)
    verified['1_1107']['label_and_source_row_review']['labels_payload_sha256']='changed'
    with pytest.raises(ValueError):s.cohorts(jobs,verified)


def test_full_metrics_uncertainty_and_exact_five_seed_interval_required():
    a=dict(strict={'1':dict(f1=.5,ci=[.3,.7])})
    s.same_metrics(a,a,dict(rtol=2e-6,atol=2e-6))
    with pytest.raises(ValueError):s.same_metrics(a,dict(strict={'1':dict(f1=.5,ci=[.3,.8])}),dict(rtol=2e-6,atol=2e-6))
    with pytest.raises(ValueError):s.interval([.1,.2,.3,.4])
    result=s.interval([1.,2.,3.,4.,5.]);assert result['mean']==3.
    assert result['seed_sample_SD']==pytest.approx(1.5811388300841898)
