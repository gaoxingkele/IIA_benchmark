"""Full official TAB worker on synthetic inputs; no benchmark or real guards."""
import numpy as np
import pytest
from scripts.flow_matching import run_common105_matched577_TAB_v1 as s


def fixture(tmp_path, monkeypatch, dataset):
    c=s.read(s.ROOT/s.CONFIG)
    entry=next(e for e in c['entries'] if e['case']['dataset']==dataset)
    metrics=s.original.load_metrics(c)
    scores=np.linspace(0,1,40)
    labels=np.zeros(40,np.uint8);labels[10:12]=1;labels[30:32]=1
    source=tmp_path/'source';source.mkdir()
    np.savez_compressed(source/'scores.npz',e000_test=scores,e000_labels=labels)
    overlays={};arrays={}
    for index,policy in enumerate(entry['case']['required_label_policies']):
        actual=labels.copy()
        if index:actual[12:12+index]=1
        path=tmp_path/(policy+'.npz')
        np.savez_compressed(path,e000_labels=actual)
        overlays[policy]=dict(path=str(path),sha256=s.sha(path));arrays[policy]=actual
    result=dict(metrics=dict(strict={str(r):dict(calibration=dict(threshold=.5)) for r in c['strict_ratios']},
        tab_core_diagnostic=dict(ratio_grid=[dict(ratio_percent=r,threshold=.4) for r in c['diagnostic_pooled_ratios']])))
    data=dict(entities=[dict(entity_id='synthetic_only',array_prefix='e000',test_points=len(scores))],
        labels_only_controls=overlays,primary_label_policy=entry['case']['required_label_policies'][0])
    bound=dict(dataset=data,lineage=dict(fixture_only=True,native_scores=str(source/'scores.npz')),
        job=dict(output_directory='source'),result=result)
    original_config=s.ROOT/s.original.CONFIG
    # Only this isolated test replaces readiness. No production proof/guard is created.
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s.original,'ROOT',tmp_path)
    monkeypatch.setattr(s.durable,'ROOT',tmp_path)
    monkeypatch.setattr(s,'CONFIG','fixture.json')
    monkeypatch.setattr(s,'ready_case',lambda config,e:bound)
    monkeypatch.setattr(s.original,'load_metrics',lambda config:metrics)
    s.write(tmp_path/s.original.CONFIG,s.read(original_config))
    s.write(tmp_path/s.CONFIG,dict(fixture_only=True))
    c['artifact_root']=str(tmp_path/'artifacts');c['state_root']='state'
    return c,entry,scores,arrays,result,metrics


@pytest.mark.parametrize('dataset,views',[('smap',4),('msl',2)])
def test_new_worker_computes_every_actual_original_metric_and_matches_direct_values(tmp_path,monkeypatch,dataset,views):
    c,entry,scores,arrays,result,metrics=fixture(tmp_path,monkeypatch,dataset)
    s.worker(c,entry,'e000')
    attempt,output=s.original.paths(c,entry,'e000');proof=s.read(output/s.REPORT)
    assert proof['passed'] and proof['track']=='common105_matched577'
    assert proof['metric_calls']==views*201
    assert len(proof['durable_cell_journal']['cell_records'])==views*201
    assert proof['durable_cell_journal']['fresh_cells']==views*201
    assert proof['durable_cell_journal']['checkpoint_cells']==0
    assert proof['all_original_metric_calls_freshly_executed_this_attempt']
    assert not proof['neural_training_or_inference_performed']
    assert not proof['original_metric_definition_or_grid_changed']
    direct=s.original.evaluate_entity(c,scores,arrays,result,metrics)
    for policy,cells in direct.items():
        assert set(proof['metric_results'][policy])==set(cells)
        for name,cell in cells.items():
            actual=proof['metric_results'][policy][name]
            # Wall time is an observation of each separate execution, not a metric.
            assert actual['elapsed_seconds']>=0 and cell['elapsed_seconds']>=0
            assert {k:v for k,v in actual.items() if k!='elapsed_seconds'}=={
                k:v for k,v in cell.items() if k!='elapsed_seconds'}
    assert proof['original_component_sha256']==s.sha(output/s.COMPONENT)
    assert not attempt.exists()
    with pytest.raises(FileNotFoundError):s.checked_fragment(c,entry,'e000')


def test_missing_native_closure_cannot_start_worker(tmp_path,monkeypatch):
    c,entry,*_=fixture(tmp_path,monkeypatch,'msl')
    monkeypatch.setattr(s,'ready_case',lambda config,e:None)
    with pytest.raises(ValueError,match='closure'):s.worker(c,entry,'e000')
    assert not (tmp_path/'artifacts').exists()


def test_overlay_changed_after_binding_is_rejected_before_metric_computation(tmp_path,monkeypatch):
    c,entry,*_=fixture(tmp_path,monkeypatch,'msl')
    bound=s.ready_case(c,entry)
    overlay=next(iter(bound['dataset']['labels_only_controls'].values()))
    with open(overlay['path'],'ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='labels changed'):s.worker(c,entry,'e000')
    assert not list((tmp_path/'artifacts').rglob('*cell_bank'))
