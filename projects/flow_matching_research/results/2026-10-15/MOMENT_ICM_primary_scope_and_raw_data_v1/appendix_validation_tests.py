from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from scripts.flow_matching import audit_moment_icm_primary_scope_v1 as s

PAPER='papers/literature/flow_matching/user_supplied/2026-10-07/moment_long_context/3e67405b691183dab729d6f2e56421280d8bb1268761f190468d8bce9e326413.pdf'


def expected(content):
    return dict(status='verified',size_bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
        git_blob_sha1=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest())


def ts_content():
    return b'@problemName fixture\n@timestamps false\n@univariate false\n@classLabel true A B\n@data\n1,2,3:4,5,6:A\n7,8:9,10:B\n'


def test_all630_cells_cover198_forecast_and234_classification_settings():
    rows=s.targets(s.ROOT/PAPER)
    assert dict(Counter(r['table'] for r in rows))=={4:108,5:288,6:52,7:182}
    cases={(r['table'],r['model'],r['dataset'],r.get('horizon')) for r in rows}
    assert len(cases)==432
    assert len({r['dataset'] for r in rows if r['table']==6})==26
    assert len({r['dataset'] for r in rows if r['table']==7})==91


@pytest.mark.parametrize('table,model,dataset,horizon,metric,value',[
    (4,s.MODELS4[0],'Exchange',96,'MAE',.229),
    (4,s.MODELS4[1],'Weather',384,'MSE',.268),
    (5,'NBeats','Exchange',384,'MSE',.933),
    (5,'Infini-Channel Mixer + Static MT (Ours)','Weather',96,'MSE',.148),
    (6,'MOMENT-Tiny + ICM','BasicMotions',None,'accuracy',.875),
    (7,'MOMENT-Tiny','GestureMidAirD2',None,'accuracy',.538),
    (7,'MOMENT-Tiny + ICM','ECG200',None,'accuracy',.840)])
def test_visual_table_anchor_coordinates_are_not_swapped(table,model,dataset,horizon,metric,value):
    rows=s.targets(s.ROOT/PAPER)
    matches=[r for r in rows if (r['table'],r['model'],r['dataset'],r.get('horizon'),r['metric'])==(table,model,dataset,horizon,metric)]
    assert len(matches)==1 and matches[0]['reported']==value


def test_forecast_parser_rejects_changed_horizon_or_non_numeric_table_cell():
    block='model\n'+'\n'.join(str(h)+'\n'+'\n'.join(['0.100']*12) for h in [96,192,384])
    assert len(s.forecasting_rows(block,['model'],5))==36
    with pytest.raises(ValueError):s.forecasting_rows(block.replace('192','193'),['model'],5)
    with pytest.raises(ValueError):s.forecasting_rows(block.replace('0.100','missing',1),['model'],5)


def test_streaming_raw_ts_audit_binds_all_bytes_rows_channels_and_labels(tmp_path):
    content=ts_content();p=tmp_path/'fixture.ts';p.write_bytes(content)
    report=s.file_audit(p,expected(content),'ts')
    assert report['status']=='verified' and report['observed_instances']==2
    assert report['observed_dimensions']=={2:2} and report['observed_class_counts']==dict(A=1,B=1)
    assert not report['series_float_values_parsed'] and not report['raw_preprocessing_or_split_leakage_certified']
    assert p.read_bytes()==content


def test_raw_modification_missing_file_and_undeclared_label_are_not_verified(tmp_path):
    content=ts_content();p=tmp_path/'fixture.ts';p.write_bytes(content+b'1:2:A\n')
    assert s.file_audit(p,expected(content),'ts')['status']=='audit_failed'
    assert s.file_audit(tmp_path/'missing.ts',expected(content),'ts')['status']=='missing'
    changed=content.replace(b':B\n',b':C\n');p.write_bytes(changed)
    assert s.file_audit(p,expected(changed),'ts')['status']=='audit_failed'


def test_csv_audit_counts_full_rows_without_normalizing_or_resplitting(tmp_path):
    content=b'date,a,b\n2020-01-01,1,2\n2020-01-02,3,4';p=tmp_path/'fixture.csv';p.write_bytes(content)
    report=s.file_audit(p,expected(content),'csv')
    assert report['status']=='verified' and report['observed_instances']==2
    assert report['CSV_columns']==['date','a','b'] and p.read_bytes()==content


def test_complete240_file_execution_fixture_and630_reference_export(tmp_path,monkeypatch):
    rows=s.targets(s.ROOT/PAPER);monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'verify',lambda _:rows)
    config=dict(pile_integrity='integrity.json',paper='paper.pdf',classification_root='classification',forecast_root='forecast',
        forecast_files={d:d+'.csv' for d in s.DATASETS},output_directory='outputs')
    (tmp_path/'paper.pdf').write_bytes(b'fixture binding only; real primary PDF tested separately')
    records=[]
    for dataset in dict.fromkeys(r['dataset'] for r in rows if r['table'] in [6,7]):
        for split in ['TRAIN','TEST']:
            relative=f'classification/{dataset}/{dataset}_{split}.ts';p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(ts_content());records.append(dict(path=relative,**expected(ts_content())))
    for dataset,name in config['forecast_files'].items():
        relative='forecast/'+name;p=tmp_path/relative;p.parent.mkdir(exist_ok=True)
        content=b'date,value\n2020,1\n2021,2\n';p.write_bytes(content);records.append(dict(path=relative,**expected(content)))
    (tmp_path/'integrity.json').write_text(json.dumps(dict(records=records)))
    report=s.execute(config)
    assert report['file_status_counts']=={'verified':240} and report['reported_scalar_targets']==630
    assert report['new_fit_or_inference']==0 and not report['original_pretraining_and_forecasting_and_SVM_experiments_run']
    assert not report['original_paper_raw_data_version_certified'] and not report['raw_files_overwritten']
    assert len((tmp_path/'outputs/all630_paper_reported_metrics.csv').read_text().splitlines())==631
    with pytest.raises(FileExistsError):s.execute(config)


def test_frozen_source_tampering_is_rejected_before_parsing(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path);p=tmp_path/'source';p.write_bytes(b'old')
    config=dict(source_receipts=[dict(path='source',sha256=s.sha(p))]);p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen source changed'):s.verify(config)
