import pytest
from scripts.flow_matching.audit_original_native_execution import extract_table


def test_primary_table_keeps_dataset_and_metric_column_identity():
    spec={'page':8,'table':'1','rows':['HBOS','COPOD'],'datasets':['SMAP','SMD']}
    text='Table 1: Results\nModel\nHBOS\n0.1000\n0.2000\n0.3000\n0.4000\n0.5000\n0.6000\nCOPOD\n0.7000\n0.8000\n0.9000\n0.1100\n0.2200\n0.3300\n'
    rows=extract_table(text,spec)
    assert len(rows)==12
    assert rows[5]['model']=='HBOS' and rows[5]['dataset']=='SMD' and rows[5]['metric']=='Best_F1'
    assert rows[5]['paper_value']==.6
    assert rows[9]['model']=='COPOD' and rows[9]['dataset']=='SMD' and rows[9]['metric']=='PRC'
    assert rows[9]['paper_value']==.11


@pytest.mark.parametrize('text',['Table 1: Results\nOther\n0.1000','Table 1: Results\nHBOS\n0.1000\nMissing\n0.3000'])
def test_missing_row_or_numeric_column_cannot_become_paper_score(text):
    with pytest.raises(ValueError):
        extract_table(text,{'page':8,'table':'1','rows':['HBOS'],'datasets':['SMAP']})
