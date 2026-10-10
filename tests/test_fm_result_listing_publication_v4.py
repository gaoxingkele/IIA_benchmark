import pytest
from scripts.flow_matching.publish_result_listing_v4 import validate_rows, interactive_html


def row(**overrides):
    return dict(task='anomaly_detection',algorithm='model',dataset='PSM',protocol='strict',metric='f1',
                n=0,required_repeats=5,mean=None,std=None,se=None,ci95_low=None,ci95_high=None,**overrides)


def test_missing_not_zero_and_measured_zero_retained():
    validate_rows([row()])
    r=row();r.update(n=1,mean=0)
    validate_rows([r])
    r['n']=0
    with pytest.raises(ValueError,match='Missing'):
        validate_rows([r])


def test_protocols_distinct_but_duplicate_rejected():
    r=row();s=row();s['protocol']='oracle'
    validate_rows([r,s])
    with pytest.raises(ValueError,match='Duplicate'):
        validate_rows([r,r])


def test_single_seed_uncertainty_rejected():
    r=row();r.update(n=1,mean=.4,std=.1)
    with pytest.raises(ValueError,match='Single-seed'):
        validate_rows([r])


def test_html_uses_text_content_and_escapes_source_data():
    r=row();r['algorithm']='</script><script>alert(1)</script>'
    page=interactive_html([r])
    assert '\\u003c/script>' in page
    assert r['algorithm'] not in page
    assert '.textContent=' in page
