from copy import deepcopy
from collections import Counter
import subprocess
import sys
import pytest
from scripts.flow_matching import audit_common105_matched577_labels_v1 as s


def test_all577_actual_common105_sources1726labels_and_original_scalar_resources():
    c=s.registration();cases=s.verify(c);old=s.read(s.ROOT/s.original.CONFIG)
    assert len(cases)==577 and sum(len(x['required_label_policies']) for x in cases)==1726
    assert Counter(x['dataset'] for x in cases)==dict(smap=286,msl=291)
    assert Counter(x['family'] for x in cases)==s.source.COUNTS
    assert c['settings']==old['settings'] and c['resource_lock']==old['resource_lock']
    assert c['source_execution']==s.source.CONFIG and c['required_view_counts']==dict(smap=1144,msl=582)
    binding=s.source.ready_data(s.read(s.ROOT/s.source.CONFIG))
    for case in cases:
        q,j=s.original.resolve_case(case)
        assert j['common105_original_full_source_id'].endswith('__original_full109_v2')
        assert j['dataset_manifest_path']==binding['manifest']


@pytest.mark.parametrize('fault',['case','label','memory','lock','source','threshold'])
def test_omitted_case_or_legacy_source_or_weakened_gate_rejected(fault):
    c=s.registration()
    if fault=='case':c['cases'].pop()
    elif fault=='label':c['cases'][0]['required_label_policies'].pop()
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='lock':c['resource_lock']='other.lock'
    elif fault=='source':c['source_execution']=s.original.source.CONFIG
    else:c['all_original_validation_quantiles_bootstrap_seeds_and_repeats_retained']=False
    with pytest.raises(ValueError):s.verify(c)


def test_exact_original_metric_worker_body_reused_and_source_not_native_reinference():
    fn=s.bound_controller()
    assert fn.__globals__['CONFIG']==s.CONFIG and fn.__globals__['MODULE']==s.MODULE
    assert fn.__globals__['source_binding'] is s.source_binding
    assert 577 in fn.__code__.co_consts and 1726 in fn.__code__.co_consts
    assert 873 not in fn.__code__.co_consts and 2037 not in fn.__code__.co_consts


def test_unfinished_producers_cannot_be_promoted_to_completed_scalar_results():
    c=s.registration()
    for case in c['cases']:
        if s.original.source_status(case) is None:
            with pytest.raises(ValueError,match='completed source'):s.source_binding(c,case)
            break
    else:pytest.skip('All actual producers already completed')


def test_both_actual_producer_interpreters_verify_same_cold_registration():
    c=s.registration();code='from scripts.flow_matching import audit_common105_matched577_labels_v1 as s;import sys;s.verify(s.registration());assert "numpy" not in sys.modules and "torch" not in sys.modules'
    for exe in sorted({case['worker_python'] for case in c['cases']}):
        p=subprocess.run([str(s.ROOT/exe),'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
        assert p.returncode==0,p.stdout+p.stderr
