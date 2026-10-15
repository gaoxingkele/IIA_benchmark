from copy import deepcopy
import subprocess
import sys
import pytest
from scripts.flow_matching import run_common105_matched577_TAB_v1 as s


def test_all577native_models1726labels_and77490entity_views_original21metrics():
    c=s.registration();entries=s.verify(c);old=s.read(s.ROOT/s.original.CONFIG)
    assert len(entries)==577 and sum(len(e['case']['required_label_policies']) for e in entries)==1726
    assert len({e['key'] for e in entries})==577 and c['required_entity_label_views']==77490
    for k in ['settings','resource_lock','metric_python','environment','score_metrics','label_metrics',
        'strict_ratios','diagnostic_pooled_ratios','metric_calls_per_entity_label_view','entity_macro_bootstrap_resamples']:
        assert c[k]==old[k]
    assert {e['track'] for e in entries}=={'common105_matched577'}


@pytest.mark.parametrize('fault',['case','label','grid','memory','VUS','native','CI'])
def test_reduced_scope_changed_source_or_metric_definitions_rejected(fault):
    c=s.registration()
    if fault=='case':c['entries'].pop()
    elif fault=='label':c['entries'][0]['case']['required_label_policies'].pop()
    elif fault=='grid':c['diagnostic_pooled_ratios']=[1]
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='VUS':c['score_metrics'].remove('VUS_ROC')
    elif fault=='native':c['source_native_configuration']=s.original.CONFIG
    else:c['entity_macro_bootstrap_resamples']=2
    with pytest.raises(ValueError):s.verify(c)


def test_every_native_family_has_exact_new_guard_command_and_full_original_readiness_checker():
    c=s.registration()
    for family in s.native.SPECS:
        entry=next(e for e in c['entries'] if e['component']==family);module=s.component(entry)
        nc=s.read(s.ROOT/module.CONFIG)
        assert entry['case'] in nc['cases']
        assert module.ready_native.__code__ is s.native.component(family).ready_native.__code__
        assert module.ready_native.__globals__['CONFIG']==s.native.CONFIGS[family]
        assert module.worker_command(entry['case'],'review')==s.native.worker_command(family,entry['case'],'review')
        assert s.ready_case(c,entry) is None


def test_original_controller_and_summary_keep_numerics_and_exact_new_worker_command():
    c=s.registration();ns=s.namespace();fn=ns['controller']
    assert 577 in fn.__code__.co_consts and 1726 in fn.__code__.co_consts
    assert ns['summary_worker'].__code__ is s.original.summary_worker.__code__
    assert ns['checked_fragment'] is s.checked_fragment and ns['ready_case'] is s.ready_case
    command=s.worker_command(c,c['entries'][0],'e000')
    assert command[command.index('-m')+1]==s.MODULE and command[-1]=='e000'


def test_other_source_tracks_cannot_be_silently_rebound():
    c=s.registration();entry=deepcopy(c['entries'][0]);entry['track']='original_full873_v2'
    with pytest.raises(ValueError):s.ready_case(c,entry)


def test_cold_registration_does_not_load_metric_or_model_libraries():
    code='from scripts.flow_matching import run_common105_matched577_TAB_v1 as s;import sys;s.verify(s.registration());assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
