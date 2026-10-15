from copy import deepcopy
import subprocess
import sys
import pytest
from scripts.flow_matching import verify_moment_masked_official_weights_v1 as s


def test_all_sizes_native_observation_protocol_original_weights_and_guards():
    c=s.registration();s.verify(c)
    assert {m['size'] for m in c['models']}=={'small','base','large'} and len(c['lengths'])==10
    original=s.read(s.ROOT/s.POLICY)
    assert c['settings']==original['settings'] and c['resource_lock']==original['resource_lock']
    for item in c['models']:
        new=s.read(s.ROOT/item['model_config'])
        old=s.read(s.ROOT/('configs/models/fm_moment_pretrained/fm_moment_pretrained_'+item['size']+'__release512.json'))
        assert {k:v for k,v in new['parameters'].items() if k!='observation_protocol'}==old['parameters']
        for key in ['citation','checkpoint_citation','source_commit','source_repository']:
            assert new[key]==old[key]


@pytest.mark.parametrize('fault',['size','length','memory','context','channels','fill','threads'])
def test_reduced_probe_or_changed_original_resource_policy_rejected(fault):
    c=s.registration()
    if fault=='size':c['models'].pop()
    elif fault=='length':c['lengths'].pop()
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=1
    elif fault=='context':c['context']=100
    elif fault=='channels':c['channels']=1
    elif fault=='fill':c['fill_values']=[0.]
    else:c['CPU_threads']=1
    with pytest.raises(ValueError):s.verify(c)


def test_guarded_diagnostic_registers_without_loading_weights_or_numpy():
    code='from scripts.flow_matching import verify_moment_masked_official_weights_v1 as s;import sys;s.verify(s.registration());assert "torch" not in sys.modules and "numpy" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
