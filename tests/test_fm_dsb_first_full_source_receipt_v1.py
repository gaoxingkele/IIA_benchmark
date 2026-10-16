from copy import deepcopy
import math
import subprocess
import sys
import pytest
from scripts.flow_matching import audit_dsb_first_full_source_receipt_v1 as s


def fixture():
    counts=dict(f=0,b=0);phases=[]
    for phase in range(1,21):
        for role in ['b','f']:
            counts[role]+=10001;phases.append(dict(phase=phase,direction=role,actual_phase_updates=10001,cumulative_updates=dict(counts)))
    return dict(phases=phases,actual_updates=400040,required_updates=400040,required_phases=40)


def test_complete_actual_source_pass_and_counter_contract():s.verify_phases(fixture())


@pytest.mark.parametrize('fault',['missing_phase','phase_order','updates','cumulative','total','required'])
def test_no_partial_or_reordered_training_can_be_called_complete(fault):
    x=fixture()
    if fault=='missing_phase':x['phases'].pop()
    elif fault=='phase_order':x['phases'][0],x['phases'][1]=x['phases'][1],x['phases'][0]
    elif fault=='updates':x['phases'][0]['actual_phase_updates']=10000
    elif fault=='cumulative':x['phases'][-1]['cumulative_updates']['f']-=1
    elif fault=='total':x['actual_updates']-=1
    else:x['required_phases']=39
    with pytest.raises(ValueError):s.verify_phases(x)


def test_even_population_uses_original_torch_lower_median_not_averaged_middle():
    mean,median=s.moments((2,5),[0.]*5+[100.]*5)
    assert mean==[50.]*5 and median==[0.]*5
    values=s.independent_moments((2,5),[0.]*5+[100.]*5,(3,5),[0.]*15)
    assert values['Mean_MSE']==2500 and values['Mean_L2']==50 and values['Median_MSE']==0


def test_full_population_moment_distances_recompute_without_cell_pairing():
    a=[1.,2.,3.,4.,5.]*3;b=[0.,0.,0.,0.,0.]*2
    r=s.independent_moments((3,5),a,(2,5),b)
    assert r==dict(Mean_MSE=11.,Mean_L2=math.sqrt(11),Mean_L1=3.,Median_MSE=11.,Median_L2=math.sqrt(11),Median_L1=3.)


@pytest.mark.parametrize('shape,data',[((0,5),[]),((2,4),[0.]*8),((2,5),[0.]*5)])
def test_never_shrink_population_or_coordinates(shape,data):
    with pytest.raises(ValueError):s.moments(shape,data)


def test_cold_scope_and_original_budget_provenance_without_neural_imports():
    code=('import sys;from scripts.flow_matching import audit_dsb_first_full_source_receipt_v1 as s;'
        'c=s.registration();s.verify(c);assert c["original_DSB_models_required"]==70;'
        'assert c["settings"]["minimum_commit_headroom_bytes"]==22*2**30;'
        'assert not c["full_native_replay_complete"] and not c["independent_full_W1_complete"];'
        'assert not any(n in sys.modules for n in ["torch","numpy","scipy","pandas"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
