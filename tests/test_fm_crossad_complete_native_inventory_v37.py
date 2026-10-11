import copy
import json

import pytest

from scripts.flow_matching.publish_crossad_complete_native_inventory_v37 import ROOT,complete_cohorts,metric_catalog


def test_all_original139jobs_and_six_seeds_must_remain_registered():
    queue=json.loads((ROOT/'configs/experiments/fm_crossad_complete_queue.v7.json').read_text(encoding='utf-8'))
    assert len(complete_cohorts(queue))==29
    for variant in ['lost_job','duplicate_seed','lost_ratio']:
        bad=copy.deepcopy(queue)
        if variant=='lost_job':bad['jobs'].pop()
        elif variant=='duplicate_seed':
            group=[j for j in bad['jobs'] if j['author_recipe']=='fresh_train' and j['dataset']=='MSL']
            group[-1]['seed']=group[0]['seed']
        else:bad['settings']['evaluation']['strict_false_alarm_percent']=[1]
        with pytest.raises(ValueError):complete_cohorts(bad)


def test_native_oracle_range_and_all_validation_operating_points_are_separate():
    catalog=metric_catalog('fresh_train')
    assert len(catalog)==29
    assert len({(p,m) for p,m,_,_ in catalog})==29
    assert sum(g=='oracle' for _,_,g,_ in catalog)==4
    assert sum(g.startswith('validation/') for _,_,g,_ in catalog)==15
    for ratio in ['0.1','1','5']:
        assert sum(g=='validation/'+ratio for _,_,g,_ in catalog)==5
    assert any(m=='R_AUC_ROC' for _,m,_,_ in catalog)
    assert any(m=='Affiliation_F1' for _,m,_,_ in catalog)
