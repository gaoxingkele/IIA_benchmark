import copy
import csv
import json
import re

import pytest
from scripts.flow_matching import publish_native_condition_catalog_v55 as p


@pytest.fixture(scope='module')
def catalog():
    with (p.ROOT/'projects/flow_matching_research/results/2026-10-12/all_algorithm_dataset_metrics_v54/all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);before=list(reader);headers=reader.fieldnames
    jobs=[dict(canonical_CPU_condition_reference='test_case_'+str(i), reconstructed_ablation=False,
               mode='fresh_train',dataset='GECCO',seed=1103+i) for i in range(139)]
    cases={'generator_'+str(i):dict(job=dict(model='diffusion_ts',dataset='stocks',
            seed=42+i,data_track='released_author_data')) for i in range(70)}
    report=dict(independent_strict_metrics={ratio:dict(micro=dict(precision=.5,recall=.25,f1=1/3,
        auroc=.75,average_precision=.6)) for ratio in ['0.1','1','5']})
    generated=dict(metrics={metric:dict(mean=.2,std=.1,se=.1/(5**.5),ci95_low=-.1,ci95_high=.5)
        for metric in ['context_fid','correlational','discriminative','predictive']})
    config=dict(cross_independent_artifact_root='proofs',cross_queue='queue.json',
                spectral_artifact_root='generations',spectral_configuration='spectral.json')
    extra=p.cross_rows(headers,jobs,{'test_case_0':report},'frozen',config)+p.spectral_rows(
        headers,cases,{'generator_0':generated},'frozen',config)
    return before,headers,extra


def test_every_legacy_row_and_every_native_condition_metric_is_retained(catalog):
    before,headers,extra=catalog
    p.validate(before,before+extra,1,1)
    assert len(extra)==2365 and len(before+extra)==10991
    assert {r['metric'] for r in extra if r['task']=='anomaly_detection'}=={'precision','recall','F1','AUROC','AP'}
    assert len({r['protocol'] for r in extra if r['task']=='anomaly_detection'})==139*3
    assert len({r['protocol'] for r in extra if r['task']=='generation'})==70


def test_complete_checkpoints_do_not_manufacture_between_seed_uncertainty(catalog):
    before,headers,extra=catalog
    known=[r for r in extra if r['mean']]
    assert len(known)==19
    assert all(r['n']=='1' and r['required_repeats']=='1' for r in known)
    assert all(not r[k] for r in known for k in ['std','se','ci95_low','ci95_high','paper_mean'])
    assert all(r['strict_TAB_result']=='False' and r['author_equivalence_certified']=='False' for r in known)
    assert all('no point adjustment' in r['selection_boundary'] for r in known if r['task']=='anomaly_detection')


def test_evaluator_repeat_uncertainty_is_only_in_evaluator_columns(catalog):
    before,headers,extra=catalog
    rows=[r for r in extra if r['mean'] and r['task']=='generation']
    assert len(rows)==4 and all(r['evaluator_repeats']=='5' for r in rows)
    assert all(float(r['evaluator_std'])==.1 and json.loads(r['evaluator_ci95'])==[-.1,.5] for r in rows)
    assert all('zero new generator updates' in r['uncertainty_scope'] for r in rows)


def test_pending_cases_keep_blank_values_and_are_not_completed(catalog):
    before,headers,extra=catalog
    pending=[r for r in extra if not r['mean']]
    assert len(pending)==2365-19
    assert all(r['n']=='0' for r in pending)
    assert all(not r[k] for r in pending for k in ['std','se','ci95_low','ci95_high','evaluator_std','evaluator_se','evaluator_ci95'])


@pytest.mark.parametrize('field,value',[('std','0.01'),('paper_mean','0.99'),
    ('strict_TAB_result','True'),('author_equivalence_certified','True'),('n','5')])
def test_reject_false_uncertainty_paper_targets_or_equivalence(catalog,field,value):
    before,headers,extra=catalog
    changed=copy.deepcopy(extra);changed[0][field]=value
    with pytest.raises(ValueError):p.validate(before,before+changed,1,1)


def test_reject_missing_condition_slots_and_changes_to_any_legacy_row(catalog):
    before,headers,extra=catalog
    with pytest.raises(ValueError):p.validate(before,before+extra[:-1],1,1)
    changed=copy.deepcopy(before);changed[-1]['comparison']='silently changed'
    with pytest.raises(ValueError):p.validate(before,changed+extra,1,1)


def test_reject_duplicate_condition_tracks_and_nonfinite_metrics(catalog):
    before,headers,extra=catalog
    changed=copy.deepcopy(extra);changed[1]=copy.deepcopy(changed[0])
    with pytest.raises(ValueError):p.validate(before,before+changed,1,1)
    changed=copy.deepcopy(extra);changed[0]['mean']='nan'
    with pytest.raises(ValueError):p.validate(before,before+changed,1,1)


def test_html_payload_roundtrips_and_keeps_untrusted_cell_text_out_of_markup():
    rows=[dict(algorithm='</script><script>injected()</script>',mean='0')]
    page=p.interactive(rows,['algorithm','mean'])
    payload=re.search(r'const rows=(.*?),cols=',page).group(1)
    assert json.loads(payload)==rows
    assert '</script><script>injected()' not in page
    assert 'kept.slice(page*size,(page+1)*size)' in page
