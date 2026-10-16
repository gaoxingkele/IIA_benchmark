from copy import deepcopy
import csv
import pytest
from scripts.flow_matching import publish_SPOT_full25_condition_catalog_v61 as s


def inputs():
    config=s.read(s.ROOT/s.spot.CONFIG)
    cases=s.read(s.ROOT/s.SPOT_PACKET/'independent_full_array_audit.json')['cases']
    with (s.ROOT/s.BASE/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        fields=csv.DictReader(f).fieldnames
    return fields+[s.EXTRA_FIELD],config,cases


def test_every25_case_65_label_view_130_control_including_pending_and_undefined_is_present():
    fields,c,cases=inputs();rows=s.spot_rows(fields,c,cases,'frozen')
    assert len(rows)==130 and sum(r['n']=='1' for r in rows)==24
    assert sum(bool(r['mean']) for r in rows)==8
    assert sum(r['n']=='0' and not r['mean'] for r in rows)==106
    partial=[r for r in rows if r['n']=='1' and not r['mean']]
    assert len(partial)==16 and all(r['entity_finite']=='53' and r['entity_total']=='54' for r in partial)
    assert all(r['entity_calibration_undefined']=='1' and r['entity_nonfinite']=='0' for r in partial)
    assert all(not r[k] for r in rows for k in ['std','se','ci95_low','ci95_high','entity_bootstrap_ci95','paper_mean'])
    assert all(r['author_equivalence_certified']==r['strict_TAB_result']=='False' for r in rows)


@pytest.mark.parametrize('fault',['zero_imputation','subset_imputation','partial_coverage','seed_interval','missing_policy','dropped_case'])
def test_partial_macros_fabricated_uncertainty_and_scope_reduction_are_rejected(fault):
    fields,c,cases=inputs();c=deepcopy(c);cases=deepcopy(cases)
    row=cases[0]['full_denominator_macro_controls'][0]
    if fault=='zero_imputation':row['full_entity_macro_f1']=0.
    if fault=='subset_imputation':row['full_entity_macro_f1']=row['explicitly_defined_subset_macro_f1']
    if fault=='partial_coverage':row['defined_entities']+=1
    if fault=='seed_interval':row['training_seed_interval']=[0,1]
    if fault=='missing_policy':cases[0]['full_denominator_macro_controls'].pop()
    if fault=='dropped_case':c['entries'].pop()
    with pytest.raises(ValueError):s.spot_rows(fields,c,cases,'frozen')


def test_full54_matched_macro_keeps_source_epsilon_protocol_and_no_selection():
    fields,c,cases=inputs();rows=s.spot_rows(fields,c,cases,'frozen')
    row=next(r for r in rows if 'seed1103__common105_v1/source_first_exclusive/raw' in r['protocol'])
    assert row['mean']=='0.0911644348798328' and row['entity_finite']=='54' and row['entity_calibration_undefined']=='0'
    assert 'No PA' in row['selection_boundary'] and 'no TEST-label threshold' in row['selection_boundary']
    pa=next(r for r in rows if 'seed1103__common105_v1/source_first_exclusive/source_PA' in r['protocol'])
    assert pa['mean']=='0.2174513637567399' and 'PA uses TEST labels' in pa['selection_boundary']
