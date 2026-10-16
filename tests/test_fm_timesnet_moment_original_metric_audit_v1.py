"""Cross-extractor checks against the actual original tables and protocol gates."""
from pathlib import Path
import re

import pytest
from pypdf import PdfReader

from scripts.flow_matching import audit_timesnet_moment_original_metrics_v1 as a


@pytest.fixture(scope='module')
def report():
    return a.run()


def test_all272_timesnet_cells_match_independent_pypdf_extractor(report):
    doc = PdfReader(a.ROOT / 'papers/literature/mtsad_pdfs/wu2023_timesnet.pdf')
    text = doc.pages[21].extract_text().split('Table 15:', 1)[1]
    # Different PDF engine and row parser: pypdf returns one model per line.
    lines = [line for line in text.splitlines()
             if re.search(r'\((?:\d{4}[a-z]?|Inception|ResNeXt)\)', line)]
    values = [re.findall(r'\d+\.\d{2}', line) for line in lines]
    assert len(values) == 17 and all(len(row) == 16 for row in values)
    assert [value for row in values for value in row] == [
        row['printed_value'] for row in report['timesnet']['table15_rows']]


@pytest.mark.parametrize('variant,expected', [
    ('Inception', ['85.12', '84.18', '70.85', '92.10', '95.21']),
    ('ResNeXt', ['85.81', '85.15', '71.52', '91.74', '97.47'])])
def test_variant_specific_f1_not_interchangeable(report, variant, expected):
    rows = [r for r in report['timesnet']['table15_rows']
            if r['model'] == f'TimesNet ({variant})' and r['metric'] == 'F1'
            and r['dataset'] != 'five_dataset_macro_average']
    assert [r['printed_value'] for r in rows] == expected
    assert all(abs(r['fraction']*100-float(r['printed_value'])) < 1e-12 for r in rows)


def test_moment_all42_summary_cells_independent_pypdf(report):
    text = PdfReader(a.ROOT / 'papers/literature/mtsad_pdfs/goswami2024_moment.pdf').pages[8].extract_text()
    text = text.split('Adj. F1', 1)[1].split('Table 7.', 1)[0]
    values = []
    for line in text.splitlines():
        if re.match(r'^(Mean|Median|Std\.)\s+\d', line):
            values += re.findall(r'\d+\.\d{3}', line)
    assert len(values) == 42
    assert values == [r['printed_value'] for r in report['moment']['table7_summary_rows']]


def test_moment_all504_visible_cells_independent_pypdf_and_original_nan(report):
    text = PdfReader(a.ROOT / 'papers/literature/mtsad_pdfs/goswami2024_moment.pdf').pages[26].extract_text()
    body = text.split('Dataset name', 1)[1].split('Table 22.', 1)[0]
    rows = report['moment']['table22_visible_cells']
    names = list(dict.fromkeys(r['dataset_printed_label'] for r in rows))
    assert len(names) == 42
    starts = [body.index(name) for name in names] + [len(body)]
    independently_read = []
    for i, name in enumerate(names):
        chunk = body[starts[i]+len(name):starts[i+1]]
        values = re.findall(r'\d+\.\d{3}|NaN', chunk)
        assert len(values) == 12, (name, values)
        independently_read += values
    assert independently_read == [r['printed_value'] for r in rows]
    missing = [r for r in rows if r['paper_missing_value']]
    assert len(missing) == 10 and all(r['value'] is None for r in missing)
    assert report['moment']['aggregation_population_identity_resolved'] is False
    assert report['moment']['table7_std_is_training_seed_uncertainty'] is False


def identity():
    return dict(architecture='Inception', dataset_identity='full_exact_data',
        split_preprocessing='frozen_split', training_budget='full_original',
        threshold_partition='validation_only', label_adjustment=False,
        aggregation_population='all_groups', metric_implementation='F1_v1',
        uncertainty_scope='five_training_seeds')


@pytest.mark.parametrize('field', list(identity()))
def test_each_mismatched_protocol_dimension_prevents_numerical_verdict(field):
    left, right = identity(), identity()
    right[field] = 'different'
    result = a.compare_eligibility(left, right)
    assert result['mismatched'] == [field]
    assert not result['numerical_agreement_verdict_allowed']


@pytest.mark.parametrize('field', list(identity()))
def test_each_unknown_protocol_dimension_prevents_numerical_verdict(field):
    left, right = identity(), identity()
    right[field] = None
    result = a.compare_eligibility(left, right)
    assert result['unresolved'] == [field]
    assert not result['numerical_agreement_verdict_allowed']


def test_matching_identity_alone_does_not_certify_author_equivalence():
    result = a.compare_eligibility(identity(), identity())
    assert result['protocol_identity_complete']
    assert not result['author_equivalence_certified']


def test_actual_local_release_and_paper_are_distinct_no_automatic_SMD_target(report):
    t, m = report['timesnet'], report['moment']
    assert not t['strict_local_vs_paper_inception']['numerical_agreement_verdict_allowed']
    assert t['pinned_release_settings']['SMAP']['epochs'] == [3]
    assert t['paper_Table7_anomaly_settings']['epochs'] == 10
    assert t['pinned_release_architecture'] == 'Inception'
    assert not t['ResNeXt_original_paper_architecture_certified']
    assert t['source_protocol']['threshold_fit_includes_test_scores']
    assert not t['source_protocol']['threshold_optimized_using_test_labels']
    assert t['source_protocol']['point_adjustment_uses_test_labels']
    assert m['pinned_release_metric']['test_labels_used_to_select_maximum_F1']
    assert not m['pinned_release_metric']['exact_all_ties_test_label_oracle_equivalent']
    assert not m['disputed_user_claim']['eligible_as_original_paper_target']
    assert report['new_complete_benchmark_conditions'] == 0
    assert not report['numerical_worker_or_configuration_changed']


def test_truncated_tables_fail_instead_of_returning_partial_reference():
    with pytest.raises(ValueError):
        a.timesnet_table15('Table 15: TimesNet (Inception) 87.76 82.63', ['TimesNet (Inception)'])
    with pytest.raises(ValueError):
        a.moment_table22('Dataset name\nbroken\n0.500\nTable 22.')
