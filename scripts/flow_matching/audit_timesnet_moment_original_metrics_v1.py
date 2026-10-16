"""Bind original anomaly tables to architecture, data and evaluation protocols.

Read-only with respect to PDFs, author sources, models and scientific runs.
The output is literature/protocol evidence; it cannot certify a benchmark result.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'configs/reproducibility/fm_timesnet_moment_original_metric_audit.2026-10-16_v1.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def page(doc, number: int) -> str:
    return doc[number - 1].get_text()


def timesnet_table15(text: str, models: list[str]) -> list[dict]:
    body = text.split('Table 15:', 1)[1]
    matches = re.findall(
        r'\((?:1997|2017|2019|2020|2021a?|2022a?|2023|Inception|ResNeXt)\)\s+'
        r'((?:\d+\.\d{2}\s+){15}\d+\.\d{2})(?!\d)', body)
    if len(matches) != len(models):
        raise ValueError(f'Table 15 row count {len(matches)} != {len(models)}')
    rows = []
    for model, values in zip(models, matches):
        tokens = values.split()
        for j, dataset in enumerate(['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']):
            for k, metric in enumerate(['precision', 'recall', 'F1']):
                printed = tokens[3*j+k]
                rows.append(dict(model=model, dataset=dataset, metric=metric,
                                 printed_value=printed, unit='percent',
                                 fraction=float(printed)/100, table=15, pdf_page=22,
                                 evidence_kind='author_reported_only'))
        rows.append(dict(model=model, dataset='five_dataset_macro_average', metric='F1',
                         printed_value=tokens[-1], unit='percent',
                         fraction=float(tokens[-1])/100, table=15, pdf_page=22,
                         evidence_kind='author_reported_only'))
    return rows


def moment_table7(text: str) -> list[dict]:
    body = text.split('Metric\nMOMENT0', 1)[1].split('Table 7.', 1)[0]
    models = ['MOMENT0', 'MOMENTLP', 'GPT4TS', 'TimesNet',
              'Anomaly Transformer', 'DGHL', 'k-NN']
    rows = []
    blocks = body.split('VUS ROC')
    if len(blocks) != 2:
        raise ValueError('Missing distinct MOMENT metric blocks')
    for metric, block in zip(['adjusted_best_F1', 'VUS_ROC'], blocks):
        for stat, marker in [('mean', 'Mean'), ('median', 'Median'), ('std', 'Std.')]:
            match = re.search(re.escape(marker) + r'\s+((?:\d+\.\d{3}\s+){6}\d+\.\d{3})', block)
            if match is None:
                raise ValueError(f'Missing {metric}/{stat}')
            for model, value in zip(models, match.group(1).split()):
                rows.append(dict(model=model, dataset='UCR_Anomaly_Archive_Table7_aggregate',
                                 metric=metric, statistic=stat, printed_value=value,
                                 value=float(value), unit='fraction', table=7, pdf_page=9,
                                 evidence_kind='author_reported_only',
                                 uncertainty_scope='across_time_series_not_training_seed_uncertainty',
                                 aggregation_population_identity_resolved=False))
    return rows


def moment_table22(text: str) -> list[dict]:
    body = text.split('Dataset name\n', 1)[1].split('Table 22.', 1)[0]
    lines = [x.strip() for x in body.splitlines() if x.strip()]
    if len(lines) % 13:
        raise ValueError('Table 22 cannot be partitioned into name plus 12 printed cells')
    models = ['Anomaly Transformer', 'MOMENT0', 'MOMENTLP', 'DGHL', 'GPT4TS', 'TimesNet']
    rows = []
    for i in range(0, len(lines), 13):
        label, tokens = lines[i], lines[i+1:i+13]
        if any(not re.fullmatch(r'\d+\.\d{3}|NaN', token) for token in tokens):
            raise ValueError(f'Unexpected original cell in {label}: {tokens}')
        for j, token in enumerate(tokens):
            rows.append(dict(dataset_printed_label=label, model=models[j % 6],
                             metric=['adjusted_best_F1', 'VUS_ROC'][j // 6],
                             printed_value=token, value=None if token == 'NaN' else float(token),
                             paper_missing_value=token == 'NaN', unit='fraction',
                             table=22, pdf_page=27, evidence_kind='author_reported_only',
                             original_file_identity_resolved=False))
    return rows


def source_slice(path: Path, start: int, end: int) -> dict:
    lines = path.read_text(encoding='utf-8').splitlines()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                first_line=start, last_line=end, text='\n'.join(lines[start-1:end]))


def compare_eligibility(local: dict, reference: dict) -> dict:
    """Refuse a numerical reproduction verdict if any protocol identity is unknown.

    Matching dataset names or scalar values cannot establish comparability.
    This gate concerns identity, not the scientific desirability of a protocol.
    """
    fields = ('architecture', 'dataset_identity', 'split_preprocessing', 'training_budget',
              'threshold_partition', 'label_adjustment', 'aggregation_population',
              'metric_implementation', 'uncertainty_scope')
    unresolved, mismatched = [], []
    for key in fields:
        a, b = local.get(key), reference.get(key)
        if a is None or b is None or a == 'unknown' or b == 'unknown':
            unresolved.append(key)
        elif a != b:
            mismatched.append(key)
    return dict(protocol_identity_complete=not unresolved and not mismatched,
                mismatched=mismatched, unresolved=unresolved,
                author_equivalence_certified=False,
                numerical_agreement_verdict_allowed=not unresolved and not mismatched)


def run(config_path: str = CONFIG) -> dict:
    import fitz
    cpath = ROOT / config_path
    c = json.loads(cpath.read_text(encoding='utf-8'))
    for binding in c['immutable_inputs']:
        if sha(ROOT / binding['path']) != binding['sha256']:
            raise ValueError('Changed immutable audit input: ' + binding['path'])
    docs = {k: fitz.open(ROOT / v) for k, v in c['papers'].items()}
    t = timesnet_table15(page(docs['timesnet_primary'], 22), c['timesnet_table15_models'])
    alternate = timesnet_table15(page(docs['timesnet_alternate'], 22), c['timesnet_table15_models'])
    assert t == alternate
    m = moment_table7(page(docs['moment'], 9))
    m22 = moment_table22(page(docs['moment'], 27))
    primary_table5 = page(docs['timesnet_primary'], 9).split('4.6', 1)[0]
    alternate_table5 = page(docs['timesnet_alternate'], 9).split('4.6', 1)[0]
    assert normalized(primary_table5) == normalized(alternate_table5)
    assert re.search(r'84\.72', primary_table5)
    assert next(r for r in t if r['model'] == 'Stationary' and r['dataset'] == 'SMD'
                and r['metric'] == 'F1')['printed_value'] == '84.62'
    # Recheck F1 cells independently against Table 5's dataset-major columns.
    for dataset in ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']:
        match = re.search(r'\n' + dataset + r'\s+((?:\d+\.\d{2}\s+){13}\d+\.\d{2})', primary_table5)
        assert match is not None, dataset
        values = match.group(1).split()
        for model, token in zip(['TimesNet (ResNeXt)', 'TimesNet (Inception)'], values[:2]):
            assert next(r for r in t if r['model'] == model and r['dataset'] == dataset and r['metric'] == 'F1')['printed_value'] == token
    # All required pages have rendered/visual review evidence in the publication packet.
    for key, entries in c['page_anchors'].items():
        for number, anchors in entries.items():
            text = normalized(page(docs[key], int(number)))
            for anchor in anchors:
                assert normalized(anchor) in text, (key, number, anchor)
    model = json.loads((ROOT / c['local_timesnet_model']).read_text(encoding='utf-8'))
    author_model = (ROOT / c['timesnet_author_root'] / 'models/TimesNet.py').read_text(encoding='utf-8')
    ast.parse(author_model)
    assert 'Inception_Block_V1' in author_model and 'ResNeXt' not in author_model
    exp = ROOT / c['timesnet_author_root'] / 'exp/exp_anomaly_detection.py'
    exp_text = exp.read_text(encoding='utf-8')
    assert 'np.concatenate([train_energy, test_energy], axis=0)' in exp_text
    assert 'np.percentile(combined_energy, 100 - self.args.anomaly_ratio)' in exp_text
    assert 'gt, pred = adjustment(gt, pred)' in exp_text
    moment_metrics = ROOT / c['moment_author_root'] / 'moment/utils/anomaly_detection_metrics.py'
    metric_text = moment_metrics.read_text(encoding='utf-8')
    assert 'n_splits: int = 100' in metric_text and 'np.max(adjusted_f1)' in metric_text
    assert 'label=(y_true > 0)' in metric_text
    assert 'y_pred = y_scores >= threshold' in metric_text
    release_settings = {}
    for dataset in ['SMD', 'MSL', 'SMAP', 'SWAT', 'PSM']:
        path = ROOT / c['timesnet_author_root'] / 'scripts/anomaly_detection' / dataset / 'TimesNet.sh'
        contents = path.read_text(encoding='utf-8')
        release_settings[dataset] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
            epochs=[int(x) for x in re.findall(r'--train_epochs\s+(\d+)', contents)],
            d_model=[int(x) for x in re.findall(r'--d_model\s+(\d+)', contents)],
            e_layers=[int(x) for x in re.findall(r'--e_layers\s+(\d+)', contents)],
            top_k=[int(x) for x in re.findall(r'--top_k\s+(\d+)', contents)])
    local = dict(architecture='Inception', dataset_identity='local_registered_SMAP51',
                 split_preprocessing='local_constant_fit_v2', training_budget='release_3_epochs',
                 threshold_partition='validation_only_99th_percentile', label_adjustment=False,
                 aggregation_population='registered_51_entities', metric_implementation='raw_F1',
                 uncertainty_scope='five_training_seeds_planned')
    reference = dict(architecture='Inception', dataset_identity='paper_SMAP_Table6',
                     split_preprocessing=None, training_budget='paper_Table7_10_epochs',
                     threshold_partition=None, label_adjustment=None,
                     aggregation_population=None, metric_implementation=None,
                     uncertainty_scope='paper_three_repeats_definition_incomplete')
    moment_claim = dict(claim='MOMENT SMD F1 0.832', verified_in_reviewed_original_anomaly_tables=False,
        eligible_as_original_paper_target=False, third_party_source_identity=None,
        reason='Original Tables7/22 evaluate UCR; SMD in Table12 describes pretraining corpus, not this downstream score.')
    result = dict(schema_version=1, captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=config_path, configuration_sha256=sha(cpath),
        status='original_tables_and_protocol_differences_audited_not_reproduced',
        evidence_kind='literature_and_source_protocol_audit',
        immutable_inputs=c['immutable_inputs'],
        timesnet=dict(table15_rows=t, two_pdf_table15_transcriptions_identical=True,
            two_pdf_table5_text_normalization_identical=True, f1_cells_table5_table15_agree=True,
            paper_repeats=3, paper_Table7_anomaly_settings=c['timesnet_paper_settings'],
            paper_Table6_datasets=c['timesnet_paper_datasets'],
            pinned_release_architecture='Inception', pinned_release_settings=release_settings,
            local_model_configuration=model,
            paper_training_schedule_matches_release_for_all_datasets=False,
            source_protocol=dict(threshold_fit_includes_test_scores=True,
                threshold_optimized_using_test_labels=False, point_adjustment_uses_test_labels=True,
                represents_pinned_release_not_proof_of_historical_paper_execution=True),
            strict_local_vs_paper_inception=compare_eligibility(local, reference),
            local_identity=local, paper_identity=reference,
            ResNeXt_original_paper_architecture_certified=False,
            disputed_user_claims=[
                dict(claim='PSM F1 0.974', original_exact_value=0.9747,
                     architecture='ResNeXt', source='Tables5/15',
                     note='Keep exact printed 97.47 percent; do not substitute for Inception 95.21 percent.'),
                dict(claim='SWaT F1 0.924', original_exact_value=None,
                     original_Inception_F1=0.9210, original_ResNeXt_F1=0.9174,
                     verified_in_reviewed_original_tables=False),
                dict(claim='TimesNet best on SWaT', supported_by_original_Table5=False,
                     original_table_highest_model='LightTS', original_table_highest_F1=0.9333)],
            original_cross_table_discrepancy=dict(model='Stationary', dataset='SMD',
                metric='F1', Table5_printed_percent='84.72', Table15_printed_percent='84.62',
                discrepancy_preserved_not_silently_corrected=True),
            footnote='Anomaly Transformer comparison uses reconstruction error alone, not its joint criterion.'),
        moment=dict(table7_summary_rows=m, table22_visible_cells=m22,
            visible_table22_row_count=len({r['dataset_printed_label'] for r in m22}),
            visible_table22_missing_cell_count=sum(r['paper_missing_value'] for r in m22),
            table7_caption_series_count=248, section4_1_series_count=44,
            table22_caption_series_count=45, sectionF3_series_count=46,
            aggregation_population_identity_resolved=False,
            table7_std_is_training_seed_uncertainty=False,
            anomaly_downstream_data='UCR Anomaly Archive',
            footnote17=dict(pdf_page=29,
                original_joint_Anomaly_Transformer_anomaly_criterion_used=False,
                author_acknowledged_potential_disadvantage_for_Anomaly_Transformer=True),
            SMD_NASA_Table12_role='Time Series Pile pretraining corpus; not downstream Table7/22 targets',
            legacy_ARA_original_datasets_list_is_not_anomaly_task_certificate=True,
            experimental_settings=dict(window=512, anomaly_score='mean_squared_reconstruction_error',
                downsample_factor_for_series_longer_than_2560=10,
                downsample_rule_scope='paper_text; exact file identity/version unresolved'),
            pinned_release_metric=dict(threshold_grid_points=100,
                test_score_min_max_linear_grid=True, prediction_operator='>=',
                test_labels_used_to_select_maximum_F1=True,
                test_labels_used_for_point_adjustment=True, strict_TAB_result=False,
                exact_all_ties_test_label_oracle_equivalent=False),
            disputed_user_claim=moment_claim),
        source_excerpts=[source_slice(exp, 155, 189),
            source_slice(ROOT / c['timesnet_author_root'] / 'utils/tools.py',95,117),
            source_slice(moment_metrics,33,51)],
        new_training_updates=0, new_complete_benchmark_conditions=0,
        numerical_worker_or_configuration_changed=False,
        strict_leaderboard_results_added=0, all_original_experiments_complete=False)
    for doc in docs.values():
        doc.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.config)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=str(output), timesnet_cells=len(result['timesnet']['table15_rows']),
        moment_summary_cells=len(result['moment']['table7_summary_rows']),
        moment_visible_rows=result['moment']['visible_table22_row_count'],
        new_complete_benchmark_conditions=0)))


if __name__ == '__main__':
    main()
