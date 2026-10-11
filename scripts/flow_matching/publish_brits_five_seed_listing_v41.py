"""Advance only the three original BRITS metrics to five fully replayed seeds."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import shutil

from scripts.flow_matching.publish_brits_two_seed_listing_v27 import (
    ROOT, legacy, previous, validated_cohort, METRICS,
)
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats

SEEDS = [26, 27, 28, 29, 30]


def complete_delta(before, after):
    if len(before) != len(after):
        raise ValueError('Preserve all original registered slots')
    changed = []
    identity = ('algorithm', 'dataset', 'task', 'protocol', 'pattern', 'missing_ratio', 'metric')
    allowed = {'n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high', 'source', 'captured_utc',
               'comparison', 'author_equivalence_certified', 'strict_TAB_result'}
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if (any(k not in old or k not in new or old[k] != new[k] for k in identity)
                or (new['algorithm'], new['dataset'], new['task'], new['protocol'],
                    new['pattern'], new['missing_ratio']) !=
                   ('brits', 'electricity', 'imputation', 'author', 'paper_mask', '0.1')
                or new['metric'] not in METRICS
                or (str(old['n']), str(new['n']), str(old['required_repeats']),
                    str(new['required_repeats'])) != ('4', '5', '5', '5')
                or any(old.get(k) != new.get(k) for k in old.keys() | new.keys() if k not in allowed)):
            raise ValueError('Only the three original BRITS metrics may advance from4/5to5/5')
        changed.append(index)
    if len(changed) != 3 or {after[i]['metric'] for i in changed} != set(METRICS):
        raise ValueError('Every one of the three original metrics must advance')
    return changed


def publish(config_path):
    config = legacy.read(ROOT / config_path)
    for item in config['source_receipts']:
        if legacy.sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen publication source differs')
    if config['original_training_seeds'] != SEEDS:
        raise ValueError('Five unique original training seeds required')
    records = validated_cohort([legacy.read(ROOT / p) for p in config['full_model_proofs']], SEEDS)
    for path in config['full_model_guard_receipts']:
        guard = legacy.read(ROOT / path)
        if guard['exit_code'] != 0 or guard['resource_guard_aborted'] or not guard['full_native_model_replay_passed']:
            raise ValueError('All original full-model replay resource guards must pass')
    for record in records:
        for field in ('config', 'result', 'checkpoint', 'replay_arrays'):
            path = record[field] if field in ('config', 'replay_arrays') else record[field + '_path']
            if legacy.sha(ROOT / path) != record[field + '_sha256']:
                raise ValueError('Full native model or data artifact changed')
        completion = record['full_training_completion']
        if (completion['original_maximum_epochs'] != 10000 or completion['full_test_units'] != 290
                or completion['held_out_coordinates'] != 1020954
                or (completion['next_epoch'] < 10000 and completion['author_patience_remaining'] > 0)):
            raise ValueError('Full original training budget and native early stop required')
    base = ROOT / config['base_publication']
    old_receipt = previous.verified_capture(base, 'publication_validation.json')
    with (base / 'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); headers = reader.fieldnames; before = list(reader)
    rows = [dict(row) for row in before]
    captured = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if ((row['algorithm'], row['dataset'], row['task'], row['protocol'], row['pattern'], row['missing_ratio'])
                == ('brits', 'electricity', 'imputation', 'author', 'paper_mask', '0.1')
                and row['metric'] in METRICS):
            row.update(independent_stats([r['original_metrics'][row['metric']] for r in records]))
            row.update(source=config['cohort_summary_path'], captured_utc=captured,
                author_equivalence_certified=False, strict_TAB_result=False,
                comparison='full_native_models_audited_5_of_5_author_alignment_unverified')
    positions = complete_delta(before, rows)
    measured = sum(bool(row['mean']) for row in rows)
    if len(rows) != 8618 or len(headers) != 31 or measured != old_receipt['measured_metric_rows']:
        raise ValueError('All8618records and31fields and existing measured coverage required')
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve all prior publication attempts')
    target.mkdir(parents=True)
    summary_path = ROOT / config['cohort_summary_path']
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        raise FileExistsError('Preserve prior cohort summary')
    legacy.write(summary_path, dict(captured_utc=captured,
        original_training_seeds=SEEDS, required_training_seeds=5, original_five_seed_group_complete=True,
        source_proofs=config['full_model_proofs'], original_metric_values=[r['original_metrics'] for r in records],
        metric_rows=[rows[i] for i in positions], full_native_checkpoint_models_replayed=5,
        additional_training_seed_slots=0, author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False))
    with (target / 'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers); writer.writeheader(); writer.writerows(rows)
    excluded = {'all_algorithm_dataset_metrics.csv', 'all_metrics.md', 'results.html', 'README.rst', 'publication_validation.json'}
    for path in base.iterdir():
        if path.is_file() and path.name not in excluded:
            shutil.copyfile(path, target / path.name)
    boundary = ('Only the three original BRITS/Electricity author10% metrics advance from4/5to5/5 seeds. '
        'All other8,615 rows and their capture times are identical. Each of the five full6,993,328-parameter '
        'models has replayed all290 test windows and1,020,954 held-out coordinates with independent '
        'original HDF5 truth/mask/unit/month/error checks. Mean, sampleSTD, SE andStudent-t95% intervals '
        'use five unique original training seeds. Original author alignment and strict TAB equivalence '
        'remain uncertified. All45papers/681methods-baselines-ablations/273original-data-task obligations remain.')
    (target / 'all_metrics.md').write_text('# Complete algorithm-dataset metrics v41\n\n' + boundary + '\n\n'
        + legacy.table(headers, [[row.get(k) or '-' for k in headers] for row in rows]) + '\n', encoding='utf-8')
    (target / 'results.html').write_text(legacy.interactive_html(rows), encoding='utf-8')
    (target / 'README.rst').write_text('Complete algorithm-dataset metrics v41\n=====================================\n\n' + boundary
        + '\n\n8,618registered metric slots:2,120measured and6,498missing. Original seeds26/27/28/29/30 '
        'completed full training and checkpoint replay. The original10,000epoch cap andpatience30 were retained. '
        'The1,397versus paper1,400preprocessing-window discrepancy and historical mask/tuning/environment '
        'differences remain. This completes only the registered BRITS/Electricity10% five-seed cohort.\n', encoding='utf-8')
    # Stage-specific claims are fresh; earlier-stage evidence remains in the linked immutable receipt.
    stable = ('aggregate_metric_rows', 'task_rows', 'strict_groups', 'strict_complete_groups', 'strict_partial_groups',
              'cfm_completed_canonical_slots', 'cfm_complete_groups', 'grasp_complete_cohorts', 'algorithm_configuration_names')
    receipt = {k: old_receipt[k] for k in stable if k in old_receipt}
    receipt.update(published_utc=captured, base_publication=config['base_publication'],
        base_publication_utc=old_receipt['published_utc'], base_receipt_sha256=legacy.sha(base / 'publication_validation.json'),
        measured_metric_rows=measured, missing_metric_rows=len(rows)-measured,
        replaced_metric_rows=3, changed_numeric_metric_rows=3, changed_row_positions=positions,
        newly_measured_metric_rows=0, newly_completed_canonical_training_slots=1,
        brits_electricity_completed_training_seeds=5, brits_electricity_required_training_seeds=5,
        brits_full_native_models_replayed=5, brits_original_five_seed_group_complete=True, performance_rows_changed=True,
        other_track_rows_and_capture_times_preserved=True, other_tracks_recaptured=False,
        all_numeric_tables_byte_identical_to_base=False, base_numeric_csv_prefix_byte_identical=False,
        additional_training_seed_slots=0, original_scope=config['original_scope'],
        all_original_experiments_complete=False, author_equivalence_certified=False, strict_TAB_result=False,
        measured_task_rows=dict(Counter(row['task'] for row in rows if row['mean'])),
        checks=dict(all8618_original_slots_and31fields_retained=True,
            exactly_three_original_BRITS_metrics_4_to_5_of_5=True,
            all_other8615_rows_and_capture_times_exact=True, five_original_full_model_replays_succeeded=True),
        source_receipts=old_receipt['source_receipts'] + config['source_receipts'] + [
            dict(path=config_path, sha256=legacy.sha(ROOT / config_path)),
            dict(path=config['cohort_summary_path'], sha256=legacy.sha(ROOT / config['cohort_summary_path']))])
    receipt['outputs'] = {p.name: legacy.sha(p) for p in target.iterdir() if p.is_file()}
    legacy.write(target / 'publication_validation.json', receipt)
    print(json.dumps(dict(rows=len(rows), measured=measured, seeds=SEEDS, changed_metrics=3, original_five_seed_group_complete=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    publish(parser.parse_args().config)
