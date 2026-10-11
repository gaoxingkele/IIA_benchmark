"""Publish all five original trained generators, keeping evaluator repetitions separate."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import math
import re
import statistics

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha
from scripts.flow_matching.audit_cfm_full_arrays_v1 import independent_stats
from scripts.flow_matching.audit_spectral_completed_generation_v1 import METRICS, evaluator_statistics
from scripts.flow_matching.publish_brits_two_seed_listing_v27 import legacy
from scripts.flow_matching.publish_pi_original_source_listing_v42 import interactive_table
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

SEEDS = [42, 43, 44, 45, 46]


def validate_native(proof, result, receipt, job):
    from scripts.flow_matching.run_spectral_original_job import fingerprint
    wanted = dict(passed=True, canonical_id=job['id'], seed=job['seed'],
        original_full_registered_generation_jobs=70, actual_full_native_model_count=1,
        planned_original_training_seed_count=5, actual_model_parameters=294390,
        paper_reported_parameters=291318, original_optimizer_updates=10000,
        original_gradient_accumulation=2, original_training_minibatches=20000,
        complete_reference_shape=[3662, 24, 6], complete_generated_shape=[4002, 24, 6], evaluated_generated_rows=3662,
        actual_full_original_checkpoint_tensors_restored_bitwise=True,
        complete_original_reference_array_bitwise_verified=True, all_original_generated_arrays_finite=True,
        all_four_five_repeat_evaluator_statistics_independently_recomputed=True,
        learned_evaluators_independently_retrained=False, generative_model_reinferred_by_this_audit=False,
        additional_training_seed_slots=0, author_equivalence_certified=False, strict_TAB_result=False,
        cuda_context_initialized=False)
    if (job['seed'] not in SEEDS or job['id'] != f"spectral_original__diffusion_ts__stocks__released_author_data__seed{job['seed']}"
            or any(proof.get(k) != v for k, v in wanted.items())
            or result['job'] != job or result['job_sha256'] != fingerprint(job)
            or result['status'] != 'completed' or result['benchmark_smoke']
            or receipt['id'] != job['id'] or receipt['exitcode'] != 0
            or receipt['resource_guard_aborted'] or not receipt['full_result_validated']
            or proof['artifacts'] != result['artifacts']):
        raise ValueError('Every actual full original generator and successful independent native artifact audit is required')
    for metric, item in proof['metrics'].items():
        if (item['raw_repeats'] != result['metrics'][metric]['raw_repeats']
                or item['evaluator_repeats'] != 5 or item['independent_training_models'] != 1
                or item['planned_original_training_models'] != 5
                or item['between_training_seed_STD'] is not None or item['between_training_seed_CI95'] is not None):
            raise ValueError('Five evaluator repetitions must never become five trained generators')
    evaluator_statistics(proof['metrics'])
    evaluator_statistics(result['metrics'])


def training_statistics(proofs):
    if len(proofs) != 5 or [p['seed'] for p in proofs] != SEEDS or len({p['canonical_id'] for p in proofs}) != 5:
        raise ValueError('All five unique original seeds42..46 are required')
    rows = []
    for metric in METRICS:
        values = [statistics.mean(p['metrics'][metric]['raw_repeats']) for p in proofs]
        summary = independent_stats(values)
        if summary['n'] != 5 or not math.isclose(summary['std'], statistics.stdev(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Training-seed statistics must use five generator means, never twenty-five evaluator runs')
        rows.append(dict(metric=metric, original_training_seed_values=values, **summary))
    return rows


def is_cohort(row):
    return (row['task'] == 'unconditional_time_series_generation' and row['algorithm'] == 'Spectral-author/diffusion_ts'
        and row['dataset'] == 'stocks' and row['protocol'] == 'released_author_data' and row['metric'] in METRICS)


def check_delta(before, after):
    if len(before) != len(after): raise ValueError('Preserve every original algorithm-data metric slot')
    permitted = {'n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high', 'source', 'captured_utc', 'comparison',
        'uncertainty_scope', 'evaluator_std', 'evaluator_se', 'evaluator_ci95'}
    changed = []
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new: continue
        if (not is_cohort(old) or not is_cohort(new) or str(old['n']) != '4' or str(new['n']) != '5'
                or str(old['required_repeats']) != '5' or str(new['required_repeats']) != '5'
                or any(old.get(k) != new.get(k) for k in old.keys() | new.keys() if k not in permitted)):
            raise ValueError('Only four existing Diffusion-TS Stocks rows may advance from4/5to5/5 training seeds')
        changed.append(index)
    if len(changed) != 4 or {after[i]['metric'] for i in changed} != set(METRICS):
        raise ValueError('Exactly all four unchanged original generation metrics are required')
    return changed


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); return reader.fieldnames, list(reader)


def audit_records(config):
    from scripts.flow_matching.run_spectral_original_job import verify
    queue = read(ROOT / config['original_queue']); verify(queue)
    jobs = {j['id']: j for j in queue['jobs']}
    terminal = read(ROOT / config['new_native_audit_receipt'])
    previous = read(ROOT / config['previous_native_audit_receipt'])
    for receipt, seeds in [(terminal, [46]), (previous, [43, 44, 45])]:
        if (not receipt['passed'] or receipt['full_native_artifact_audits_completed'] != len(seeds)
                or receipt['audited_new_original_seeds'] != seeds
                or receipt['original_required_seeds'] != SEEDS or len(receipt['records']) != len(seeds)):
            raise ValueError('Every previous and final actual native artifact worker must complete')
    combined = previous['records'] + terminal['records']
    if len({r['id'] for r in combined}) != 4:
        raise ValueError('All four non-initial independent artifact audits must be unique')
    native_records = {r['id']: r for r in combined}
    proofs = []; records = []
    for entry in config['native_records']:
        proof = read(ROOT / entry['proof']); result = read(ROOT / entry['result']); receipt = read(ROOT / entry['receipt'])
        job = jobs[proof['canonical_id']]; validate_native(proof, result, receipt, job)
        if sha(ROOT / entry['result']) != proof['original_result_sha256'] or sha(ROOT / entry['receipt']) != proof['actual_terminal_receipt_sha256']:
            raise ValueError('Independent complete artifact proof must bind the actual terminal native result')
        audit_config = read(ROOT / entry['audit_configuration'])
        for pin in audit_config['source_receipts']:
            if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Full native frozen source/artifact changed')
        validation = read((ROOT / entry['proof']).parent / 'validation.json')
        if not validation['passed'] or validation['outputs']['independent_full_native_artifact_audit.json'] != sha(ROOT / entry['proof']):
            raise ValueError('Complete native independent audit output hash differs')
        for artifact in result['artifacts']:
            if sha(ROOT / artifact['path']) != artifact['sha256']: raise ValueError('Full original checkpoint/generated/reference bytes changed')
        if job['seed'] != 42:
            native = native_records[job['id']]
            if (native['native_exit_code'] != 0 or native['resource_guard_aborted']
                    or not native['full_native_artifact_audit_passed']
                    or native['proof_sha256'] != sha(ROOT / entry['proof'])
                    or native['configuration_sha256'] != sha(ROOT / entry['audit_configuration'])):
                raise ValueError('Native independent worker must actually finish every complete artifact review')
        proofs.append(proof); records.append(dict(seed=job['seed'], id=job['id'], **entry,
            proof_sha256=sha(ROOT / entry['proof']), result_sha256=sha(ROOT / entry['result']),
            terminal_receipt_sha256=sha(ROOT / entry['receipt']), metrics=proof['metrics']))
    training_statistics(proofs)
    return proofs, records


def verify_published(config, before, target, proofs, summary):
    headers, rows = read_csv(target / 'all_algorithm_dataset_metrics.csv')
    positions = check_delta(before, rows)
    if len(headers) != 31 or len(rows) != 8626 or sum(bool(r['mean']) for r in rows) != 2128:
        raise ValueError('Retain the entire8626slot/31field/2128measured inventory')
    from scipy.stats import t
    for item in summary:
        row = next(r for r in rows if is_cohort(r) and r['metric'] == item['metric'])
        values = [statistics.mean(p['metrics'][item['metric']]['raw_repeats']) for p in proofs]
        mean = statistics.mean(values); std = statistics.stdev(values); se = std / math.sqrt(5)
        if not all(math.isclose(float(row[k]), v, rel_tol=1e-12, abs_tol=1e-12) for k, v in [('mean', mean), ('std', std), ('se', se)]):
            raise ValueError('Independent five-training-seed mean/STD/SE disagrees')
        for field, probability in [('ci95_low', .025), ('ci95_high', .975)]:
            if not math.isclose(float(t.cdf((float(row[field]) - mean) / se, 4)), probability, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Unclipped complete five-seed interval must use Student t with df4')
        if row['n'] != '5' or row['required_repeats'] != '5' or any(row[k] for k in ['evaluator_std', 'evaluator_se', 'evaluator_ci95']):
            raise ValueError('Within-model evaluator uncertainty belongs in per-seed proofs, separately from training-seed uncertainty')
    exclusive_json(target / 'independent_publication_review.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, independently_recomputed_mean_STD_SE_and_Student_t_CDF=True, degrees_of_freedom=4,
        original_training_seeds=SEEDS, required_training_seeds=[42, 43, 44, 45, 46],
        exactly_four_original_rows_updated=True, other8622rows_and_capture_times_unchanged=True,
        learned_evaluators_independently_retrained=False, all_original_experiments_complete=False,
        author_equivalence_certified=False, strict_TAB_result=False))
    return positions


def publish(path):
    config = read(ROOT / path)
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']: raise ValueError('Frozen five-seed publication binding changed')
    proofs, records = audit_records(config); summary = training_statistics(proofs)
    base = ROOT / config['base_publication']; validation = read(base / 'publication_validation.json')
    for name, digest in validation['outputs'].items():
        if sha(base / name) != digest: raise ValueError('Preserve every existing published base artifact')
    headers, before = read_csv(base / 'all_algorithm_dataset_metrics.csv')
    if len(headers) != 31 or len(before) != 8626 or sum(bool(r['mean']) for r in before) != 2128:
        raise ValueError('Entire original metric scope must stay intact')
    captured = datetime.now(timezone.utc).isoformat(); rows = deepcopy(before)
    for item in summary:
        position = [i for i, row in enumerate(rows) if is_cohort(row) and row['metric'] == item['metric']]
        if len(position) != 1: raise ValueError('One existing canonical metric slot is required')
        row = rows[position[0]]
        if not math.isclose(float(row['mean']), statistics.mean(item['original_training_seed_values'][:4]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Retain the exact previously published four-generator mean')
        row.update({k: str(item[k]) for k in ['n', 'mean', 'std', 'se', 'ci95_low', 'ci95_high']})
        row.update(captured_utc=captured, source=config['cohort_summary_path'],
            evaluator_std='', evaluator_se='', evaluator_ci95='',
            comparison='five_of_five_full_native_generators_complete_artifacts_audited_author_alignment_unverified',
            uncertainty_scope='All five original trained generators42..46 of5; primary sampleSTD/SE/unclippedStudent-t95CI across five generator evaluator-means. Each model has5full evaluator repeats; per-model raw/STD/SE/CI preserved separately in source summary, never twenty-five independent training seeds.')
    positions = check_delta(before, rows)
    target = ROOT / config['output_directory']; target.mkdir(exist_ok=False, parents=True)
    exclusive_json(ROOT / config['cohort_summary_path'], dict(captured_utc=captured,
        original_training_seeds=SEEDS, required_training_seeds=[42, 43, 44, 45, 46],
        full_native_trained_generators_completed=5, total_original_optimizer_updates=50000,
        total_original_training_minibatches=100000, total_full_evaluator_runs=100,
        metric_statistics=summary, independent_complete_native_artifact_reviews=records,
        original_scope=config['original_scope'], generative_models_reinferred_by_audits=False,
        learned_evaluators_independently_retrained=False, additional_training_seed_slots=0,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
    with (target / 'all_algorithm_dataset_metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers); writer.writeheader(); writer.writerows(rows)
    boundary = ('Only four existing Diffusion-TS/Stocks/released-author-data metric rows advance from4/5to5/5original trained generators. '
        'All other8622rows and capture times stay unchanged. Each generator completed10000updates/20000minibatches, '
        'full3662reference/4002generated windows and four original evaluators repeated5times each. Complete native '
        'model/EMA/optimizer tensors and arrays were independently inspected; evaluator statistics were recomputed. '
        'Primary uncertainty summarizes5generator means with sampleSTD/SE/unclippedStudent-t95CI(df4); per-generator '
        'evaluation uncertainty stays in the source summary. All five registered generator seeds are complete. No learned evaluator retraining '
        'or independent generative-model reinference is claimed. Full-source scaling/distribution reconstruction, '
        'runtime TF graph seeds0..4 and294390actualvs291318paper parameters remain explicit. No future holdout, '
        'original-author or TAB equivalence is certified. All45papers/681methods/273original-data tasks remain in scope.')
    (target / 'all_metrics.md').write_text('# Algorithm-dataset metrics v48\n\n' + boundary + '\n\n'
        + legacy.table(headers, [[r.get(k) or '-' for k in headers] for r in rows]) + '\n', encoding='utf-8')
    page = interactive_table(rows, headers).replace('v42', 'v48')
    page = re.sub(r'<p>[^<]*</p>(?=<input)', '<p>Diffusion-TS Stocks:5/5original training seeds. '
        'Between-generator uncertainty and per-generator evaluator repetitions remain separate;author/TAB alignment uncertified.</p>', page)
    (target / 'results.html').write_text(page, encoding='utf-8')
    (target / 'README.rst').write_text('Algorithm-dataset metrics v48\n============================\n\n' + boundary
        + '\n\n8626metric rows;2128measured;6498pending. No metric slot is added or dropped.\n', encoding='utf-8')
    verify_published(config, before, target, proofs, summary)
    exclusive_json(target / 'publication_validation.json', dict(published_utc=captured, base_publication=config['base_publication'],
        aggregate_metric_rows=8626, fields=31, measured_metric_rows=2128, missing_metric_rows=6498,
        changed_metric_rows=4, changed_positions=positions, other8622rows_and_capture_times_unchanged=True,
        newly_completed_original_training_seeds=[46], completed_original_training_seeds=SEEDS,
        required_original_training_seeds=[42, 43, 44, 45, 46], additional_training_seed_slots=0,
        original_scope=config['original_scope'], learned_evaluators_independently_retrained=False,
        all_original_experiments_complete=False, author_equivalence_certified=False, strict_TAB_result=False,
        source_receipts=config['source_receipts'], outputs={p.name: sha(p) for p in target.iterdir() if p.is_file()}))
    print(dict(metric_rows=8626, changed_metrics=4, original_training_seeds=5, required_original_seeds=5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    publish(parser.parse_args().config)
