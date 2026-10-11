"""Full paired calibration counterfactuals on a fully replayed original model.

This does not undo the original test-selected checkpoint or make a TAB result.
Every original temperature/stream combination is retained without selection.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import psutil

from scripts.flow_matching import run_pi_transformer_job as source
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def train_only_threshold(train, percentile):
    import numpy as np
    if train.ndim != 1 or len(train) == 0 or not np.isfinite(train).all() or not 0 <= percentile <= 100:
        raise ValueError('A full finite training-only reference and frozen percentile are required')
    return float(np.percentile(train, percentile))


def controls_ast(train_only):
    tree = ast.parse(Path(source.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score_controls')
    class Changes(ast.NodeTransformer):
        captures = 0
        thresholds = 0
        metadata = 0
        fidelity = 0
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'threshold' and train_only:
                    wanted = ast.parse('np.percentile(np.concatenate([train, test]), 100 - anomaly_ratio)', mode='eval').body
                    if ast.dump(node.value) != ast.dump(wanted): raise ValueError('Keep the exact original pooled percentile recipe')
                    node.value = ast.parse('train_only_threshold(train, 100 - anomaly_ratio)', mode='eval').body
                    self.thresholds += 1
                if node.targets[0].id == 'predicted':
                    self.captures += 1
                    return [node, ast.parse('capture_full_case(factor, t, stream, train, test, labels, predicted)').body[0]]
            return self.generic_visit(node)
        def visit_Dict(self, node):
            if train_only:
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == 'test_scores_used_for_calibration':
                        if not isinstance(value, ast.Constant) or value.value is not True: raise ValueError('Original pooled calibration marker changed')
                        value.value = False; self.metadata += 1
                    if isinstance(key, ast.Constant) and key.value == 'fidelity':
                        value.value = 'Full train-only threshold counterfactual; original test-selected weights remain; not strict TAB'
                        self.fidelity += 1
            return self.generic_visit(node)
    changes = Changes(); function = changes.visit(function)
    expected = (1, 1, 1, 1) if train_only else (1, 0, 0, 0)
    if (changes.captures, changes.thresholds, changes.metadata, changes.fidelity) != expected:
        raise ValueError('Only threshold calibration, evidence capture and truthful metadata may change')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def binding(config):
    from scripts.flow_matching.verify_pi_original_full_replay_completion_v1 import verify_registration
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen full original control registration changed')
    reviewed = read(ROOT / config['source_review_configuration'])
    producer, completion, replay, receipt, case, job, result = verify_registration(reviewed)
    proof = read(ROOT / config['full_native_review'])
    if (config['settings'] != reviewed['settings'] or config['resource_lock'] != reviewed['resource_lock']
            or config['python'] != producer['python'] or config['cpu_threads'] != producer['cpu_threads']
            or not proof['passed'] or not proof['full_original_model_checkpoint_reinference_completed']
            or not proof['all_fifteen_full_saved_and_replay_streams_independently_reread']
            or proof['canonical_id'] != job['id'] or proof['original_training_seed_slots'] != 1
            or config['multipliers'] != case['settings']['score_temperature_multipliers']
            or config['streams'] != case['settings']['score_streams']
            or len(config['multipliers']) * len(config['streams']) != 15
            or config['required_cases'] != 16 or config['train_score_points'] != 13238200
            or config['test_score_points'] != 87800 or config['dropped_original_tail_points'] != 41):
        raise ValueError('Retain the entire original control grid, full source data, native environment and readonly safeguards')
    return job, result, proof


def load_values(config, result):
    import numpy as np
    path = ROOT / config['score_archive']
    if sha(path) != next(a['sha256'] for a in result['artifacts'] if a['path'].endswith('/author_scores.npz')):
        raise ValueError('Exact fully replayed original score archive required')
    with zipfile.ZipFile(path) as archive:
        if sum(a.file_size for a in archive.infolist()) > config['maximum_uncompressed_score_bytes']:
            raise ValueError('Every full source array must fit the original readonly admission budget')
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if len(values) != 15 or any(not np.isfinite(a).all() for a in values.values()):
        raise ValueError('Every complete finite original score stream is required')
    return values


def worker(config, path):
    import numpy as np
    import torch
    job, result, proof = binding(config)
    if torch.__version__ != result['torch_version']: raise ValueError('Retain the original native score-control Torch backend')
    torch.set_num_threads(config['cpu_threads'])
    values = load_values(config, result)
    target = ROOT / config['artifact_directory']
    if target.exists(): raise FileExistsError('Preserve complete calibration-control attempts')
    if shutil.disk_usage(target.parent).free < config['minimum_free_disk_bytes']:
        raise OSError('Complete all-case array evidence needs its registered disk reservation')
    target.mkdir(parents=True)
    quantile = 100 - job['author_config']['data']['anomaly_ratio']; records = []
    def capture(factor, temperature, stream, train, test, labels, predicted):
        if train.shape != (config['train_score_points'],) or test.shape != (config['test_score_points'],) or not np.array_equal(labels, values['labels']):
            raise ValueError('Every original overlapping training score and covered test point is required')
        train_threshold = train_only_threshold(train, quantile)
        pooled_threshold = float(np.percentile(np.concatenate([train, test]), quantile))
        train_prediction = (test > train_threshold).astype(int)
        pooled_prediction = (test > pooled_threshold).astype(int)
        if not np.array_equal(predicted, train_prediction): raise ValueError('The actual counterfactual predictions must use training-only calibration')
        metrics = {}
        for name, prediction in [('train_only', train_prediction), ('pooled', pooled_prediction)]:
            metrics[name] = dict(pointwise=source.binary_metrics(labels, prediction, test),
                author_PA=source.binary_metrics(labels, source.author_pa(labels, prediction)))
        train_path = target / f'case_{len(records):02d}_full_train.npy'
        test_path = target / f'case_{len(records):02d}_full_test.npz'
        with train_path.open('xb') as stream_file: np.save(stream_file, train, allow_pickle=False)
        with test_path.open('xb') as stream_file:
            np.savez_compressed(stream_file, scores=test, labels=labels,
                train_only_predictions=train_prediction, pooled_predictions=pooled_prediction,
                train_only_PA=source.author_pa(labels, train_prediction), pooled_PA=source.author_pa(labels, pooled_prediction))
        row = dict(case_index=len(records), temperature_multiplier=factor, temperature=temperature, stream=stream,
            train_only_threshold=train_threshold, pooled_threshold=pooled_threshold, metrics=metrics,
            full_train_scores=str(train_path), full_train_scores_sha256=sha(train_path),
            full_test_scores=str(test_path), full_test_scores_sha256=sha(test_path),
            complete_original_training_score_points=len(train), complete_original_scored_test_points=len(test),
            test_scores_used_for_train_only_calibration=False, labels_used_for_PA=True,
            original_test_selected_checkpoint_retained=True, strict_TAB_result=False)
        records.append(row)
        write(target / 'progress.json', dict(required_cases=16, complete_cases=len(records), latest=row))
        print(dict(complete_cases=len(records), required_cases=16, stream=stream), flush=True)
    direct_train, direct_test = values['fused_train'], values['test_energy']
    direct_predicted = (direct_test > train_only_threshold(direct_train, quantile)).astype(int)
    capture(None, None, 'original_saved_source_energy', direct_train, direct_test, values['labels'], direct_predicted)
    if records[0]['pooled_threshold'] != float(values['threshold']) or not np.array_equal(values['raw_predictions'],
            (direct_test > records[0]['pooled_threshold']).astype(int)):
        raise ValueError('Direct original source calibration and every original prediction must remain identical')
    arguments = (values, job['author_config']['testing']['temperature'], job['author_config']['data']['anomaly_ratio'],
        job['author_config']['data']['win_size'], config['multipliers'], config['streams'])
    original_controls = source.score_controls(*arguments)
    if original_controls != read(ROOT / config['original_posthoc_controls'])['records']:
        raise ValueError('All fifteen complete native original pooled controls must reproduce exactly before comparison')
    namespace = dict(vars(source), train_only_threshold=train_only_threshold, capture_full_case=capture)
    exec(compile(controls_ast(True), source.__file__, 'exec'), namespace)
    train_controls = namespace['score_controls'](*arguments)
    if len(records) != 16 or len(train_controls) != 15: raise ValueError('Run every original control, without selecting a test-label-favored subset')
    for row, baseline, counterfactual in zip(records[1:], original_controls, train_controls):
        if (row['metrics']['pooled']['pointwise'] != baseline['pointwise'] or row['metrics']['pooled']['author_PA'] != baseline['author_PA']
                or row['metrics']['train_only']['pointwise'] != counterfactual['pointwise']
                or row['metrics']['train_only']['author_PA'] != counterfactual['author_PA']
                or row['train_only_threshold'] != counterfactual['threshold'] or row['pooled_threshold'] != baseline['threshold']):
            raise ValueError('Every actual full paired score-control metric must agree')
    if torch.cuda.is_initialized(): raise ValueError('Posthoc score controls must remain CPU-only')
    write(target / 'full_train_only_calibration_controls.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=path, configuration_sha256=sha(ROOT / path), passed=True, required_cases=16, complete_cases=16,
        original_id=job['id'], original_seed=job['seed'], original_required_training_seeds=proof['original_required_training_seeds'],
        original_training_seeds_used=1, dropped_original_test_tail_points=41, complete_original_pooled_controls_reproduced=15,
        original_full_model_checkpoint_reinference_previously_verified=True, records=records,
        additional_training_seed_slots=0, additional_model_inference_executed=False, cuda_context_initialized=False,
        original_test_selected_checkpoint_retained=True, strict_TAB_result=False, author_equivalence_certified=False,
        original_scope=config['original_scope'], all_original_experiments_complete=False,
        boundary='Complete one-seed paired calibration diagnostics; train-only thresholds cannot undo original test-based model selection. Full overlapping train values are calibration observations, not independent training replicates. PA remains a label-dependent retrospective transformation. No test-label selection of cases, missing-tail recovery, TAB or paper equivalence claim.'))


def review_worker(config, path):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scripts.flow_matching.audit_pi_original_recovery_completion_v1 import count_metrics, interval_pa, same_numbers
    job, result, proof = binding(config); torch.set_num_threads(config['cpu_threads'])
    if torch.__version__ != result['torch_version']: raise ValueError('Native complete score reference backend changed')
    report_path = ROOT / config['artifact_directory'] / 'full_train_only_calibration_controls.json'
    report = read(report_path); records = report['records']; values = load_values(config, result)
    if not report['passed'] or report['complete_cases'] != 16 or len(records) != 16 or report['configuration_sha256'] != sha(ROOT / path):
        raise ValueError('Every actual complete native paired control must finish before independent review')
    quantile = 100 - job['author_config']['data']['anomaly_ratio']; reviewed = []
    for index, row in enumerate(records):
        train_path = Path(row['full_train_scores']); test_path = Path(row['full_test_scores'])
        if sha(train_path) != row['full_train_scores_sha256'] or sha(test_path) != row['full_test_scores_sha256']:
            raise ValueError('Full original calibration evidence array hash changed')
        train = np.load(train_path, mmap_mode='r', allow_pickle=False)
        with np.load(test_path, allow_pickle=False) as archive:
            test, labels = archive['scores'], archive['labels']
            if (train.shape != (13238200,) or test.shape != (87800,) or not np.isfinite(train).all()
                    or not np.isfinite(test).all() or not np.array_equal(labels, values['labels'])):
                raise ValueError('All full original calibration/reference/test points required')
            thresholds = dict(train_only=float(np.percentile(train, quantile)),
                pooled=float(np.percentile(np.concatenate([train, test]), quantile)))
            for name in ['train_only', 'pooled']:
                if thresholds[name] != row[name + '_threshold']: raise ValueError('Independent complete percentile calibration differs')
                prediction = (test > thresholds[name]).astype(int)
                if not np.array_equal(prediction, archive[name + '_predictions']) or not np.array_equal(interval_pa(labels, prediction), archive[name + '_PA']):
                    raise ValueError('Every independently counted original raw/interval-PA prediction must agree')
                pointwise, counts = count_metrics(labels, prediction)
                pointwise.update(auroc=float(roc_auc_score(labels, test)), average_precision=float(average_precision_score(labels, test)))
                adjusted, adjusted_counts = count_metrics(labels, archive[name + '_PA'])
                same_numbers(pointwise, row['metrics'][name]['pointwise']); same_numbers(adjusted, row['metrics'][name]['author_PA'])
            reviewed.append(dict(case_index=index, stream=row['stream'], temperature_multiplier=row['temperature_multiplier'],
                full_train_scores=13238200, full_test_scores=87800, thresholds_independently_recomputed=thresholds,
                full_raw_and_PA_predictions_independently_verified=True))
        del train; gc.collect()
    matched = []
    def match(factor, temperature, stream, train, test, labels, predicted):
        row = records[len(matched) + 1]
        with np.load(row['full_test_scores'], allow_pickle=False) as saved:
            if (row['temperature_multiplier'] != factor or row['temperature'] != temperature or row['stream'] != stream
                    or not np.array_equal(train, np.load(row['full_train_scores'], mmap_mode='r', allow_pickle=False))
                    or not np.array_equal(test, saved['scores']) or not np.array_equal(labels, saved['labels'])
                    or not np.array_equal(predicted, saved['pooled_predictions'])):
                raise ValueError('Every full original native score transform must match bitwise; only percentile reference changes')
        matched.append(row['case_index'])
    with np.load(records[0]['full_test_scores'], allow_pickle=False) as saved:
        if not np.array_equal(values['fused_train'], np.load(records[0]['full_train_scores'], mmap_mode='r', allow_pickle=False)) or not np.array_equal(values['test_energy'], saved['scores']):
            raise ValueError('Direct counterfactual must preserve all original saved native energy values')
    namespace = dict(vars(source), capture_full_case=match)
    exec(compile(controls_ast(False), source.__file__, 'exec'), namespace)
    originals = namespace['score_controls'](values, job['author_config']['testing']['temperature'],
        job['author_config']['data']['anomaly_ratio'], job['author_config']['data']['win_size'], config['multipliers'], config['streams'])
    if len(matched) != 15 or originals != read(ROOT / config['original_posthoc_controls'])['records']:
        raise ValueError('All original full native pooled controls must independently repeat without changing their source recipe')
    binding(config)
    if torch.cuda.is_initialized(): raise ValueError('CPU-only full normalized-array review must not initialize CUDA')
    exclusive_json(ROOT / config['output_directory'] / 'independent_full_calibration_controls_review.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), passed=True, complete_cases_independently_verified=16,
        native_control_report_sha256=sha(report_path), all15_full_original_normalized_control_streams_bitwise_recomputed=True,
        direct_original_native_score_arrays_bitwise_verified=True, every_full_percentile_raw_PA_and_metric_independently_verified=True,
        reviews=reviewed, original_training_seed_slots=1, original_required_training_seeds=proof['original_required_training_seeds'],
        original_test_selected_checkpoint_retained=True, original_dropped_tail_points=41, strict_TAB_result=False,
        author_equivalence_certified=False, additional_training_seed_slots=0, additional_model_inference_executed=False,
        original_scope=config['original_scope'], all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--review-worker', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config); binding(config)
    if args.worker: return worker(config, args.config)
    if args.review_worker: return review_worker(config, args.config)
    base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=False); api = resource_api()
    lane = api['exclusive_lock'](base / 'controller.lock')
    if lane is None: raise RuntimeError('This complete native calibration-control controller is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, state='starting')
    def update(values): state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)
    stages = []
    try:
        for flag, name in [('--worker', 'full_native_controls'), ('--review-worker', 'independent_full_array_review')]:
            with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
                command = [config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.run_pi_original_train_only_controls_v1', '--config', args.config, flag]
                with (base / (name + '.stdout.log')).open('xb') as out, (base / (name + '.stderr.log')).open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                            PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS=str(config['cpu_threads']),
                            MKL_NUM_THREADS=str(config['cpu_threads'])), creationflags=subprocess.CREATE_NO_WINDOW)
                    handle = psutil.Process(child.pid); update(dict(active_pid=child.pid,
                        active_create_time=handle.create_time(), actual_command=handle.cmdline(), stage=name))
                    code, usage = capped_wait(child, config, api, update)
            stage = dict(stage=name, exit_code=code, **usage); stages.append(stage)
            if code != 0 or usage['resource_guard_aborted']:
                write(base / 'receipt.json', dict(passed=False, stages=stages)); update(dict(state='failed_preserved', active_pid=None))
                raise RuntimeError('Preserve any failed full native calibration or independent-array stage')
        verified = read(ROOT / config['output_directory'] / 'independent_full_calibration_controls_review.json')
        if not verified['passed'] or verified['complete_cases_independently_verified'] != 16: raise ValueError('Full independent completion required')
        write(base / 'receipt.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
            stages=stages, complete_native_cases=16, complete_independent_cases=16,
            author_equivalence_certified=False, all_original_experiments_complete=False))
        update(dict(state='completed', active_pid=None))
    finally: lane.close()


if __name__ == '__main__': main()
