"""Full-checkpoint Tiny/ICM representations and the pinned source SVM procedure.

No pretrained release weights or random representations may replace the two
complete local PILE training results. These are disclosed CPU reconstructions,
not certification of the unpublished ICM pipeline or its historical metrics.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import types
import warnings
from scripts.flow_matching import run_moment_icm_complete_pretraining_v1 as pretrain
from scripts.flow_matching import prepare_moment_icm_classification_train_only_v1 as strict

ROOT, read, sha, write = strict.ROOT, strict.read, strict.sha, strict.write
SVM_SOURCE = 'experiments/runs/mtsad_protocol_audit/sources/moment_research/original/moment/models/statistical_classifiers.py'


def trained_checkpoint(config, variant):
    """This gate reads metadata/hash evidence only; it never imports Torch."""
    training = read(ROOT/pretrain.CONFIG)
    job = next(j for j in training['jobs'] if j['mixer'] == variant)
    output = Path(training['artifact_root'])/job['id']
    result_path = output/'pretraining_result.json'
    guard_path = ROOT/training['state_root']/job['id']/'guard_receipt.json'
    if not result_path.exists() or not guard_path.exists():
        return None
    result, guard = read(result_path), read(guard_path)
    if (not result['passed'] or result['job'] != job or result['configuration_sha256'] != sha(ROOT/pretrain.CONFIG)
            or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['proof_sha256'] != sha(result_path) or result['full_optimizer_updates'] != 39479
            or result['full_epoch_items'] != 80850189):
        raise ValueError('Actual complete successful PILE training and guard required')
    proof_path = output/'complete_training_state.json'
    if sha(proof_path) != result['complete_training_state_sha256']:
        raise ValueError('Complete training state proof changed')
    proof = read(proof_path); final = proof['final_checkpoint']
    if (not proof['passed'] or proof['job'] != job or proof['configuration_sha256'] != sha(ROOT/pretrain.CONFIG)
            or not proof['complete_declared_epoch_training_state_saved']
            or proof['actual_audited_optimizer_updates'] != 39479 or proof['scope']['items'] != 80850189
            or proof['scope']['batches'] != 39479 or final['phase'] != 'final_post_scheduler'
            or final['optimizer_updates'] != 39479 or not final['all_saved_state_after_cpu_load_exact']
            or not final['training_rng_unchanged_by_checkpoint_IO'] or sha(final['path']) != final['sha256']):
        raise ValueError('All full final state and original checkpoint bytes required')
    return dict(job=job, checkpoint=final, training_scope=proof['scope'], parameter_count=result['parameter_count'],
        pretraining_result_sha256=sha(result_path), complete_training_state_sha256=sha(proof_path),
        original_training_guard_sha256=sha(guard_path), pretraining_configuration_sha256=sha(ROOT/pretrain.CONFIG))


def dataset_binding(config, dataset, track):
    parent = read(ROOT/strict.PARENT)
    job = next(j for j in parent['jobs'] if j['dataset'] == dataset)
    source_dir = Path(parent['artifact_root'])/dataset
    proof_path = source_dir/'preprocessing.json'
    guard_path = ROOT/parent['state_root']/'attempts'/dataset/'guard_receipt.json'
    original, guard = read(proof_path), read(guard_path)
    if (not original['passed'] or original['job'] != job or original['configuration_sha256'] != sha(ROOT/strict.PARENT)
            or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['proof_sha256'] != sha(proof_path)):
        raise ValueError('Every complete original preprocessing proof/guard required')
    directories = dict(TRAIN=str(source_dir/'TRAIN'), TEST=str(source_dir/'TEST'))
    records = original['records']
    binding = dict(original_proof_sha256=sha(proof_path), original_guard_sha256=sha(guard_path))
    if track == 'TRAIN_fitted_control':
        control = read(ROOT/strict.CONFIG); root = Path(control['artifact_root'])/dataset
        p, g = root/'preprocessing.json', ROOT/control['state_root']/'attempts'/dataset/'guard_receipt.json'
        if not p.exists() or not g.exists():
            return None
        report, guard = read(p), read(g)
        if (not report['passed'] or report['job'] != job or report['configuration_sha256'] != sha(ROOT/strict.CONFIG)
                or not report['strict_train_only_standardization'] or report['test_scaler_fit_calls'] != 0
                or not report['unchanged_TRAIN']['all_original_train_coordinates_checked']
                or report['unchanged_TRAIN']['directory'] != directories['TRAIN']
                or not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
                or guard['proof_sha256'] != sha(p)):
            raise ValueError('Actual full successful separate TRAIN-fitted control required')
        directories['TEST'] = str(root/'TEST')
        records = dict(TRAIN=records['TRAIN'], TEST=report['TEST'])
        binding.update(control_proof_sha256=sha(p), control_guard_sha256=sha(g))
    elif track != 'source_TEST_fitted':
        raise ValueError('Keep the two preprocessing tracks explicit')
    for split in ['TRAIN', 'TEST']:
        if sha(ROOT/job['splits'][split]['path']) != job['splits'][split]['sha256']:
            raise ValueError('Official raw split changed')
    return dict(dataset=dataset, track=track, original_job=job, directories=directories,
                records=records, preprocessing_binding=binding)


def load_trained_core(config, trained):
    import torch
    body = read(ROOT/pretrain.masks.TRAIN_BODY)
    parameters = dict(read(ROOT/body['core_model_configuration'])['parameters'], mixer=trained['job']['mixer'])
    core = pretrain.train.core_class(ROOT/'src/iia_benchmark/models')(**parameters)
    payload = torch.load(trained['checkpoint']['path'], map_location='cpu', weights_only=False)
    if (payload['job'] != trained['job'] or payload['configuration_sha256'] != trained['pretraining_configuration_sha256']
            or payload['optimizer_updates'] != 39479 or payload['phase'] != 'final_post_scheduler'
            or payload['scope'] != trained['training_scope']):
        raise ValueError('The complete trained final checkpoint identity is mandatory')
    # The original wrapper stores every parameter under core.*,including the
    # reconstruction head. Load all of it before changing only the output mode.
    wrapper = torch.nn.Module(); wrapper.add_module('core', core)
    state = payload['model_state_dict']
    if any(not torch.isfinite(x).all() for x in state.values()):
        raise ValueError('Finite complete trained parameters required')
    incompatible = wrapper.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError('Full checkpoint keys must load strictly')
    if sum(p.numel() for p in wrapper.parameters()) != trained['parameter_count']:
        raise ValueError('No parameter or reconstruction head may be omitted')
    for record in payload['optimizer_state_dict']['state'].values():
        if float(record['step']) != 39479:
            raise ValueError('Original complete optimizer update counters required')
    if not payload['optimizer_state_dict']['state']:
        raise ValueError('Actual trained optimizer state required')
    del payload, state
    core.task = 'features'; core.requires_grad_(False); core.eval()
    return core


def verify_bound_arrays(binding):
    for split in ['TRAIN', 'TEST']:
        for name, digest in binding['records'][split]['files_sha256'].items():
            if sha(Path(binding['directories'][split])/name) != digest:
                raise ValueError('Full immutable original preprocessing arrays changed')


def verify_existing_evidence(config, job, output, phase):
    """Successful old receipts do not permit changed current inputs or outputs."""
    trained = trained_checkpoint(config, job['variant'])
    binding = dataset_binding(config, job['dataset'], job['track'])
    if trained is None or binding is None:
        raise ValueError('Actual complete model/data bindings are still required')
    verify_bound_arrays(binding)
    output = Path(output); native_path = output/'native_classification.json'; native = read(native_path)
    if (not native['passed'] or native['job'] != job or native['trained_checkpoint'] != trained
            or native['dataset_binding'] != binding or native['configuration_sha256'] != sha(ROOT/config['_config_path'])
            or sha(output/'classifier.joblib') != native['classifier_sha256']):
        raise ValueError('Existing native proof must bind current complete inputs')
    for name, digest in native['feature_proof']['files_sha256'].items():
        if sha(output/'features'/name) != digest: raise ValueError('Saved complete features changed')
    for split in ['TRAIN', 'TEST']:
        if sha(output/(split+'_predictions.npy')) != native['predictions_sha256'][split]:
            raise ValueError('Saved classifier predictions changed')
    if phase == 'independent':
        review = read(output/'independent_review.json')
        if (not review['passed'] or review['job'] != job or review['native_proof_sha256'] != sha(native_path)
                or review['metrics'] != native['metrics']):
            raise ValueError('Saved independent proof must bind current full native result')
    elif phase != 'native': raise ValueError('Known complete scientific phase required')
    return native


def pooled_features(core, values, masks, tolerance):
    """Keep every channel;source masked-patch mean then flatten C by D."""
    import numpy as np
    import torch
    if values.ndim != 3 or values.shape[-1] != 256 or masks.shape != (len(values), 256):
        raise ValueError('Complete [case,channel,256] and case masks required')
    if not np.isfinite(values).all() or not np.isin(masks, [0, 1]).all():
        raise ValueError('Finite source values and exact binary masks required')
    x = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
    observed = torch.from_numpy(np.asarray(masks).copy()).to(dtype=torch.int64)
    valid = observed.unfold(-1, 8, 8).sum(-1).eq(8)
    if not valid.any(dim=-1).all():
        raise ValueError('No full observed patch;preserve without invented normalization')
    with torch.inference_mode():
        features = core(x, input_mask=observed, reconstruction_mask=observed)
        if features.shape != (len(values), values.shape[1], 32, 256) or not torch.isfinite(features).all():
            raise ValueError('Every actual channel/patch/feature must be finite and complete')
        pooled = (features*valid[:, None, :, None]).sum(2)/valid.sum(1)[:, None, None]
        actual = pooled.flatten(start_dim=1).cpu().numpy()
    # Every live representation is also reduced with independent float64 NumPy.
    f = features.cpu().numpy().astype(np.float64)
    v = valid.cpu().numpy()
    expected = ((f*v[:, None, :, None]).sum(2)/v.sum(1)[:, None, None]).reshape(len(values), -1)
    if not np.allclose(actual, expected, **tolerance):
        raise ValueError('Complete independent patch-pooling check failed')
    return actual, float(np.max(np.abs(actual-expected), initial=0.))


def export_features(core, binding, output, tolerance):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    coverage = []
    for split in ['TRAIN', 'TEST']:
        directory = Path(binding['directories'][split]); record = binding['records'][split]
        for name, digest in record['files_sha256'].items():
            if sha(directory/name) != digest:
                raise ValueError('Full immutable preprocessed values/masks/labels changed')
        values = np.load(directory/'values.npy', mmap_mode='r', allow_pickle=False)
        masks = np.load(directory/'input_mask.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(directory/'labels.npy', allow_pickle=False)
        rows = np.load(directory/'original_instance_rows.npy', allow_pickle=False)
        n, c, t = values.shape
        if (n != binding['original_job']['splits'][split]['instances'] or t != 256
                or labels.shape != (n,) or not np.array_equal(rows, np.arange(n))):
            raise ValueError('All original official instances and labels required')
        target = np.lib.format.open_memmap(output/(split+'_features.npy'), mode='w+', dtype=np.float32, shape=(n, c*256))
        error = 0.
        # One complete multivariate case per inference call is disclosed. All
        # channels are simultaneous; no channel chunks break ICM dependencies.
        for index in range(n):
            feature, difference = pooled_features(core, values[index:index+1], masks[index:index+1], tolerance)
            target[index] = feature[0]; error = max(error, difference)
        target.flush(); del target
        np.save(output/(split+'_labels.npy'), labels, allow_pickle=False)
        np.save(output/(split+'_original_instance_rows.npy'), rows, allow_pickle=False)
        coverage.append(dict(split=split, instances=n, channels=c, sequence_length=256,
                             feature_dimensions=c*256, full_case_forwards=n, maximum_pooling_absolute_difference=error))
    return dict(coverage=coverage, files_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                full_original_cases_and_channels_represented=True, new_training_updates=0)


def source_svm(features, labels, expected_source_sha256):
    """Run the exact original fit_svm;capture its unchanged grid and warnings."""
    import numpy as np
    import numpy.typing as npt
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.svm import SVC
    path = ROOT/SVM_SOURCE
    if sha(path) != expected_source_sha256:
        raise ValueError('Exact registered source SVM code required')
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'fit_svm')
    searches = []
    def capture_search(*args, **kwargs):
        search = GridSearchCV(*args, **kwargs); searches.append(search); return search
    namespace = dict(np=np, npt=npt, SVC=SVC, GridSearchCV=capture_search, train_test_split=train_test_split)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier = namespace['fit_svm'](features=features, y=labels)
    grid = []
    for search in searches:
        for i, parameters in enumerate(search.cv_results_['params']):
            value = float(search.cv_results_['mean_test_score'][i])
            grid.append(dict(C=('positive_infinity' if math.isinf(parameters['C']) else parameters['C']),
                mean_training_CV_accuracy=value if math.isfinite(value) else None,
                candidate_failed=not math.isfinite(value), rank=int(search.cv_results_['rank_test_score'][i])))
    selected = len(features) if not searches else min(len(features), 10000)
    chosen = {k: ('positive_infinity' if float(v) > 0 else 'negative_infinity')
              if isinstance(v, (float, np.floating)) and math.isinf(v) else v
              for k, v in classifier.get_params().items()}
    return classifier, dict(exact_original_fit_svm_AST=True, original_CV5_n_jobs10_and_MAX_SAMPLES10000_retained=True,
        original_SMALL_branch_C100000_retained=True, source_grid_infinite_C_retained_without_replacement=True,
        complete_official_TRAIN_feature_instances=len(features), original_source_SVM_fit_instances=selected,
        original_source_SVM_training_subsample_applied=selected != len(features),
        grid=grid, warnings=[dict(category=w.category.__name__, text=str(w.message)) for w in caught],
        chosen_parameters=chosen, nonfinite_parameter_symbols_are_logging_only=True,
        actual_chosen_C_is_positive_infinity=math.isinf(float(classifier.C)) and classifier.C > 0,
        test_labels_used_for_selection=False)


def independent_classification(labels, predictions):
    """Scalar confusion/F1/Wilson arithmetic;fixed-model descriptive uncertainty."""
    import numpy as np
    y, pred = np.asarray(labels), np.asarray(predictions)
    if y.ndim != 1 or pred.shape != y.shape or len(y) == 0:
        raise ValueError('Every original test instance prediction required')
    classes = sorted(set(y.tolist())|set(pred.tolist())); confusion = [[0 for _ in classes] for _ in classes]
    lookup = {v: i for i, v in enumerate(classes)}
    for truth, guess in zip(y.tolist(), pred.tolist()): confusion[lookup[truth]][lookup[guess]] += 1
    n = len(y); correct = sum(confusion[i][i] for i in range(len(classes))); accuracy = correct/n
    f1 = []
    for i in range(len(classes)):
        tp = confusion[i][i]; fp = sum(row[i] for row in confusion)-tp; fn = sum(confusion[i])-tp
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    z = 1.959963984540054; denominator = 1+z*z/n
    middle = (accuracy+z*z/(2*n))/denominator
    half = z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/denominator
    return dict(instances=n, correct=correct, accuracy=accuracy, macro_F1=sum(f1)/len(f1), classes=classes,
                confusion=confusion, descriptive_Wilson95_low=max(0., middle-half), descriptive_Wilson95_high=min(1., middle+half),
                unknown_subject_group_dependence_accounted_for=False, model_seed_uncertainty_estimated=False)


def run_case(config, job, output):
    trained = trained_checkpoint(config, job['variant'])
    binding = dataset_binding(config, job['dataset'], job['track'])
    if trained is None or binding is None:
        raise ValueError('Actual full trained model and preprocessing required before numerical imports')
    verify_bound_arrays(binding)
    if pretrain.numerical_environment() != read(ROOT/pretrain.masks.TRAIN_BODY)['verified_native_environment']:
        raise ValueError('Exact original full-training numerical environment required')
    import joblib
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    utils = ROOT/read(ROOT/pretrain.masks.TRAIN_BODY)['original_source_root']/'moment/utils/utils.py'
    seed = next(n for n in ast.parse(utils.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'control_randomness')
    namespace = dict(random=random, os=os, np=np, torch=torch)
    exec(compile(ast.Module(body=[seed], type_ignores=[]), str(utils), 'exec'), namespace)
    namespace['control_randomness'](job['seed'])
    model = load_trained_core(config, trained)
    feature_proof = export_features(model, binding, output/'features', config['pooling_tolerance'])
    del model
    values = {name: np.load(output/'features'/name, mmap_mode='r', allow_pickle=False)
              for name in ['TRAIN_features.npy', 'TRAIN_labels.npy', 'TEST_features.npy', 'TEST_labels.npy']}
    classifier, audit = source_svm(values['TRAIN_features.npy'], values['TRAIN_labels.npy'], config['SVM_source_sha256'])
    model_path = output/'classifier.joblib'; joblib.dump(classifier, model_path)
    metrics = {}
    for split in ['TRAIN', 'TEST']:
        x, y = values[split+'_features.npy'], values[split+'_labels.npy']
        prediction = classifier.predict(x); np.save(output/(split+'_predictions.npy'), prediction, allow_pickle=False)
        independent = independent_classification(y, prediction)
        if (float(accuracy_score(y, prediction)) != independent['accuracy']
                or not math.isclose(float(f1_score(y, prediction, average='macro', zero_division=0)), independent['macro_F1'], abs_tol=1e-12)):
            raise ValueError('All original predictions and independent metrics must agree')
        metrics[split] = independent
    write(output/'native_classification.json', dict(passed=True, job=job, configuration_sha256=sha(ROOT/config['_config_path']),
        trained_checkpoint=trained, dataset_binding=binding, feature_proof=feature_proof, SVM_source_audit=audit, metrics=metrics,
        classifier_sha256=sha(model_path), predictions_sha256={split: sha(output/(split+'_predictions.npy')) for split in ['TRAIN', 'TEST']},
        all_cases_and_channels_retained=True, feature_inference_device='cpu', feature_dtype='float32', case_batch_size=1,
        new_training_updates=0, historical_paper_equivalence_certified=False, strict_leaderboard_result=False,
        original_periodic_AD_callbacks_completed=False, all_original_experiments_complete=False))


def review_case(config, job, output):
    trained = trained_checkpoint(config, job['variant']); binding = dataset_binding(config, job['dataset'], job['track'])
    if trained is None or binding is None:
        raise ValueError('Full successful prerequisites remain required')
    verify_bound_arrays(binding)
    output = Path(output); path = output/'native_classification.json'; native = read(path)
    if (not native['passed'] or native['job'] != job or native['trained_checkpoint'] != trained or native['dataset_binding'] != binding
            or native['configuration_sha256'] != sha(ROOT/config['_config_path'])
            or sha(output/'classifier.joblib') != native['classifier_sha256']):
        raise ValueError('Actual unchanged native result and full input identity required')
    import joblib
    import numpy as np
    for name, digest in native['feature_proof']['files_sha256'].items():
        if sha(output/'features'/name) != digest: raise ValueError('Complete native feature arrays changed')
    classifier = joblib.load(output/'classifier.joblib'); metrics = {}
    for split in ['TRAIN', 'TEST']:
        x = np.load(output/'features'/(split+'_features.npy'), mmap_mode='r', allow_pickle=False)
        y = np.load(output/'features'/(split+'_labels.npy'), allow_pickle=False)
        original_labels = np.load(Path(binding['directories'][split])/'labels.npy', allow_pickle=False)
        saved = np.load(output/(split+'_predictions.npy'), allow_pickle=False)
        if (not np.array_equal(y, original_labels) or not np.isfinite(x).all()
                or sha(output/(split+'_predictions.npy')) != native['predictions_sha256'][split]):
            raise ValueError('Every complete original feature,label and prediction required')
        prediction = classifier.predict(x)
        if prediction.dtype != saved.dtype or not np.array_equal(prediction, saved):
            raise ValueError('Reloaded entire classifier predictions must be byte-identical')
        metrics[split] = independent_classification(y, prediction)
        if metrics[split] != native['metrics'][split]: raise ValueError('Full independent metric table differs')
    write(output/'independent_review.json', dict(passed=True, job=job, native_proof_sha256=sha(path), metrics=metrics,
        every_original_classifier_prediction_recomputed=True, every_original_label_and_feature_checked=True,
        fresh_neural_inference_in_review=False, historical_paper_equivalence_certified=False,
        original_periodic_AD_callbacks_completed=False, all_original_experiments_complete=False))
