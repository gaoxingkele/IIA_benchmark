"""Observe unchanged native TF evaluator updates, predictions and trained states."""
import inspect
import json
from pathlib import Path

from scripts.flow_matching.run_gtgan_native_job_v1 import sha


def recompute_score(role, prediction, reference=None):
    import numpy as np
    if role == 'discriminative':
        real, fake = prediction
        labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake))))
        probabilities = np.squeeze(np.concatenate((real, fake), axis=0))
        return float(abs(0.5 - np.mean(labels == (probabilities > 0.5))))
    from sklearn.metrics import mean_absolute_error
    if reference is None or len(reference) != len(prediction):
        raise ValueError('All original predictive targets are required')
    return float(sum(mean_absolute_error(reference[i], prediction[i]) for i in range(len(reference))) / len(reference))


def observe_evaluator(function, role, output, expected_steps, cpu_threads=2):
    """Call the entire source function; never replace its training or returned score."""
    import numpy as np
    import tensorflow as tf
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    original_session = tf.compat.v1.Session
    sessions = []

    class ObservedSession(original_session):
        def __init__(self, *args, **kwargs):
            kwargs['config'] = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=cpu_threads,
                inter_op_parallelism_threads=1, device_count={'GPU': 0})
            super().__init__(*args, **kwargs)
            self.updates = 0
            self.prediction = None
            self.reference = None
            self.test_feed = None
            sessions.append(self)

        def run(self, fetches, feed_dict=None, *args, **kwargs):
            local = inspect.currentframe().f_back.f_locals
            solver = local.get('d_solver' if role == 'discriminative' else 'p_solver')
            fetching = fetches if isinstance(fetches, (list, tuple)) else [fetches]
            optimizing = solver is not None and any(item is solver for item in fetching)
            value = super().run(fetches, feed_dict, *args, **kwargs)
            if optimizing:
                self.updates += 1
            prediction_key = 'y_pred_real' if role == 'discriminative' else 'y_pred'
            native_prediction = local.get(prediction_key)
            if not optimizing and native_prediction is not None and any(item is native_prediction for item in fetching):
                if self.updates != expected_steps or self.prediction is not None:
                    raise ValueError('Full original evaluator update count and single native test pass required')
                self.prediction = value
                self.test_feed = {key.name: np.asarray(data) for key, data in feed_dict.items()}
                if role == 'predictive':
                    self.reference = np.asarray(local['Y_mb'])
            return value

    tf.compat.v1.Session = ObservedSession
    try:
        score = function()
        if len(sessions) != 1 or sessions[0].prediction is None:
            raise ValueError('The original evaluator session and test predictions are required')
        session = sessions[0]
        rebuilt = recompute_score(role, session.prediction, session.reference)
        if not np.isfinite(score) or not np.isclose(score, rebuilt, rtol=1e-7, atol=1e-7):
            raise ValueError('Original evaluator returned score differs from all saved test predictions')
        arrays = dict(session.test_feed)
        if role == 'discriminative':
            arrays.update(prediction_real=session.prediction[0], prediction_fake=session.prediction[1])
        else:
            arrays.update(prediction=session.prediction, original_reference=session.reference)
        array_path = folder / 'native_test_predictions.npz'
        np.savez_compressed(array_path, **arrays)
        with session.graph.as_default():
            variables = tf.compat.v1.global_variables()
            if not variables:
                raise ValueError('All trained evaluator weights and optimizer states required')
            saver = tf.compat.v1.train.Saver(var_list=variables)
            saver.save(session, str(folder / 'native_evaluator'), write_meta_graph=True)
        checkpoints = [dict(path=str(p), sha256=sha(p), size=p.stat().st_size)
            for p in sorted(folder.glob('native_evaluator.*'))]
        if len(checkpoints) != 3:
            raise ValueError('Native TF index, data and full meta graph required')
        receipt = dict(role=role, evaluator_function=function.native_function_name,
            optimizer_updates=session.updates, expected_optimizer_updates=expected_steps,
            original_returned_score=float(score), full_saved_predictions_recomputed_score=rebuilt,
            full_saved_prediction_score_check_passed=True, trained_variable_count=len(variables),
            checkpoint_files=checkpoints, array_path=str(array_path), array_sha256=sha(array_path),
            independent_checkpoint_replay_complete=False, evaluator_training_instances=1,
            additional_generator_training_seeds=0,
            uncertainty_scope='Post-hoc evaluator repetition; not another generator training seed')
        (folder / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        return score, receipt
    finally:
        tf.compat.v1.Session = original_session
        for session in sessions:
            session.close()
