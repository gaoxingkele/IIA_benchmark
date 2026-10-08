"""Fresh runtime-only repairs to the preserved official ModMaelNet snapshot."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Separate fresh corrected directory required')
    if destination.exists():
        manifest = json.loads((destination / 'runtime_patch_manifest.json').read_text(encoding='utf-8'))
        for record in manifest['files']:
            if digest(source / record['path']) != record['original_sha256'] or digest(destination / record['path']) != record['corrected_sha256']:
                raise ValueError('Previously prepared source changed')
        return manifest
    edits = {
        'run_anomaly.py': [("'--chunk_size', type=str, default='500'", "'--chunk_size', type=int, default=500")],
        'data_provider/data_factory.py': [
            ('    data_set = Data(\n', '    extra = {"chunk_size": args.chunk_size} if args.data == "MSL" else {}\n    data_set = Data(\n'),
            ('        chunk_size=args.chunk_size,', '        **extra,')],
        'layers/Attention.py': [
            ('self.distances = torch.zeros((window_size, window_size)).cuda()',
             'self.register_buffer("distances", torch.zeros((window_size, window_size)), persistent=False)'),
            ('repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()',
             'repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(queries.device)')],
        'utils/agentreward.py': [
            ('spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32)',
             'spaces.Box(low=-np.inf, high=np.inf, shape=(4, ), dtype=np.float32)'),
            ('observation = np.zeros(4)', 'observation = np.zeros(4, dtype=np.float32)')],
        'utils/tools.py': [('np.Inf', 'np.inf')],
        'exp/opt_rl2_anomaly.py': [
            ('        env_off=TrainEnvOffline_dist_conf(list_pred_sc=list_pred_models, list_thresholds=list_thresholds, list_gtruth=test_labels)',
             '        np.savez_compressed(os.path.join(path, "author_rl_inputs.npz"), model_scores=np.stack(list_pred_models), thresholds=np.asarray(list_thresholds), labels=test_labels)\n        env_off=TrainEnvOffline_dist_conf(list_pred_sc=list_pred_models, list_thresholds=list_thresholds, list_gtruth=test_labels)'),
            ('        model.learn(total_timesteps=len(list_pred_models[0]),progress_bar=True)',
             '        model.learn(total_timesteps=len(list_pred_models[0]),progress_bar=True)\n        model.save(os.path.join(path, "dqn_policy"))'),
            ('        csvreader.writerows([[prec,rec,f1,reward]])',
             '        csvreader.writerows([[prec,rec,f1,reward]])\n        np.savez_compressed(os.path.join(path, "author_rl_predictions.npz"), labels=test_labels, predictions=np.asarray(list_preds))\n        import json\n        with open(os.path.join(path, "author_rl_metrics.json"), "w", encoding="utf-8") as metrics_file:\n            json.dump({"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "last_step_reward": float(reward), "policy_timesteps": int(model.num_timesteps), "label_points": int(len(test_labels))}, metrics_file, indent=2)')]
    }
    corrected_text = {}
    for relative, substitutions in edits.items():
        text = (source / relative).read_text(encoding='utf-8')
        for before, after in substitutions:
            expected = 2 if relative == 'layers/Attention.py' else 3 if relative == 'utils/tools.py' else 1
            if text.count(before) != expected:
                raise ValueError('Unexpected author source: ' + relative)
            text = text.replace(before, after)
        corrected_text[relative] = text
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    for relative, text in corrected_text.items():
        (destination / relative).write_text(text, encoding='utf-8', newline='\n')
    records = []
    for path in sorted(source.rglob('*')):
        if path.is_file() and '.git' not in path.parts and '__pycache__' not in path.parts:
            relative = path.relative_to(source).as_posix()
            records.append({'path': relative, 'original_sha256': digest(path), 'corrected_sha256': digest(destination / relative),
                            'changed': relative in edits})
    manifest = {'source': str(source), 'corrected': str(destination), 'files': records,
                'repairs': ['Integer chunk argument; non-MSL loaders receive only supported arguments',
                            'NumPy 2 infinity spelling, retaining early-stopping comparisons',
                            'Attention distances follow model/tensor device; checkpoint parameter keys unchanged',
                            'Gym observation dtype/bounds match actual unclipped score and confidence values',
                            'Instrumentation persists raw base scores, labels, thresholds, policy and original RL metrics'],
                'retained_author_semantics': ['Train/validation overlap', 'Overlapping-window flattened test labels',
                                             'Train+test score percentile', 'PA-derived environment states',
                                             'RL reward training and evaluation on the same test labels'],
                'boundary': 'Executable author snapshot with documented runtime changes; no strict independence or original-paper equivalence claim.'}
    (destination / 'runtime_patch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
