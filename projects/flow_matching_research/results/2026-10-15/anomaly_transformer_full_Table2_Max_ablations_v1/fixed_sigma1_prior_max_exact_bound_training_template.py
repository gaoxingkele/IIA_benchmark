def run_worker(config, job, config_path):
    contract = verify(config)
    import random
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    torch.manual_seed(job['seed'])
    torch.set_default_device('cpu')
    base = Path(config['artifact_root']) / job['id']
    base.mkdir(parents=True, exist_ok=False)
    folder, audit = prepare_data(config, job['dataset'], contract)
    workspace = base / 'source'
    shutil.copytree(ROOT / config['source_directory'], workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    attention = workspace / 'model/attn.py'
    original = sha(attention)
    attention.write_text(author.patched('model/attn.py', attention.read_text(encoding='utf-8')), encoding='utf-8')
    write(base / 'source_patch.json', dict(path='model/attn.py', original_sha256=original, runtime_sha256=sha(attention), numerical_algorithm_changes='CPU compatibility; fixed Gaussian sigma1; sigma projection removed after consuming its paired initialization RNG'))
    sys.path.insert(0, str(workspace))
    from model.AnomalyTransformer import AnomalyTransformer
    model = AnomalyTransformer(win_size=100, enc_in=audit['split']['features'], c_out=audit['split']['features'], d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0)
    if any((p.device.type != 'cpu' for p in model.parameters())):
        raise ValueError('Actual paper reconstruction parameters must be on CPU')
    arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r') for name in ['train', 'validation', 'test']}

    class Windows(Dataset):

        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data) // 100

        def __getitem__(self, index):
            return np.array(self.data[index * 100:(index + 1) * 100], copy=True)
    loaders = {k: DataLoader(Windows(v), batch_size=32, shuffle=k == 'train', num_workers=0) for k, v in arrays.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = []
    best = None
    patience = 3
    updates = 0
    checkpoint = base / 'model.pt'
    start = time.perf_counter()
    for epoch in range(1, 11):
        model.train()
        sums = [0.0, 0.0, 0.0]
        count = 0
        steps = 0
        for batch in loaders['train']:
            optimizer.zero_grad()
            reconstruction, series, priors, _ = model(batch)
            minimum, maximum, rec = minimax_losses(reconstruction, batch, priors, series)
            if not all((torch.isfinite(v) for v in [minimum, maximum, rec])):
                raise ValueError('Nonfinite mathematical training loss; preserve attempt without retry')
            maximum.backward()
            optimizer.step()
            steps += 1
            updates += 1
            count += len(batch)
            for i, v in enumerate([minimum, maximum, rec]):
                sums[i] += float(v.detach()) * len(batch)
        model.eval()
        validation_sums = [0.0, 0.0]
        n = 0
        with torch.no_grad():
            for batch in loaders['validation']:
                reconstruction, series, priors, _ = model(batch)
                minimum, maximum, _ = minimax_losses(reconstruction, batch, priors, series)
                if not torch.isfinite(minimum) or not torch.isfinite(maximum):
                    raise ValueError('Nonfinite validation loss')
                n += len(batch)
                validation_sums[0] += float(minimum) * len(batch)
                validation_sums[1] += float(maximum) * len(batch)
        validation = [v / n for v in validation_sums]
        improved = best is None or (validation[0] <= best[0] and validation[1] <= best[1])
        if improved:
            best = validation
            patience = 3
            torch.save(model.state_dict(), checkpoint)
        else:
            patience -= 1
        epochs.append(dict(epoch=epoch, steps=steps, train_windows=count, train_losses=[v / count for v in sums], validation_windows=n, validation_losses=validation, checkpoint_saved=improved, remaining_patience=patience, learning_rate=optimizer.param_groups[0]['lr']))
        write(base / 'epochs.json', epochs)
        if patience == 0:
            break
        for group in optimizer.param_groups:
            group['lr'] = 0.0001 * 0.5 ** (epoch - 1)
    training_seconds = time.perf_counter() - start
    if updates != sum((e['steps'] for e in epochs)) or any((e['steps'] != math.ceil(len(loaders['train'].dataset) / 32) for e in epochs)):
        raise ValueError('Original full non-overlapping training coverage missing')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    all_scores = {}
    score_artifacts = []
    start = time.perf_counter()
    for split, data in arrays.items():
        loader = DataLoader(Windows(data), batch_size=32, shuffle=False, num_workers=0)
        buffers = {k: [] for k in STREAMS}
        rec_buffers = []
        discrepancy_buffers = []
        with torch.no_grad():
            for batch in loader:
                reconstruction, series, priors, _ = model(batch)
                scores, rec, layers = score_components(reconstruction, batch, priors, series)
                for name, value in scores.items():
                    buffers[name].append(value.numpy().reshape(-1))
                rec_buffers.append(rec.numpy().reshape(-1))
                discrepancy_buffers.append(layers.numpy().transpose(1, 2, 0).reshape(-1, 3))
        full = {k: np.concatenate(v) for k, v in buffers.items()}
        full['component.reconstruction'] = np.concatenate(rec_buffers)
        full['component.layer_discrepancies'] = np.concatenate(discrepancy_buffers, axis=0)
        if not all((np.isfinite(v).all() for v in full.values())):
            raise ValueError('Nonfinite full score components')
        expected = len(data) // 100 * 100
        if any((v.shape[0] != expected for v in full.values())):
            raise ValueError('Incomplete split score coverage')
        path = base / (split + '_scores.npz')
        np.savez(path, **full)
        score_artifacts.append(path)
        all_scores[split] = full
    labels = np.asarray(np.load(folder / 'labels.npy', mmap_mode='r')[:len(all_scores['test']['all.multiplication'])])
    np.save(base / 'covered_test_labels.npy', labels, allow_pickle=False)
    metrics = {name: evaluation(labels, all_scores['test'][name], all_scores['validation'][name], contract['paper_evaluation']['r_percent'][job['dataset']], contract['paper_evaluation']['fixed_delta'][job['dataset']], contract['paper_evaluation']['ROC_predefined_r_percent']) for name in STREAMS}
    score_seconds = time.perf_counter() - start
    artifacts = [checkpoint, base / 'epochs.json', base / 'source_patch.json', base / 'covered_test_labels.npy', *score_artifacts]
    write(base / 'result.json', dict(status='completed', job=job, configuration_sha256=sha(config_path), parameter_count=sum((p.numel() for p in model.parameters())), epochs=epochs, optimizer_updates=updates, actual_backward_calls=updates, training_seconds=training_seconds, score_seconds=score_seconds, data_audit=audit, data_manifest_sha256=sha(folder / 'manifest.json'), covered_points={k: len(all_scores[k]['all.multiplication']) for k in arrays}, omitted_tail_points={k: len(arrays[k]) % 100 for k in arrays}, metrics=metrics, checkpoint_selection_uses_test=False, threshold_calibration_uses_test=False, paper_algorithm_equivalence_certified=False, strict_TAB_result=False, independent_full_checkpoint_replay_complete=False, explicit_reconstruction_choices=config['explicit_reconstruction_choices'], artifacts=[dict(path=str(p), sha256=sha(p)) for p in artifacts], boundary=config['boundary']))
