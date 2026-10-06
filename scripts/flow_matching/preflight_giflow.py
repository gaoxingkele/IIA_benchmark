"""Validate GiFlow real graph data, backward pass and author Euler sampler on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.audit_author_code import working_copy
from scripts.flow_matching.prepare_saits_physio import sha


def main():
    import numpy as np
    import torch
    from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_giflow_cpu_preflight.v1.json')
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding='utf-8'))
    working_copy('giflow')
    source = ROOT / config['author_source']
    sys.path.insert(0, str(source))
    import data
    import models
    from utils.preprocessing import operator_generation, generate_time_laplacian, preprocess_fm, preprocess_fm_test
    args = argparse.Namespace(**config['arguments'])
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    registered = json.loads((ROOT / config['data_config']).read_text(encoding='utf-8'))
    data_root = ROOT / registered['output_root']
    audit = json.loads((data_root / 'data_audit.json').read_text(encoding='utf-8'))
    if sha(data_root / 'small36.h5') != audit['files']['small36.h5']['sha256']:
        raise ValueError('Author graph data changed')
    dm, edges, weights, nodes, time_dim = data.dataset_loading(args.dataset_name, args.missing_rate, args.missing_type,
        args.window, args.stride, args.adj_threshold, args.val_len, args.test_len, args.seed, args.batch_size, root=str(data_root))
    args.time_dim = time_dim
    node_ids = torch.arange(nodes)
    lap_edges, lap_weights = get_laplacian(edges, normalization='sym')
    lap = to_scipy_sparse_matrix(lap_edges, lap_weights, num_nodes=nodes).astype('float64')
    spatial = operator_generation(lap, args.k_eig, args.tau_s)
    temporal = operator_generation(generate_time_laplacian(args.window), args.k_eig, args.tau_t)
    model = models.FlowMatching(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    batch = next(iter(dm.train_dataloader(batch_size=2, shuffle=False)))
    x0, x, velocity, u, mask, time = preprocess_fm(batch, spatial, temporal, args.window)
    prediction = model(node_embed=node_ids, x=x, x0=x0, ex=u, edge_index=edges, mask=mask)
    loss = (prediction[mask] - velocity[mask]).abs().mean()
    if not torch.isfinite(loss) or not mask.any():
        raise ValueError('Invalid real graph training loss')
    loss.backward()
    optimizer.step()
    model.eval()
    test_batch = next(iter(dm.test_dataloader(batch_size=2, shuffle=False)))
    x0, x, target, u, mask = preprocess_fm_test(test_batch, spatial, temporal)
    with torch.no_grad():
        for step in range(20):
            times = torch.full((len(x), args.window, 1), step / 20)
            ex = torch.cat((u, times), dim=-1)
            velocity = model(node_embed=node_ids, x=x, x0=x0, ex=ex, edge_index=edges, mask=mask)
            x = torch.where(mask, x + velocity / 20, x)
    if not torch.isfinite(x).all():
        raise ValueError('Non-finite graph Euler samples')
    result = {'status': 'passed', 'diagnostic': True, 'config_sha256': sha(arguments.config),
              'nodes': nodes, 'time_covariates': time_dim, 'training_targets': int(batch.eval_mask.sum()),
              'test_targets': int(mask.sum()), 'training_loss_finite': True, 'sample_finite': True,
              'parameters': sum(p.numel() for p in model.parameters()), 'boundary': config['boundary'],
              'data_sha256': audit['files']['small36.h5']['sha256']}
    output = ROOT / config['output_root']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
