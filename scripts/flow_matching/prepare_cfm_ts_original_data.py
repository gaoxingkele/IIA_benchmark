"""Generate the paper's three original ODE systems without overwriting raw data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def field(dataset, time, state):
    if dataset == 'lotka_volterra':
        x, y = state
        return [2*x-x*y, -4*y+x*y]
    if dataset == 'sine':
        return np.sin(state)
    if dataset == 'pendulum':
        theta, omega = state
        return [omega, -9.81*np.sin(theta)]
    raise ValueError('Unknown registered ODE')


def generate_case(protocol, track, dataset, seed):
    rng = np.random.default_rng(seed)
    count = track['trajectory_counts'][dataset['id']]
    horizon, points = dataset['horizon'], protocol['evaluation_points']
    if dataset['id'] == 'lotka_volterra':
        initial = rng.uniform(1, 7, size=(count, 2))
    elif dataset['id'] == 'sine':
        initial = rng.normal(size=(count, 1))
    else:
        initial = np.broadcast_to(dataset['initial_state'], (1, 2)).copy()
    observed_times, observed_states, test_times, test_states, physical = [], [], [], [], []
    solver = protocol['ground_truth_solver']
    for x0 in initial:
        evaluation = np.sort(rng.uniform(0, horizon, points))
        solution = solve_ivp(lambda t, x: field(dataset['id'], t, x), (0, horizon), x0,
                             dense_output=True, **solver)
        if not solution.success:
            raise RuntimeError(solution.message)
        if dataset['id'] == 'lotka_volterra':
            locations = np.sort(rng.choice(points, dataset['observation_points'], replace=False))
            observed = evaluation[locations]
        elif dataset['id'] == 'pendulum':
            observed = np.arange(dataset['observation_points']) * .1
        else:
            observed = np.sort(rng.uniform(0, horizon, dataset['observation_points']))
        y = solution.sol(observed).T[:, :dataset['observed_dimensions']]
        true = solution.sol(evaluation).T
        observed_times.append(observed); observed_states.append(y)
        test_times.append(evaluation); test_states.append(true[:, :dataset['observed_dimensions']]); physical.append(true)
    ids = rng.permutation(count)
    if count == 1:
        train_ids, test_ids = np.array([0]), np.array([0])
    else:
        split = int(count * protocol['train_trajectory_fraction'])
        train_ids, test_ids = np.sort(ids[:split]), np.sort(ids[split:])
    arrays = dict(observed_times=np.asarray(observed_times), observed_states=np.asarray(observed_states),
                  test_times=np.asarray(test_times), test_states=np.asarray(test_states),
                  physical_ground_truth=np.asarray(physical), initial_states=initial[:, :dataset['observed_dimensions']],
                  train_ids=train_ids, test_ids=test_ids)
    audit_case(arrays, dataset, count)
    return arrays


def audit_case(arrays, dataset, count):
    assert arrays['observed_states'].shape == (count, dataset['observation_points'], dataset['observed_dimensions'])
    assert all(np.isfinite(a).all() for a in arrays.values())
    assert all(np.all(np.diff(arrays[k], axis=1) > 0) for k in ['observed_times', 'test_times'])
    train, test = set(arrays['train_ids']), set(arrays['test_ids'])
    if count > 1:
        assert not train & test and train | test == set(range(count))
    else:
        assert train == test == {0}
        assert not np.isin(arrays['test_times'][0], arrays['observed_times'][0]).any()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/reproducibility/cfm_ts_original_protocol.v1.json')
    args = parser.parse_args()
    config_path = ROOT / args.config
    protocol = read(config_path)
    if sha(ROOT / protocol['paper_pdf']) != protocol['paper_sha256']:
        raise ValueError('Original paper changed')
    target = ROOT / protocol['storage_root']
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / protocol['manifest']
    if manifest_path.exists():
        existing = read(manifest_path)
        if existing['protocol_sha256'] != sha(config_path) or any(sha(ROOT / d['path']) != d['sha256'] for d in existing['cases']):
            raise ValueError('Existing raw simulation registration changed')
        print('Existing complete simulation dataset verified; no files overwritten')
        return
    cases = []
    for track in protocol['tracks']:
        for dataset in protocol['datasets']:
            for seed in protocol['seeds']:
                name = f"{track['id']}__{dataset['id']}__seed{seed}"
                path = target / (name + '.npz')
                if path.exists():
                    raise FileExistsError('Partial/raw files preserved: ' + str(path))
                arrays = generate_case(protocol, track, dataset, seed)
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                cases.append({'id': name, 'track': track['id'], 'dataset': dataset['id'], 'seed': seed,
                              'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                              'trajectory_count': len(arrays['initial_states']), 'train_trajectories': len(arrays['train_ids']),
                              'test_trajectories': len(arrays['test_ids']), 'observation_points': dataset['observation_points'],
                              'test_points_per_trajectory': protocol['evaluation_points'],
                              'observed_dimensions': dataset['observed_dimensions'],
                              'test_initial_time': 0, 'horizon': dataset['horizon'],
                              'split': 'new_times_same_trajectory' if dataset['id'] == 'pendulum' else 'held_out_complete_trajectories'})
                print(json.dumps({'generated': name, 'trajectories': len(arrays['initial_states'])}), flush=True)
    manifest = {'paper_id': protocol['paper_id'], 'citation': protocol['citation'], 'protocol': args.config,
                'protocol_sha256': sha(config_path), 'generator_sha256': sha(Path(__file__)), 'cases': cases,
                'raw_preserved': True, 'author_equivalence_certified': False, 'diagnostic_smoke': False,
                'explicit_interpretations': protocol['explicit_interpretations']}
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'cases': len(cases), 'manifest': protocol['manifest']}))


if __name__ == '__main__':
    main()
