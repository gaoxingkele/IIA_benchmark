"""Full real-data DSB source cache/gradient checks for every available leaveout."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as source

MODULE = 'scripts.flow_matching.preflight_dsb_singlecell_full_inputs_v1'
CONFIG = 'configs/reproducibility/fm_dsb_singlecell_all12_full_input_preflight.v1.json'
MODEL = 'configs/models/fm_dsb_singlecell_source_E4_components.v1.json'
ADAPTER = 'src/iia_benchmark/models/dsb_singlecell_source_components_v1.py'
INPUT = 'configs/reproducibility/fm_otcfm_source_cnf_all12_full_input_preflight.v1.json'


def registration():
    original = read(ROOT / INPUT)
    profile = read(ROOT / MODEL)
    paper = 'papers/literature/flow_matching/pdfs/ot_cfm.pdf'
    files = [MODULE.replace('.', '/') + '.py', MODEL, ADAPTER, INPUT,
             'tests/test_fm_dsb_singlecell_source_components_v1.py',
             'tests/test_fm_dsb_singlecell_full_input_preflight_v1.py',
             original['environment_lock'], paper,
             *[p.relative_to(ROOT).as_posix() for p in (ROOT / profile['source_root']).rglob('*.py')]]
    return dict(cases=original['cases'], required_full_input_cases=12, distinct_original_input_tracks=5,
                model_configuration=MODEL, source_root=profile['source_root'],
                source_runtime_queue=original['source_runtime_queue'], python=original['python'],
                environment_lock=original['environment_lock'], numerical_threads=2,
                settings=original['settings'], resource_lock=original['resource_lock'],
                source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(set(files))],
                seed=42, dimensions=5, train_val_test_split=original['train_val_test_split'],
                actual_cache_particles=10000, actual_cache_steps=20, actual_cache_rows=200000,
                actual_training_batch=512, gamma=0.001, mean_match=False,
                parameters_per_network=14261, network_count=2, optimizer_updates=0,
                full_training_models_pending=60, full_updates_per_model_pending=400040,
                source_gamma_to_paper_sigma_mapping_certified=False,
                original_DSB_2021_environment_reproduced=False, historical_paper_equivalence_certified=False,
                state_root='experiments/runs/fm_dsb_singlecell_all12_full_input_preflight_v1',
                artifact_root='F:/aicoding/IIA_Data/experiments/dsb_singlecell_all12_full_input_preflight_v1',
                original_scope=original['original_scope'], all_original_experiments_complete=False,
                boundary=profile['boundary'])


def verify(c):
    if c != registration():
        raise ValueError('Full DSB source/data/environment/cache/guard registration changed')


def worker(c):
    torch, pl, distribution, _, _ = source.source_runtime(read(ROOT / c['source_runtime_queue']))
    spec = importlib.util.spec_from_file_location('_iia_dsb_singlecell_components', ROOT / ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    _, Langevin, CacheLoader = adapter.original_components(ROOT / c['source_root'])
    records = []
    for case in c['cases']:
        data = case['data']
        path = ROOT / data['path']
        if sha(path) != data['sha256']:
            raise ValueError('Untouched complete single-cell input required')
        pl.seed_everything(c['seed'], workers=True)
        dm = distribution.CustomTrajectoryDataModule(
            system=str(path), train_val_test_split=tuple(c['train_val_test_split']),
            max_dim=5, whiten=True, batch_size=512, num_workers=0, pin_memory=False, seed=c['seed'])
        counts = {str(level): len(x) for level, x in zip(dm.ulabels, dm.timepoint_data)}
        if counts != data['days'] or sum(counts.values()) != data['cells'] or dm.dim != 5:
            raise ValueError('All original populations and first5 PCs required')
        orientation = adapter.endpoint_indices(len(dm.timepoint_data), case['leftout_timepoint'])
        def endpoint(index):
            part = dm.split_timepoint_data[index][0]
            # Torch Subset exposes exact author split indices,not first-N rows.
            return torch.as_tensor(dm.timepoint_data[index][list(part.indices)], dtype=torch.float32)
        later = endpoint(orientation['source_initial_index'])
        earlier = endpoint(orientation['source_final_index'])
        initial_dl = adapter.endpoint_cache_loader(later, 10000, c['seed'])
        final_dl = adapter.endpoint_cache_loader(earlier, 10000, c['seed'] + 1000)
        nets = {k: adapter.build_score_network(ROOT / c['source_root']) for k in ('f', 'b')}
        if set(map(id, nets['f'].parameters())) & set(map(id, nets['b'].parameters())):
            raise ValueError('DSB requires independently parameterized forward and backward networks')
        gammas = torch.full((20,), c['gamma'])
        langevin = Langevin(20, (5,), gammas, torch.distributions.Categorical(torch.ones(20)),
                            device='cpu', mean_final=earlier.mean(dim=0),
                            var_final=earlier.var(), mean_match=False)
        directions = []
        for direction in ('b', 'f'):
            cache = CacheLoader(direction, nets['f' if direction == 'b' else 'b'],
                                iter(initial_dl) if direction == 'b' else None,
                                num_batches=1, langevin=langevin, n=1, mean=None, std=None,
                                batch_size=10000, device='cpu', dataloader_f=iter(final_dl), transfer=True)
            if cache.data.shape != (200000, 2, 5) or cache.steps_data.shape != (200000, 1):
                raise ValueError('All10000particles and20original Langevin steps must be retained')
            if not torch.isfinite(cache.data).all() or not torch.isfinite(cache.steps_data).all():
                raise ValueError('Actual full source DSB cache must be finite')
            x, target, times = next(iter(torch.utils.data.DataLoader(cache, batch_size=512)))
            if len(x) != 512:
                raise ValueError('Original full512training batch required')
            # This is the original mean_match=False branch of IPFSequential.ipf_step.
            prediction = nets[direction](x, gammas.sum() - times)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                                 for p in nets[direction].parameters()):
                raise ValueError('Real full-batch source DSB objective and every gradient must be finite')
            directions.append(dict(direction=direction, original_cache_rows=len(cache), training_batch=512,
                                   training_objective=float(loss.detach()), gradients_finite=True, optimizer_updates=0))
            nets[direction].zero_grad(set_to_none=True)
            del prediction, loss, x, target, times, cache
            gc.collect()
        # Native backward SDE midpoint using the complete earlier endpoint test population.
        with torch.no_grad():
            complete_test = torch.as_tensor(dm.timepoint_data[orientation['backward_start_index']])
            trajectory, _, _ = langevin.record_langevin_seq(nets['b'], complete_test, ipf_it=1, sample=True)
        if trajectory.shape != (len(complete_test), 20, 5) or not torch.isfinite(trajectory).all():
            raise ValueError('All original earlier-endpoint observations must support native20-step backward sampling')
        records.append(dict(id=case['id'], full_data_cells=data['cells'], full_timepoint_counts=counts,
                            orientation=orientation, earlier_training_rows=len(earlier), later_training_rows=len(later),
                            every_endpoint_training_row_eligible=True, cache_sampling='fixed_seed_replacement',
                            actual_test_trajectory_shape=list(trajectory.shape), midpoint_shape=list(trajectory[:, 9, :].shape),
                            independent_networks=2, parameters_per_network=14261, directions=directions,
                            optimizer_updates=0, benchmark_performance_claim=False))
        write(ROOT / c['state_root'] / 'progress.json', dict(completed_cases=len(records), required_cases=12, current_id=case['id']))
        del nets, langevin, dm, initial_dl, final_dl, later, earlier, trajectory, complete_test
        gc.collect()
    out = Path(c['artifact_root'])
    out.mkdir(parents=True, exist_ok=False)
    exclusive(out / 'full12_source_cache_preflight.json', dict(
        passed=True, records=records, required_cases=12, configuration_sha256=sha(ROOT / CONFIG),
        optimizer_updates=0, full_training_complete=False, historical_paper_equivalence_certified=False,
        all_original_experiments_complete=False))


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades()
    base = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Full DSB preflight already owned')
    if (base / 'status.json').exists():
        owner.close()
        raise FileExistsError('Preserve all original DSB attempts')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
                 required_full_input_cases=12, all_original_experiments_complete=False)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / c['resource_lock'], c['settings'], update):
            verify(c)
            command = [c['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                update(dict(state='running_DSB_all12_full_input_preflight', active_pid=child.pid, exact_child_handle=handle))
                code, usage = capped_wait(child, c, api, update)
            report = Path(c['artifact_root']) / 'full12_source_cache_preflight.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
            exclusive(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                exact_child_handle=handle, report_sha256=sha(report) if report.exists() else None))
            update(dict(state='full12_source_preflight_passed_training_pending' if passed else 'preflight_failed_preserved', active_pid=None))
    finally:
        owner.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--register', action='store_true')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if a.register:
        exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    verify(c)
    if a.worker:
        worker(c)
    else:
        controller(c)


if __name__ == '__main__':
    main()
