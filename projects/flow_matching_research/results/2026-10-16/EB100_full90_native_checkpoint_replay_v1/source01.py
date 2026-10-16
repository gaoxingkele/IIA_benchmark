"""Replay complete original single-cell models and independently solve full W1.

Native source equations/data/ODE/test loops remain unchanged. The distance
observer retains every full test population and requests solver diagnostics,
then independently builds double-precision Euclidean costs with SciPy.
"""
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import run_otcfm_EB100_code_referenced_native90_v1 as native
from scripts.flow_matching.otcfm_EB100_result_integrity_v1 import inspect

MODULE = 'scripts.flow_matching.audit_otcfm_EB100_full90_replay_v1'
CONFIG = 'configs/reproducibility/fm_otcfm_EB100_full90_checkpoint_replay.v1.json'


def interior_geometry(timepoint_sizes):
    if len(timepoint_sizes) != 5 or any(type(n) is not int or n<=0 for n in timepoint_sizes):
        raise ValueError('Complete five positive original EB100 timepoint populations required')
    return [(timepoint_sizes[i],timepoint_sizes[i+1]) for i in range(len(timepoint_sizes)-2)]


def registration():
    queue = read(ROOT / native.QUEUE); native.verify(queue)
    paths = [native.QUEUE, MODULE.replace('.', '/') + '.py',
        'scripts/flow_matching/otcfm_EB100_result_integrity_v1.py',
        'tests/test_fm_otcfm_EB100_full90_replay_v1.py',
        'tests/test_fm_otcfm_EB100_result_integrity_v1.py']
    return dict(original_queue=native.QUEUE, original_queue_sha256=sha(ROOT / native.QUEUE),
        required_full_native_cases=90, required_case_ids=[j['id'] for j in queue['jobs']],
        python=queue['python'], state_root='experiments/runs/fm_otcfm_EB100_full90_checkpoint_replay_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/otcfm_EB100_full90_checkpoint_replay_v1',
        resource_lock='experiments/runs/fm_otcfm_singlecell_readonly_replay_cpu.lock',
        numerical_threads=2, poll_seconds=10, settings=queue['settings'],
        native_metric_tolerance=dict(rtol=2e-6, atol=2e-6),
        independent_W1_comparison='LP objective perturbation bound from every entry of full native float32 and independent SciPy float64 Euclidean matrices, plus1e-8 relative solver rounding; both solvers must report OPTIMAL.',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in sorted(paths)],
        original_scope=queue['original_scope'], citation=queue['citation'],
        independent_source_replay_queue_preserved='configs/reproducibility/fm_otcfm_singlecell_full120_checkpoint_replay.v2.json',
        EB100_complete_metric_count=44,
        preserved_full150_replay_queue='configs/reproducibility/fm_otcfm_EB_CITE_full150_checkpoint_replay.v1.json',
        boundary='All90 original EB100 full source models require successful original process guard, full training/early-stop budget, '
            'all original artifacts, exact package lock and validation-selected weights. Reexecute original complete '
            'test/auxiliary archive paths; retain all3 original EB100 complete interior-timepoint prediction/truth arrays. '
            'Check every native scalar within2e-6 and independently solve full uniform-mass1-Wasserstein with '
            'SciPy costs/POT exact LP, certifying cost perturbation bound and OPTIMAL codes. '
            'Original90 source stores complete training and restored selected-checkpoint updates separately; both counters, all original epoch histories and latest retained validation weights are checked without source overwrite. Only the actual registered6000-second paper cap may end its final epoch partway. No extra model training, changed seed, dataset subsampling, native tolerance reduction or automatic failed retry. '
            'Exact original2020 public EB100 Git blob and code reference verified; historical training recipe/weight identity and strictTAB claims remain false. Existing120 and150 source/replay conditions and all other45-paper work remain required.',
        all_original_experiments_complete=False)


def verify(config):
    if config != registration(): raise ValueError('Exact original EB100 full90 replay source/data/resource contract required')
    return read(ROOT / config['original_queue'])


def worker(config, queue, job):
    import numpy as np
    from scipy.spatial.distance import cdist
    result, provenance = inspect(queue, job)
    torch, pl, dm, model = native.instantiate_case(queue, job)
    checkpoint = torch.load(result['best_checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, checkpoint['state_dict'][name]):
            raise ValueError('Every original learned tensor and shared state alias must remain exact')
    best_epoch = int(checkpoint['epoch']) + 1
    val = [r for r in result['validation_history'] if r['finite']]
    best = min(r['validation_loss'] for r in val)
    selected = next(r for r in val if r['epoch'] == best_epoch)
    if selected['validation_loss'] != best: raise ValueError('The original best checkpoint must minimize all recorded validation losses')
    batches = len(dm.train_dataloader()); updates = int(checkpoint['global_step'])
    if updates != result['posttest_selected_checkpoint_optimizer_updates']:
        raise ValueError('Actual original selected-checkpoint update counter must match its separate source field')
    if updates != best_epoch * batches: raise ValueError('Exact selected-checkpoint epoch/update pair required')
    optimizers = checkpoint['optimizer_states']
    if len(optimizers) != 1: raise ValueError('One actual original AdamW optimizer required')
    optimizer = optimizers[0]
    for group in optimizer['param_groups']:
        if group['lr'] != .001 or group['weight_decay'] != 1e-5:
            raise ValueError('Actual original optimizer hyperparameters must remain frozen')
    if (sum(s['exp_avg'].numel() for s in optimizer['state'].values()) != 9093
            or any(int(s['step']) != updates for s in optimizer['state'].values())):
        raise ValueError('Every original optimizer moment and update counter required')
    last = torch.load(Path(job['artifact_directory']) / 'checkpoints/last.ckpt', map_location='cpu')
    last_epoch = int(last['epoch']) + 1
    if (last_epoch != max(row['epoch'] for row in result['validation_history'])
            or int(last['global_step']) != result['epoch_history'][last_epoch-1]['optimizer_updates']):
        raise ValueError('Latest retained validation checkpoint must independently confirm actual training history')
    del last
    out = Path(config['artifact_root']) / Path(job['artifact_directory']).name
    out.mkdir(parents=True, exist_ok=False); os.chdir(out)
    import src.models.cfm_module as source
    import src.models.components.optimal_transport as transport
    original_distance = source.compute_distribution_distances
    original_emd = transport.pot.emd2
    observations = []
    geometry = interior_geometry([len(x) for x in dm.timepoint_data])
    interior_timepoints = len(geometry)
    def observe(predictions, truths):
        if observations or not isinstance(predictions, list) or not isinstance(truths, list) or len(predictions) != interior_timepoints or len(truths) != interior_timepoints:
            raise ValueError('All three EB or two CITE complete native interior-timepoint distributions are required')
        arrays = []
        for i, (prediction, truth) in enumerate(zip(predictions, truths)):
            a, b = prediction.detach().cpu().numpy(), truth.detach().cpu().numpy()
            if a.shape != (geometry[i][0], 5) or b.shape != (geometry[i][1], 5) or not np.array_equal(b, dm.timepoint_data[i+1]):
                raise ValueError('Every complete original predicted and observed cell must be retained')
            for name, values in [('prediction', a), ('truth', b)]:
                np.save(out / f't{i+1}_{name}.npy', values, allow_pickle=False)
            arrays.append((a, b))
        solves = []; counter = [0]
        def observed_emd(a, b, costs, **kwargs):
            index = counter[0]; counter[0] += 1
            if index >= 2*interior_timepoints or kwargs.get('log'):
                raise ValueError('Every unchanged original interior-timepoint exact W1/W2 solve required')
            cost, log = original_emd(a, b, costs, log=True, **kwargs)
            if int(log['result_code']) != 1 or log['warning'] is not None:
                raise ValueError('Original full-distribution LP must reach certified OPTIMAL, not iteration-limited output')
            row = dict(timepoint=index // 2 + 1, power=index % 2 + 1, full_cost_shape=list(costs.shape),
                native_LP_objective=float(cost), native_LP_result_code=int(log['result_code']))
            if index % 2 == 0:
                x, y = arrays[index // 2]
                independent = cdist(x, y, metric='euclidean')
                maximum = 0.
                for start in range(0, len(x), 256):
                    maximum = max(maximum, float(np.max(np.abs(independent[start:start+256] - costs[start:start+256]))))
                other, other_log = original_emd(a, b, independent, log=True, **kwargs)
                if int(other_log['result_code']) != 1 or other_log['warning'] is not None:
                    raise ValueError('Independent full SciPy-cost W1 LP must reach OPTIMAL')
                absolute = abs(float(other) - float(cost))
                bound = maximum + 1e-8 * max(1., abs(float(cost)), abs(float(other)))
                if absolute > bound:
                    raise ValueError('Independent full W1 differs beyond measured all-entry cost perturbation bound')
                row.update(independent_SciPy_float64_W1=float(other), independent_LP_result_code=int(other_log['result_code']),
                    maximum_full_cost_entry_difference=maximum, objective_absolute_difference=absolute,
                    LP_objective_perturbation_bound=bound, full_costs_independently_recomputed=True)
                del independent
            solves.append(row)
            return cost
        transport.pot.emd2 = observed_emd
        try: names, distances = original_distance(predictions, truths)
        finally: transport.pot.emd2 = original_emd
        if counter[0] != 2*interior_timepoints: raise ValueError('Every full native W1/W2 timepoint metric required')
        observations.append(dict(solves=solves, original_unlogged_metrics=dict(zip(names, map(float, distances)))))
        return names, distances
    source.compute_distribution_distances = observe
    try:
        trainer = pl.Trainer(default_root_dir=out, accelerator='cpu', devices=1, precision=32,
            logger=False, enable_progress_bar=False, enable_checkpointing=False)
        values = trainer.test(model=model, datamodule=dm)
    finally:
        source.compute_distribution_distances = original_distance; transport.pot.emd2 = original_emd
    if len(values) != 1 or set(values[0]) != set(result['test_metrics']) or len(observations) != 1:
        raise ValueError('Every original full test scalar and exactly one original all-distribution evaluation required')
    tolerance = config['native_metric_tolerance']
    if any(not math.isfinite(v) or not math.isclose(v, result['test_metrics'][k], rel_tol=tolerance['rtol'], abs_tol=tolerance['atol'])
            for k, v in values[0].items()):
        raise ValueError('Actual complete native checkpoint reinference must match every source scalar within unchanged2e-6')
    if torch.cuda.is_initialized(): raise ValueError('CPU-only native replay must not create a CUDA context')
    # Recheck immutable training artifacts after the whole readonly replay.
    again, proof = inspect(queue, job)
    if again != result or proof != provenance: raise ValueError('Original artifacts changed during replay')
    files = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    exclusive(out / 'full_checkpoint_replay.json', dict(status='passed', id=job['id'], source_provenance=provenance,
        full_training_epochs=result['actual_completed_epochs'], full_training_optimizer_updates=result['epoch_history'][-1]['optimizer_updates'],
        original_posttest_selected_checkpoint_updates=result['posttest_selected_checkpoint_optimizer_updates'],
        source_total_and_selected_optimizer_counters_separately_verified=True,
        selected_checkpoint_epoch=best_epoch, selected_checkpoint_updates=updates, all_learned_tensors_and_optimizer_moments_checked=True,
        complete_interior_timepoint_cost_shapes=[list(x) for x in geometry],
        full_native_test_reinference_passed=True, full_test_population_distances_independently_recomputed=observations,
        native_metrics=values[0], original_metrics=result['test_metrics'], native_metric_tolerance=tolerance,
        all_full_test_arrays_and_artifacts=files, extra_model_training_updates=0, cuda_context_initialized=False,
        historical_paper_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False,
        configuration_sha256=sha(ROOT / CONFIG), original_scope=config['original_scope']))


def controller(config, queue):
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value: None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Full90 original EB100 native model replay already owned')
    done = []; failed = {}; state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value):
        state.update(value, verified_ids=done, verified_count=len(done), failed_preserved=failed,
            required_full_native_cases=90, captured_utc=datetime.now(timezone.utc).isoformat(), all_original_experiments_complete=False)
        write(base / 'status.json', state)
    try:
        while len(done) + len(failed) < 90:
            available = False
            for job in queue['jobs']:
                if job['id'] in done or job['id'] in failed: continue
                key = Path(job['artifact_directory']).name
                original_guard = ROOT / queue['state_root'] / 'attempts' / key / 'guard_receipt.json'
                if not original_guard.exists(): continue
                if not read(original_guard)['passed']:
                    failed[job['id']] = 'Original training/test failed; preserved, not replayed or retried'; continue
                available = True; attempt = base / 'attempts' / key
                try:
                    if attempt.exists(): raise FileExistsError('Preserve all earlier replay attempts; no automatic failed retry')
                    with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
                        attempt.mkdir(parents=True, exist_ok=False)
                        command = [config['python'], '-B', '-X', 'utf8', '-u', '-m', MODULE, '--job-id', job['id']]
                        with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg', WANDB_MODE='disabled'),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                            handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                            update(dict(state='running_full_native_checkpoint_replay', active_pid=child.pid, exact_child_handle=handle, active_job=job['id']))
                            code, usage = capped_wait(child, config, api, update)
                        path = Path(config['artifact_root']) / key / 'full_checkpoint_replay.json'
                        passed = code == 0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['status'] == 'passed'
                        exclusive(attempt / 'guard_receipt.json', dict(passed=passed, exit_code=code, **usage,
                            exact_child_handle=handle, full_replay_report_sha256=sha(path) if path.exists() else None))
                        if not passed: raise ValueError('Full original replay failed; all partials preserved')
                        done.append(job['id'])
                except Exception as error: failed[job['id']] = str(error)
                update(dict(state='between_full_native_replays', active_pid=None))
            update(dict(state='waiting_actual_full_training_test_results' if not available else 'between_full_native_replays', active_pid=None))
            time.sleep(config['poll_seconds'])
        update(dict(state='all90_replay_attempts_terminal_other_original_work_remains', active_pid=None))
    finally: owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--job-id')
    args = parser.parse_args()
    if args.register: exclusive(ROOT / CONFIG, registration()); return
    config = read(ROOT / CONFIG); queue = verify(config)
    if args.job_id: worker(config, queue, next(j for j in queue['jobs'] if j['id'] == args.job_id))
    else: controller(config, queue)


if __name__ == '__main__': main()
