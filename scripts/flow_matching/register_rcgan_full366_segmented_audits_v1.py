"""Register a complete, provenance-bound audit of both native seed1104 segments."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha
from scripts.flow_matching.rcgan_segmented_checkpoint_evidence_v1 import (
    TRACE_EPOCHS, MISSING_TRACE_EPOCHS, verify_manifest, verify_registration)


CHAIN = 'configs/reproducibility/fm_rcgan_seed1104_full366_random_order_chain.2026-10-12_v3.json'
STATE = 'configs/reproducibility/fm_rcgan_seed1104_full366_actual358_trace_resume_state.2026-10-12_v4.json'
MANIFEST = 'configs/reproducibility/fm_rcgan_seed1104_full366_segmented_evidence.2026-10-12_v1.json'
RESULT = 'projects/flow_matching_research/results/2026-10-12/rcgan_seed1104_full366_segmented_audits_v1'


def pin(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=sha(path))


def register():
    import psutil
    prior_chain_path = 'configs/reproducibility/fm_rcgan_seed1104_full84_random_order_chain.2026-10-11_v2.json'
    prior_state_path = 'configs/reproducibility/fm_rcgan_seed1104_full84_actual81_trace_resume_state.2026-10-11_v3.json'
    interrupted_config = 'configs/reproducibility/fm_rcgan_seed1104_full416_state_recovery.2026-10-11_v2.json'
    old_chain, old_state, recovery = [read(ROOT / p) for p in (prior_chain_path, prior_state_path, interrupted_config)]
    queue = read(ROOT / old_chain['queue'])
    canonical = recovery['canonical_job']
    first, second = ROOT / canonical['output_directory'], ROOT / recovery['recovery_job']['output_directory']
    terminal = ROOT / recovery['state_root'] / 'receipt.json'
    owner_state = read(ROOT / recovery['state_root'] / 'status.json')
    old_pid, old_ct = 17476, 1791740026.1733491
    if psutil.pid_exists(old_pid) and psutil.Process(old_pid).create_time() == old_ct:
        raise ValueError('Never snapshot the evidence of a live native training worker')
    if owner_state['active_pid'] is not None or read(terminal)['exit_code'] == 0:
        raise ValueError('Require the actual terminal failed recovery, not an observation timeout')
    view = Path('F:/aicoding/IIA_Data/experiments/rcgan_seed1104_full366_readonly_evidence_v1')
    for target in (view, ROOT / CHAIN, ROOT / STATE, ROOT / MANIFEST, ROOT / RESULT):
        if target.exists():
            raise FileExistsError('Preserve previous registrations and every evidence view: ' + str(target))
    view.mkdir(parents=True)
    sources, derived = {}, []
    def add_source(path):
        item = pin(path)
        sources[item['path']] = item
    def link(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        if not os.path.samefile(source, destination) or sha(source) != sha(destination):
            raise ValueError('Read-only native evidence links must preserve exact source bytes')
        add_source(source)
        derived.append(pin(destination))
    for epoch in range(366):
        source_root = first if epoch < 84 else second
        directory = source_root / 'native_checkpoints' / ('epoch_%03d' % epoch)
        files = sorted(p for p in directory.iterdir() if p.is_file())
        expected = {'checkpoint', 'model.data-00000-of-00001', 'model.index', 'model.meta',
                    'numpy_random_state.npz', 'python_random_state.json'}
        if {p.name for p in files} != expected:
            raise ValueError('Require every full checkpoint/optimizer/meta and both RNG files')
        for source in files:
            link(source, view / 'native_checkpoints' / directory.name / source.name)
    link(first / 'full_native_split_indices.npz', view / 'full_native_split_indices.npz')
    link(first / 'experiments/settings' / (canonical['id'] + '.txt'),
         view / 'experiments/settings' / (canonical['id'] + '.txt'))
    for root, job in ((first, canonical), (second, recovery['recovery_job'])):
        for source in sorted((root / 'experiments/parameters').glob(job['id'] + '_*.npy')):
            epoch = int(source.stem.rsplit('_', 1)[1])
            if 0 <= epoch < 366 and (epoch < 84) == (root == first):
                link(source, view / 'experiments/parameters' / (canonical['id'] + '_' + str(epoch) + '.npy'))
    header = 'epoch time D_loss G_loss mmd2 that pdf real_pdf'
    joined, provenance = [], []
    for root, job, required in ((first, canonical, list(range(81))),
                                (second, recovery['recovery_job'], list(range(84, 361)))):
        path = root / 'experiments/traces' / (job['id'] + '.trace.txt')
        lines = path.read_text(encoding='utf-8').splitlines()
        if lines[0] != header or [int(line.split()[0]) for line in lines[1:]] != required:
            raise ValueError('Only actually persisted trace rows may enter the derived view')
        joined.extend(lines[1:])
        provenance.append(dict(source=str(path), epochs=required, time_semantics='actual_source_segment_elapsed_seconds'))
        add_source(path)
    trace = view / 'experiments/traces' / (canonical['id'] + '.trace.txt')
    trace.parent.mkdir(parents=True)
    trace.write_text(header + '\n' + '\n'.join(joined) + '\n', encoding='utf-8')
    derived.append(pin(trace))
    progress = deepcopy(read(second / 'progress.json'))
    progress.update(job=canonical, derived_readonly_evidence_view=True)
    write(view / 'progress.json', progress)
    derived.append(pin(view / 'progress.json'))
    for p in (ROOT / prior_chain_path, ROOT / prior_state_path, ROOT / interrupted_config, terminal,
              ROOT / recovery['state_root'] / 'status.json', second / 'progress.json',
              second / 'completed_full_epochs.json', second / 'completed_full_evaluators.json',
              second / 'actual_full_resume_restoration.json', first / 'completed_full_evaluators.json',
              ROOT / old_state['output_directory'] / 'checkpoint_resume_state_audit.json',
              ROOT / old_state['state_root'] / 'receipt.json'):
        add_source(p)
    manifest = dict(captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job=canonical,
        interrupted_job=recovery['recovery_job'], completed_original_epochs=366,
        remaining_original_epochs=134, required_original_epochs=500,
        persisted_trace_epochs=list(TRACE_EPOCHS), missing_trace_epochs=list(MISSING_TRACE_EPOCHS),
        trace_provenance=provenance, time_is_not_reconstructed_as_one_continuous_clock=True,
        evidence_view=str(view), prior_chain_configuration=prior_chain_path,
        prior_state_configuration=prior_state_path,
        prior_checkpoint_proof=str(ROOT / old_state['output_directory'] / 'checkpoint_resume_state_audit.json'),
        first_resume_restoration=str(second / 'actual_full_resume_restoration.json'),
        interrupted_progress=str(second / 'progress.json'), failed_terminal_receipt=str(terminal),
        completed_epochs=str(second / 'completed_full_epochs.json'),
        completed_evaluators=str(second / 'completed_full_evaluators.json'),
        first_completed_evaluators=str(first / 'completed_full_evaluators.json'),
        source_receipts=list(sources.values()), derived_receipts=derived,
        original_scope=old_chain['original_scope'], benchmark_performance=False,
        additional_training_seed_slots=0, old_worker_identity_verified_not_live=dict(pid=old_pid, create_time=old_ct),
        view_is_recovery_evidence_not_completed500epoch_result=True)
    verify_manifest(manifest)
    write(ROOT / MANIFEST, manifest)
    code_paths = ['scripts/flow_matching/' + name for name in (
        'rcgan_segmented_checkpoint_evidence_v1.py', 'register_rcgan_full366_segmented_audits_v1.py',
        'audit_rcgan_random_order_chain_v1.py', 'audit_rcgan_random_order_chain_v3.py',
        'audit_rcgan_checkpoint_resume_state_v1.py', 'audit_rcgan_checkpoint_resume_state_v4.py')]
    chain = deepcopy(old_chain)
    chain.update(completed_original_epochs=366, segmented_evidence_manifest=MANIFEST,
        output_directory=RESULT + '/random_order_chain',
        artifact_root='F:/aicoding/IIA_Data/experiments/rcgan_seed1104_full366_random_order_chain_v3',
        state_root='experiments/runs/fm_rcgan_seed1104_full366_random_order_chain_v3',
        failed_terminal_receipt=str(terminal),
        source_receipts=[pin(ROOT / p) for p in code_paths + [MANIFEST]],
        boundary='Audit every366 source epoch and complete4984-point evaluator, all RNG states and full36000 image order. Original500 budget,134 remaining. Eight actual trace gaps preserved. No native GAN training or performance credit.')
    write(ROOT / CHAIN, chain)
    verify_registration(chain)
    state = deepcopy(old_state)
    state.update(random_chain_configuration=CHAIN,
        output_directory=RESULT + '/checkpoint_state',
        state_root='experiments/runs/fm_rcgan_seed1104_full366_actual358_trace_resume_state_v4',
        source_receipts=[pin(ROOT / p) for p in code_paths + [MANIFEST, CHAIN]],
        boundary='After successful full366 chain: restore every366 real96929-parameter native GAN/optimizer checkpoint, reproduce358 persisted float32 losses, recompute all366 losses, preserve eight absent times, replay latest4984 generated sequences and repeated original D/G transition. No continuation until native proof passes. Both trace clocks retain their actual segment semantics.')
    write(ROOT / STATE, state)
    report = dict(configuration=MANIFEST, configuration_sha256=sha(ROOT / MANIFEST),
        completed_epochs=366, persisted_trace_rows=358, missing_trace_epochs=list(MISSING_TRACE_EPOCHS),
        required_original_epochs=500, remaining_original_epochs=134,
        checkpoint_directories=366, checkpoint_files=366 * 6,
        checkpoint_source_bytes_unchanged=True, hardlinks_not_duplicate_raw_datasets=True,
        all366_native_random_order_and_state_audits_complete=False,
        additional_training_seed_slots=0, benchmark_performance=False,
        original_scope=old_chain['original_scope'], all_original_experiments_complete=False)
    write(ROOT / RESULT / 'actual_segmented_evidence_registration.json', report)
    return report


if __name__ == '__main__':
    print(register())
