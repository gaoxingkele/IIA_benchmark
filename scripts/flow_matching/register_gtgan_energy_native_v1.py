"""Register the two original README Energy commands and complete native evidence."""
import argparse
from pathlib import Path

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.gtgan_energy_native_common_v1 import recipes, release_recipes, full_budget
from scripts.flow_matching.register_gtgan_native_commands_v1 import native_arguments

MODEL = 'configs/models/fm_gtgan_native_original_energy.v1.json'
DATA = 'configs/datasets/fm_gtgan_original_energy.v1.json'
PREP = 'configs/reproducibility/fm_gtgan_energy_runtime_preparation.2026-10-11_v1.json'
AUDIT = 'configs/reproducibility/fm_gtgan_energy_native_audit.2026-10-11_v3.json'
QUEUE = 'configs/experiments/fm_gtgan_native_original_energy_queue.v1.json'
PROOF = 'projects/flow_matching_research/results/2026-10-11/gtgan_energy_native_audit_v3/native_energy_data_and_model_audit.json'
ARTIFACT_ROOT = 'F:/aicoding/IIA_Data/experiments/gtgan_energy_native_runtime_v1'


def receipts(paths):
    return [dict(path=p, sha256=sha(ROOT / p)) for p in paths]


def write_once(path, value):
    if (ROOT / path).exists():
        raise FileExistsError('Preserve immutable registration: ' + path)
    write(ROOT / path, value)


def preparation():
    stock = read(ROOT / 'configs/models/fm_gtgan_native_original_stock.v1.json')
    source = ROOT / stock['source_root']
    cases = []
    for rate, arguments in recipes():
        parsed = native_arguments(source, 'GTGAN_energy.py', arguments)
        case = dict(id='gtgan_original_energy_missing_' + str(rate).replace('.', 'p'),
            missing_rate=rate, source_entry=(Path(stock['source_root']) / 'GTGAN_energy.py').as_posix(),
            source_entry_sha256=sha(source / 'GTGAN_energy.py'), original_seed=7777,
            author_arguments=arguments, parsed_original_arguments=parsed,
            embedding_training_steps=5000, joint_training_steps=parsed['max_steps'],
            native_evaluator_repeats=10, full_native_training_complete=False,
            boundary='Exact original README Energy command; hardcoded original seed7777, no pretrain reload during training')
        full_budget(case)
        cases.append(case)
    released = []
    for rate, arguments in release_recipes():
        case = dict(id='gtgan_original_energy_missing_' + str(rate).replace('.', 'p') + '__released_evaluation',
            missing_rate=rate, source_entry=(Path(stock['source_root']) / 'GTGAN_energy.py').as_posix(),
            source_entry_sha256=sha(source / 'GTGAN_energy.py'), original_seed=7777,
            author_arguments=arguments, parsed_original_arguments=native_arguments(source, 'GTGAN_energy.py', arguments),
            execution_mode='native_release_checkpoint_evaluation', full_native_training_complete=False,
            boundary='Exact original README Energy published generator/recovery checkpoint evaluation; zero generator training updates')
        full_budget(case)
        released.append(case)
    write_once(MODEL, dict(id='fm_gtgan_native_original_energy_v1', method='GT-GAN',
        task='irregular_to_regular_time_series_generation', source_commit=stock['source_commit'],
        source_root=stock['source_root'], dataset_config=DATA, cases=cases, release_evaluation_cases=released,
        implementation='Original GTGAN_energy.py:main and train, unchanged five-network original GT-GAN architecture',
        reproduction_status='exact_original_energy_commands_frozen_full_data_model_audit_pending_training_and_evaluation_pending',
        tests=['tests/test_fm_gtgan_energy_native_v1.py'], citations=stock['citations'],
        environment_lock='configs/reproducibility/gtgan_author_py38.lock.json',
        data_and_model_audit_config=AUDIT, author_equivalence_certified=False, performance_result=False,
        boundary='Original Energy0/50% README tasks. Full19712 overlapping24x28 windows, global normalization and whole-row masking; no grouped heldout generator split. Energy discriminator uses300 native steps, predictor5000; regular predicts last variable, irregular predicts all28. Other paper adopted configurations and robustness/transfer tasks remain separate required work. No strict TAB or historical author equivalence.'))
    raw = source / 'datasets/energy_data.csv'
    write_once(DATA, dict(id='gtgan_original_energy', task='time_series_generation',
        raw_file=raw.relative_to(ROOT).as_posix(), raw_sha256=sha(raw), raw_rows=19735,
        variables=28, sequence_length=24, complete_windows=19712,
        source_repository='https://github.com/Jinsung-Jeon/GT-GAN', source_commit=stock['source_commit'],
        raw_storage_preserved=True, native_loader=(Path(stock['source_root']) / 'time_dataset.py:TimeDataset').as_posix(),
        preprocessing='Original row reversal,global per-variable MinMax with1e-7,all stride-one windows including last,torch7777 window shuffle,whole raw-row removal seed56789 at0/50%; full original cubic spline coefficients',
        audit_config=AUDIT, citations=stock['citations']))
    prep = read(ROOT / 'configs/reproducibility/fm_gtgan_native_runtime_preparation.2026-10-11_v1.json')
    prep.update(artifact_root=ARTIFACT_ROOT,
        boundary='Full245 pinned original author assets preserved. Existing unused loader imports removed in isolated runtime plus one unused Energy import. Entire Energy main/train/network AST untouched; native CUDA training remains unchanged. Data/model proof only until full command finishes.')
    write_once(PREP, prep)
    dependencies = [MODEL, DATA, PREP, 'configs/reproducibility/gtgan_author_py38.lock.json',
        'scripts/flow_matching/prepare_gtgan_energy_runtime_v1.py',
        'scripts/flow_matching/prepare_gtgan_native_runtime_v1.py',
        'scripts/flow_matching/audit_gtgan_energy_native_v1.py',
        'scripts/flow_matching/gtgan_energy_native_common_v1.py',
        'scripts/flow_matching/register_gtgan_native_commands_v1.py',
        'scripts/flow_matching/audit_gtgan_native_models_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'tests/test_fm_gtgan_energy_native_v1.py']
    dependencies += [(Path(stock['source_root']) / n).as_posix() for n in
        ['GTGAN_energy.py', 'ctfp_tools.py', 'time_dataset.py', 'README.md', 'datasets/energy_data.csv',
         'metrics/discriminative_metrics.py', 'metrics/predictive_metrics.py']]
    write_once(AUDIT, dict(config_path=AUDIT, python='.venv/gtgan-author-py38/Scripts/python.exe',
        environment_lock='configs/reproducibility/gtgan_author_py38.lock.json', model_config=MODEL,
        source_root=stock['source_root'], runtime_root=ARTIFACT_ROOT + '/constructor_audit/runtime',
        output_directory=str(Path(PROOF).parent).replace('\\', '/'),
        artifact_root='F:/aicoding/IIA_Data/experiments/gtgan_energy_native_data_model_proof_v1',
        state_root='experiments/runs/fm_gtgan_energy_native_audit_v1',
        resource_lock='experiments/runs/fm_readonly_additional_checkpoint_cpu.lock',
        settings=dict(minimum_available_memory_bytes=16 * 2**30, minimum_commit_headroom_bytes=26 * 2**30,
            emergency_commit_headroom_bytes=16 * 2**30, maximum_worker_rss_bytes=4 * 2**30,
            maximum_worker_private_bytes=6 * 2**30), source_receipts=receipts(dependencies),
        boundary='Complete original Energy0/50% raw-data normalization/shuffle/whole-row masks and all cubic coefficients; CPU device-only preprocessing, exact full five-network native CPU constructors and strict released generator/recovery checkpoint loads. Not optimizer execution or full native generation. No performance or author-equivalence certification.'))
    print(dict(exact_energy_commands=2, full_windows_per_profile=19712, full_training_complete=False))


def full_queue():
    import psutil
    audit = read(ROOT / AUDIT)
    audit_state = read(ROOT / audit['state_root'] / 'status.json')
    if (ROOT / PROOF).exists():
        proof = read(ROOT / PROOF)
        if (not proof['passed'] or not proof['full_raw_data_processing_verified'] or proof['full_original_constructor_cases'] != 2
                or proof['performance_result'] or proof['author_equivalence_certified']):
            raise ValueError('Both full native data/model proofs are required before execution admission')
    else:
        parent = psutil.Process(audit_state['pid'])
        if parent.create_time() != audit_state['create_time'] or AUDIT not in parent.cmdline():
            raise ValueError('Queue must wait on the exact currently live full native audit')
    from scripts.flow_matching.prepare_gtgan_energy_runtime_v1 import prepare_energy
    model = read(ROOT / MODEL)
    prep = read(ROOT / PREP)
    jobs = []
    for case in model['release_evaluation_cases'] + model['cases']:
        runtime = ARTIFACT_ROOT + '/full_commands/' + case['id'] + '/runtime'
        prepare_energy(prep, runtime)
        identifier = case['id'] + '__native_command__seed7777'
        jobs.append(dict(id=identifier, case_id=case['id'], runtime_root=runtime,
            output_directory='experiments/runs/fm_gtgan_native_original_energy_v1/jobs/' + identifier))
    original = read(ROOT / 'configs/experiments/fm_gtgan_native_original_stock_queue.v1.json')
    dependencies = [MODEL, DATA, PREP, AUDIT,
        'configs/reproducibility/gtgan_author_py38.lock.json',
        'scripts/flow_matching/run_gtgan_energy_native_job_v1.py',
        'scripts/flow_matching/run_gtgan_energy_native_queue_v1.py',
        'scripts/flow_matching/gtgan_energy_native_common_v1.py',
        'scripts/flow_matching/gtgan_energy_metric_receipts_v1.py',
        'scripts/flow_matching/run_gtgan_native_job_v1.py',
        'scripts/flow_matching/register_gtgan_native_commands_v1.py',
        'scripts/flow_matching/full_gpu_memory_slot_v2.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'tests/test_fm_gtgan_energy_native_v1.py']
    extension = 'configs/reproducibility/gtgan_author_py38_runtime_extension.v2.lock.json'
    dependencies += [extension, 'tests/test_fm_gtgan_energy_report_compat_v2.py',
        'tests/test_fm_gtgan_energy_admission_v1.py',
        'scripts/flow_matching/audit_gtgan_energy_native_v2.py']
    write_once(QUEUE, dict(queue_path=QUEUE, python=original['python'], environment_lock=extension,
        model_config=MODEL, construction_proof=PROOF, cpu_threads=2,
        construction_audit_config=AUDIT, construction_audit_config_sha256=sha(ROOT / AUDIT),
        construction_audit_state_root=audit['state_root'],
        construction_audit_parent=dict(pid=audit_state['pid'], create_time=audit_state['create_time']),
        state_root='experiments/runs/fm_gtgan_native_original_energy_v1', resource_lock=original['resource_lock'],
        settings=original['settings'], jobs=jobs, source_receipts=receipts(dependencies),
        boundary='Execute all four exact original README Energy commands: two released-checkpoint evaluations (zero generator training updates) and two from-scratch full trainings without reducing5000 embedding or6000/8500 joint steps,batch128,seq24,hidden24,r/d2,28variables,full19712 windows or29 full training-periodic evaluations. Total31 full evaluations, each10 native300-update Energy discriminators and10 native5000-update predictors (regular last-variable vs irregular all-variable); save actual generator noise/times/RNG,complete outputs,all four trained network checkpoints and all620 original evaluator trained states/test predictions. Native seed7777 unchanged; evaluator repetitions are not generator seeds. Other adopted paper tasks remain required. No strict TAB or historical author-equivalence certification.'))
    print(dict(full_original_energy_jobs=len(jobs), full_original_evaluations=31, native_evaluator_trainings=620))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['preparation', 'queue'], required=True)
    a = p.parse_args()
    (preparation if a.phase == 'preparation' else full_queue)()


if __name__ == '__main__':
    main()
