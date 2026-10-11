"""Preregister full WT uncertainty/paired robustness and guarded automatic review."""
from datetime import datetime, timezone
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, verify_receipts
from scripts.flow_matching.register_flow_mismatching_full_image_evaluation_v1 import create, receipt
from scripts.flow_matching.wt_full_cohort_summary_contract_v1 import METRICS, UNCERTAINTY, validate_configuration


def main():
    score_path='configs/experiments/fm_wt_full_image_evaluation_queue.v1.json'
    score=read(ROOT/score_path);verify_receipts(score['source_receipts'])
    config_path='configs/reproducibility/fm_wt_full_cohort_summary.2026-10-11_v1.json'
    sources=[score_path,score['environment_configuration'],
        'scripts/flow_matching/register_wt_full_cohort_summary_v1.py',
        'scripts/flow_matching/wt_full_cohort_summary_contract_v1.py',
        'scripts/flow_matching/summarize_wt_full_image_cohort_v1.py',
        'scripts/flow_matching/wt_full_image_evaluation_contract_v1.py',
        'scripts/flow_matching/flow_mismatching_full_training_contract_v1.py',
        'scripts/flow_matching/run_flow_mismatching_full_image_evaluation_v1.py',
        'scripts/flow_matching/audit_cfm_full_arrays_v1.py',
        'tests/test_fm_wt_full_cohort_summary_v1.py']
    config=dict(schema_version=1,configuration_path=config_path,score_queue=score_path,
        output_directory='data/public_datasets/flow_matching/experiments/wt_full_cohort_summary_v1',
        uncertainty=UNCERTAINTY,metric_names=list(METRICS),required_case_seed_evaluations=30,
        original_scope=score['original_scope'],source_receipts=[receipt(p) for p in sources],
        additional_training_seed_slots=0,author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False,
        boundary='All30 fully verified native test scores; five independent original2000epoch training seeds per fixed step. Related-source sensitivity and exploratory paper differences; no best-step selection or target-author equivalence.')
    validate_configuration(config)
    created=[create(config_path,config)]
    parent_path='configs/runtime/fm_wt_full_image_evaluation_queue.v1.json'
    parent=read(ROOT/parent_path)
    runtime_sources=[config_path,parent_path,
        'scripts/flow_matching/run_wt_full_cohort_summary_queue_v1.py',
        'scripts/flow_matching/wt_full_cohort_summary_contract_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/run_light_controller.py','scripts/flow_matching/run_tsad_queue.py',
        'scripts/flow_matching/heavy_resources.py','scripts/flow_matching/heavy_scheduler_v2.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py']
    runtime_path='configs/runtime/fm_wt_full_cohort_summary_queue.v1.json'
    keys=['minimum_available_memory_bytes','minimum_commit_headroom_bytes',
        'emergency_commit_headroom_bytes','cpu_threads','controller_yield_seconds']
    runtime=dict(schema_version=1,summary_configuration=config_path,python=parent['python'],
        settings={k:parent['settings'][k] for k in keys},resource_lock='experiments/runs/fm_heavy_recovery_cpu.lock',
        lane_lock='experiments/runs/fm_wt_full_cohort_summary_v1/queue.lock',
        state_root='experiments/runs/fm_wt_full_cohort_summary_v1',
        source_receipts=[receipt(p) for p in runtime_sources],
        cuda_context_allowed=False,heavy_libraries_imported_by_controller=False,
        boundary='Separate summary/review CPU workers only, unchanged existing memory/commit gates, no GPU reservation.')
    created.append(create(runtime_path,runtime))
    evidence='projects/flow_matching_research/results/2026-10-11/wt_full_cohort_summary_registration_v1'
    created.append(create(evidence+'/registration.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=receipt(config_path),runtime=receipt(runtime_path),score_queue=receipt(score_path),
        full_required_verified_score_cases=30,independent_original_training_seeds=5,
        aggregate_metric_rows=42,per_category_metric_rows=630,paired_metric_rows=84,
        predeclared_reference_steps=UNCERTAINTY['paired_reference_steps'],created_files=created.copy(),
        original_scope=score['original_scope'],new_training_seed_slots=0,new_measured_seed_slots=0,
        native_benchmark_performance=False,all_original_experiments_complete=False)))
    print(dict(full_native_cases_required=30,training_seed_unit=5,category_rows=630,paired_rows=84))


if __name__=='__main__':main()
