"""Audit every reviewed GiFlow test array and correct branch-specific boundaries."""
import argparse
from datetime import datetime, timezone

from scripts.flow_matching import audit_giflow_full_arrays_v2 as arrays
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def validate_selection(result):
    if (result['job']['track'] != 'reviewed_corrected' or result['selection']['test_used_in_validation']
            or result['selection']['tested_weights_match_validation_checkpoint'] is not True
            or result['selection']['actually_evaluated_weights_saved'] != 'evaluated_model.pt'
            or result['data_binding']['evaluation_targets_in_input'] != 0
            or result['data_binding']['timestamp_intersections'] != 0):
        raise ValueError('Reviewed native full test and validation-selected weights required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    target = ROOT / config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve full corrected array audit attempts')
    proof = arrays.audit(config)
    queue = read(ROOT / config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    result = read(ROOT / job['output_directory'] / 'result.json')
    validate_selection(result)
    proof.update(captured_utc=datetime.now(timezone.utc).isoformat(), selection=result['selection'],
        actual_evaluated_checkpoint=str(ROOT / job['output_directory'] / 'evaluated_model.pt'),
        actual_evaluated_checkpoint_sha256=sha(ROOT / job['output_directory'] / 'evaluated_model.pt'),
        underlying_full_array_implementation='scripts/flow_matching/audit_giflow_full_arrays_v2.py',
        test_used_in_validation=False, metadata_boundary_clarification_only=True,
        boundary='Every saved reviewed native masked prediction and target checked with complete BNLC geometry in float64 under original reduction tolerance. Native unweighted batch means/MAPE cutoff1e-6 retained. Actual validation-selected weights and no test-in-validation, persistent EMA branch verified. Full-array audit alone is not independent model replay. One of five original trained seeds; no author/TAB equivalence certification.')
    target.mkdir(parents=True)
    write(target / 'full_array_audit.json', proof)
    write(target / 'validation.json', dict(passed=True, source_receipts=config['source_receipts'],
        config=args.config, config_sha256=sha(ROOT / args.config),
        outputs={'full_array_audit.json': sha(target / 'full_array_audit.json')}))
    print(dict(epochs=proof['actual_full_training_epochs'], updates=proof['actual_optimizer_updates'],
               full_batches=proof['full_test_batches'], targets=proof['masked_targets_with_window_repetitions'],
               independently_recomputed_metrics=proof['independently_recomputed_metrics']))


if __name__ == '__main__':
    main()
