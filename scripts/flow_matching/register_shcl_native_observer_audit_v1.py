"""Bind a diagnostic full64x110x25 observation check to all passed native audits."""
import json

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def main():
    predecessor='configs/runtime/fm_shcl_author_operators.v1.json'; config=native.read(native.ROOT/predecessor)
    report=native.ROOT/config['output_root']/'full_released_operator_audit.json'
    values=native.read(report)
    if not values['passed'] or values['full_operator_profiles']!=13: raise ValueError('All13 actual native operator profiles required')
    extra=[predecessor,report.relative_to(native.ROOT).as_posix(),'scripts/flow_matching/shcl_native_observer_v1.py',
        'scripts/flow_matching/audit_shcl_native_observer_v1.py','scripts/flow_matching/run_shcl_native_observer_audit_v1.py',
        'scripts/flow_matching/register_shcl_native_observer_audit_v1.py']
    config['source_receipts'].extend(dict(path=p,sha256=native.sha(native.ROOT/p)) for p in extra)
    config.update(state_root='experiments/runs/fm_shcl_native_observer_audit_v1',
        output_root='projects/flow_matching_research/results/2026-10-12/shcl_author_environment_v1/observer_fixture',
        diagnostic_scope=dict(full_input_shape=[64,110,25],synthetic_optimizer_updates=1,
            original_benchmark_optimizer_updates=0,synthetic_operator_only_not_benchmark_performance=True),
        boundary='This separate diagnostic observes a full-size native VAE batch and one Adam update, '
            'checks all parameters/state/loss/RNG against its unobserved counterpart, and checks full '
            'threshold/test array capture. Not an original-data training, not the49/364 experiment '
            'matrix, and not a benchmark metric. The upstream13 native profiles have already passed.')
    exclusive(native.ROOT/'configs/runtime/fm_shcl_native_observer_audit.v1.json',config)
    print(json.dumps(dict(full_input_shape=[64,110,25],original_benchmark_optimizer_updates=0)))


if __name__=='__main__':main()
