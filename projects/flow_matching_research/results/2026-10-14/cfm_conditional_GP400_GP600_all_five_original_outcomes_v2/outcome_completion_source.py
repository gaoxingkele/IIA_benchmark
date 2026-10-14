"""Complete all five original budget-pair outcomes without waiting for failed metrics.

Reuse the frozen metadata coordinator and change only outcome readiness/labels.
Actual numerical training and replay workers are never replaced or stopped.
"""
import ast
import inspect
from scripts.flow_matching import audit_cfm_gp_conditional_budget_outcomes_v2 as outcomes
from scripts.flow_matching import audit_cfm_gp_conditional_five_seed_budgets_v1 as planned

FAILED = 'cfm_ts_original__appendix_d__gp_gaussian_conditional_e600__sine__seed1104'


def idle_handoff_allowed(role, state, output_exists, descendants):
    return (role == 'original' and state.get('state') == 'waiting_every_original_full_seed_pair'
        and state.get('active_pid') is None and not state.get('shared_lock_owned')
        and state.get('required_complete_models') == 10
        and FAILED in state.get('missing_complete_model_ids', [])
        and all(planned.previous.cold.handoff.allowed_console_child(p['name'], p['exe']) for p in descendants))


def select_outcomes(records, pairs):
    if pairs != planned.identities():
        raise ValueError('All original five400/600 pairs remain required')
    indexed = {r['original_canonical_id']: r for r in records}
    if len(indexed) != len(records) or FAILED in indexed:
        raise ValueError('Keep unique canonical models and the preserved original failure')
    missing = [i for pair in pairs for i in pair if i != FAILED and i not in indexed]
    return (None if missing else True), missing


def metadata_main_ast():
    tree = ast.parse(inspect.getsource(planned.previous.main))
    labels = dict(waiting_every_original_full_seed_pair='waiting_every_original_full_outcome_certificate',
        auditing_all_ten_full_models='auditing_all_five_full_attempt_outcomes',
        completed_full_five_seed_pair_analysis='completed_all_original_full_outcomes_with_preserved_failure')
    counts = dict.fromkeys(labels, 0)
    keywords = dict(required_complete_models='required_original_model_outcomes',
        missing_complete_model_ids='missing_terminal_outcome_ids')
    keyword_counts = dict.fromkeys(keywords, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in labels:
            counts[node.value] += 1
            node.value = labels[node.value]
        if isinstance(node, ast.keyword) and node.arg in keywords:
            keyword_counts[node.arg] += 1
            node.arg = keywords[node.arg]
    if set(counts.values()) != {1} or set(keyword_counts.values()) != {2}:
        raise ValueError('Only exact original coordinator outcome labels may change')
    return ast.fix_missing_locations(tree)


def main():
    def verify(config):
        comparison, runtime = outcomes.verify_config(config)
        if config['paired_canonical_ids'] != comparison['paired_canonical_ids']:
            raise ValueError('Keep all ten original model outcomes')
        return runtime
    namespace = dict(planned.previous.__dict__, __doc__=__doc__, verify_config=verify,
        select_complete_pairs=select_outcomes, audit_complete=lambda config, runtime, snapshot: outcomes.audit(config))
    exec(compile(metadata_main_ast(), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
