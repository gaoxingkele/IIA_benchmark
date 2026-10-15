"""Register and execute the source-loop/paper-epoch compatibility audit.

Counter projections are plumbing evidence only; no model or corpus is trained.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.flow_matching import moment_icm_epoch_contract_v1 as implementation
from scripts.flow_matching.moment_icm_pretraining_inputs_v1 import ROOT, read, sha, write

CONFIG = 'configs/reproducibility/fm_moment_ICM_one_full_epoch_contract.2026-10-15_v1.json'
PREVIOUS = 'configs/reproducibility/fm_moment_ICM_pretraining_corpus_audit.2026-10-15_v2.json'


def registration():
    previous = read(ROOT / PREVIOUS)
    scope = 'configs/reproducibility/fm_moment_ICM_primary_scope_data.2026-10-15_v1.json'
    paper = read(ROOT / scope)
    source = previous['source_root'] + '/moment/tasks/pretrain.py'
    pins = [source, PREVIOUS, scope, paper['paper'],
            previous['source_root'] + '/configs/default.yaml',
            previous['source_root'] + '/configs/pretraining/pretrain.yaml',
            previous['source_root'] + '/moment/tasks/base.py',
            'scripts/flow_matching/moment_icm_epoch_contract_v1.py',
            'scripts/flow_matching/audit_moment_icm_epoch_contract_v1.py',
            'tests/test_fm_moment_icm_epoch_contract_v1.py']
    return dict(schema_version=1, paper_id='moment_long_context', paper_citation=paper['paper_citation'],
        paper=paper['paper'], related_source_commit=previous['related_source_commit'],
        source_path=source, source_sha256=sha(ROOT / source), paper_epochs=1,
        source_stopping_policy='source_two_minima_OR', paper_stopping_policy='paper_complete_epochs',
        keep_complete_final_partial_batch=True, adapter_status='prepared_not_neurally_executed',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in pins],
        previous_complete_input_audit=PREVIOUS,
        audit_output='F:/aicoding/IIA_Data/experiments/moment_icm_one_full_epoch_contract_v1/audit.json',
        boundary=('One isolated train-method AST with only the while test changed. Numerical body,optimizer,'
            'AMP,masking,gradient clipping,scheduler calls and per-batch ordering retained. Original source '
            'file/task and the source50epoch/5millionstep policy unchanged. Bound task scheduler max_epoch '
            'must equal1 and drop_lastFalse. Complete data scope,grouped multichannel loader,model/environment,'
            'source sampling,exact optimizer steps/tokens,resource admission,final checkpoint writing and '
            'native independent replay still require the actual scientific entry point. Counter projections '
            'and analytical examples are not neural training or measured corpus budgets. The sourceOR '
            'policy may be intentional for its own recipe;it does not enforce this paper oneepoch target.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen epoch-contract dependency changed: ' + pin['path'])
    if (config['paper_epochs'] != 1 or config['source_stopping_policy'] != 'source_two_minima_OR'
            or config['paper_stopping_policy'] != 'paper_complete_epochs'
            or not config['keep_complete_final_partial_batch']):
        raise ValueError('Explicit complete oneepoch and preserved source tracks required')
    implementation.paper_train_node(ROOT / config['source_path'], config['source_sha256'])


def source_integer(path, key):
    # These pinned flat YAML integer fields need no model/config import.
    matches = [line.split(':', 1)[1].split('#', 1)[0].strip()
               for line in Path(path).read_text(encoding='utf-8').splitlines()
               if line.startswith(key + ':')]
    if len(matches) != 1:
        raise ValueError('One explicit source integer expected: ' + key)
    return int(matches[0])


def audit(config):
    verify(config)
    source = ROOT / config['source_path']
    original = implementation.train_node(source, config['source_sha256'])
    private = implementation.paper_train_node(source, config['source_sha256'])
    examples = []
    for batches, steps, epochs in [(7, 20, 1), (7, 21, 1), (7, 22, 1), (3, 2, 4)]:
        task = SimpleNamespace(args=SimpleNamespace(max_opt_steps=steps, max_epoch=epochs))
        policies = {}
        for policy in ['source_two_minima_OR', 'paper_complete_epochs']:
            actual = implementation.counter_projection(source, config['source_sha256'], policy)(task, batches)
            expected = implementation.budget_outcome(batches, steps, epochs, policy)
            assert actual == (expected['epochs'], expected['optimizer_steps'])
            policies[policy] = expected
        examples.append(dict(counter_only_fixture=True, batches_per_epoch=batches,
                             step_minimum=steps, requested_epochs=epochs, outcomes=policies))
    source_root = source.parents[2]
    defaults = {}
    for relative in ['configs/default.yaml', 'configs/pretraining/pretrain.yaml']:
        defaults[relative] = {key: source_integer(source_root / relative, key)
                              for key in ['max_opt_steps', 'max_epoch']}
    assert all(v == dict(max_opt_steps=5000000, max_epoch=50) for v in defaults.values())
    paper_case = SimpleNamespace(args=SimpleNamespace(max_opt_steps=5000000, max_epoch=1))
    assert implementation.counter_projection(source, config['source_sha256'], 'paper_complete_epochs')(paper_case, 2048) == (1, 2048)
    counterexample = dict(hypothetical_batches_per_epoch=2048, measured_real_corpus_budget=False,
        source_after_only_setting_max_epoch_1=implementation.budget_outcome(2048, 5000000, 1, 'source_two_minima_OR'),
        isolated_full_epoch_policy=implementation.budget_outcome(2048, 5000000, 1, 'paper_complete_epochs'))
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG),
        source_sha256=config['source_sha256'], original_stop_condition=implementation.SOURCE_CONDITION,
        isolated_stop_condition=implementation.PAPER_CONDITION, source_default_budgets=defaults,
        original_ast_sha256=__import__('hashlib').sha256(ast.dump(original).encode()).hexdigest(),
        isolated_ast_sha256=__import__('hashlib').sha256(ast.dump(private).encode()).hexdigest(),
        only_stopping_condition_changed=True, source_file_unchanged=sha(source) == config['source_sha256'],
        original_loop_counter_projections=examples, illustrative_counterexample=counterexample,
        source_policy_may_be_intentional_for_its_own_recipe=True,
        adapted_callable_available=True, integrated_full_scientific_pretraining_runner_available=False,
        actual_model_updates=0, neural_pretraining_epochs_executed=0,
        actual_full_corpus_steps_or_tokens_certified=False, model_performance_results=0,
        numerical_source_or_running_task_hotpatched=False, all_original_experiments_complete=False)
    output = Path(config['audit_output'])
    output.parent.mkdir(parents=True, exist_ok=False)
    write(output, report)
    print(json.dumps(dict(passed=True, counter_projections=8, source_epoch_override_alone_insufficient=True,
                         isolated_callable_prepared=True, neural_epochs_executed=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.register:
        path = ROOT / CONFIG
        if path.exists():
            raise FileExistsError('Preserve the frozen epoch contract')
        config = registration()
        verify(config)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, indent=2)
            stream.write('\n')
    else:
        config = read(ROOT / CONFIG)
    audit(config)


if __name__ == '__main__':
    main()
