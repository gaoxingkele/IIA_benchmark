import ast
import csv
import inspect
import subprocess
import sys

import pytest
import torch

from scripts.flow_matching import recover_otcfm_completed1000_CPU_test_v1 as s


def test_only_actual_hundred_validation_rows_no_inferred_per_epoch_history(tmp_path):
    path = tmp_path / 'metrics.csv'
    rows = [dict(epoch=epoch-1, step=epoch*10-1, **{'val/loss': 1/epoch}) for epoch in range(10, 1001, 10)]
    def save(values):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'val/loss'])
            writer.writeheader()
            writer.writerows(values)
    save(rows)
    result = s.validation_rows(path)
    assert len(result) == 100 and result[-1]['optimizer_updates'] == 10000
    for invalid in [rows[:-1], rows + [rows[0]], [dict(r, step=0) for r in rows]]:
        save(invalid)
        with pytest.raises(ValueError): s.validation_rows(path)


def checkpoint(epoch):
    sizes = [384, 64, 4096, 64, 4096, 64, 320, 5]
    return dict(epoch=epoch-1, global_step=epoch*10,
        loops={'fit_loop': {'epoch_loop.batch_progress': {'total': dict(ready=epoch*10, started=epoch*10, processed=epoch*10, completed=epoch*10)}}},
        state_dict=dict(w=torch.ones(3)), optimizer_states=[dict(
            state={i: dict(step=torch.tensor(float(epoch*10)), exp_avg=torch.zeros(n), exp_avg_sq=torch.zeros(n)) for i,n in enumerate(sizes)},
            param_groups=[dict(lr=.001, weight_decay=1e-5)])])


def test_full_checkpoint_batch_optimizer_budget_rejects_partial_or_changed_state():
    actual = checkpoint(1000)
    assert sum(v['exp_avg'].numel() for v in actual['optimizer_states'][0]['state'].values()) == 9093
    s.validate_checkpoint(actual, 1000, torch)
    for field, value in [('epoch', 998), ('global_step', 9999)]:
        with pytest.raises(ValueError): s.validate_checkpoint(dict(actual, **{field:value}), 1000, torch)
    actual['optimizer_states'][0]['state'][0]['step'] = torch.tensor(9999.)
    with pytest.raises(ValueError): s.validate_checkpoint(actual, 1000, torch)


def test_replay_changes_only_two_metadata_reads_and_preserves_all_numerical_statements():
    old = ast.parse(inspect.getsource(s.replay.worker))
    new = s.replay_worker_ast()
    class RestoreMetadata(ast.NodeTransformer):
        def visit_Call(self,node):
            if ast.dump(node) == ast.dump(ast.parse('_observed_epoch(result,last_epoch)',mode='eval').body):
                return ast.parse("result['epoch_history'][last_epoch-1]",mode='eval').body
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            node = self.generic_visit(node)
            if ast.dump(node) == ast.dump(ast.parse("result['observed_checkpoint_training_history']",mode='eval').body):
                return ast.parse("result['epoch_history']",mode='eval').body
            return node
    assert ast.dump(RestoreMetadata().visit(new)) == ast.dump(old)
    rows = dict(observed_checkpoint_training_history=[dict(epoch=350, optimizer_updates=3500)])
    assert s.observed_epoch(rows,350)['optimizer_updates'] == 3500
    with pytest.raises(ValueError): s.observed_epoch(rows,351)


def test_recovery_entry_import_is_metadata_only_without_numerical_libraries():
    code = ('import sys;from scripts.flow_matching import recover_otcfm_completed1000_CPU_test_v1;'
            'assert not any(k in sys.modules for k in ["numpy","torch"])')
    subprocess.run([sys.executable,'-B','-c',code],cwd=s.ROOT,check=True)
