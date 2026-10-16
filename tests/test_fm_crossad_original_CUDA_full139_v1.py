import ast
from pathlib import Path
import pytest

from scripts.flow_matching import run_crossad_original_CUDA_full139_v1 as s


def test_all139_original_conditions_keep_every_recipe_seed_and_frozen_file():
    c = s.registration()
    originals = {j['id']: j for j in s.read(s.ROOT / s.first.preflight.QUEUE)['jobs']}
    assert len(c['jobs']) == len(originals) == 139
    assert sum(j['reconstructed_ablation'] for j in c['jobs']) == 90
    assert sum(j['mode'] == 'release_checkpoint' for j in c['jobs']) == 7
    for job in c['jobs']:
        old = originals[job['canonical_CPU_condition_reference']]
        for key in ['model_parameters', 'train_parameters', 'seed', 'raw_sha256', 'train_length',
                    'release_checkpoint', 'frozen_files', 'reconstructed_ablation', 'mode']:
            assert job[key] == old[key], (job['id'], key)
        if old['reconstructed_ablation']:
            assert job['ablation_row'] == old['ablation_row']
        assert not job['activation_checkpointing'] and job['output_directory'] != old['output_directory']
    b = s.read(s.ROOT / s.first.CONFIG)
    assert c['host_guards'] == b['host_guards'] and c['GPU'] == b['GPU']
    assert c['delegated_first_original_condition'] in originals
    assert not c['original_CPU_conditions_replaced'] and c['new_independent_training_seed_slots'] == 0


def test_main_and_each_reconstructed_ablation_use_their_actual_distinct_builder(monkeypatch):
    from iia_benchmark.models import fm_crossad_author_runtime as runtime
    from iia_benchmark.models import fm_crossad_ablation as ablation
    calls = []
    class Model:
        def float(self): return self
    def main(runtime, parameters):
        calls.append(('main', runtime, parameters)); return Model()
    def reconstructed(runtime, parameters, row):
        calls.append(('ablation', runtime, parameters, row)); return Model()
    monkeypatch.setattr(runtime, 'build_model', main)
    monkeypatch.setattr(ablation, 'build_ablation', reconstructed)
    c = s.registration()
    jobs = [next(j for j in c['jobs'] if not j['reconstructed_ablation'])]
    jobs += [next(j for j in c['jobs'] if j.get('ablation_row') == row) for row in range(1, 6)]
    for job in jobs:
        s.build(c, job)
        s.build(c, job, cpu=True)
    assert [x[0] for x in calls] == ['main', 'main'] + ['ablation'] * 10
    assert [x[3] for x in calls[2::2]] == [1, 2, 3, 4, 5]
    assert all(x[1] == s.ROOT / c['cpu_reference_root'] for x in calls[1::2])


def test_no_later_condition_may_launch_before_both_actual_first_guards_pass(monkeypatch, tmp_path):
    c = s.registration(); b = s.read(s.ROOT / s.first.CONFIG)
    state = tmp_path / b['state_root']; state.mkdir(parents=True)
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    monkeypatch.setattr(s, 'read', lambda p: b if p == tmp_path / c['delegated_first_configuration'] else __import__('json').loads(Path(p).read_text()))
    def write(name, value):
        (state / name).write_text(__import__('json').dumps(value))
    write('status.json', dict(completed_conditions=0, state='running_original_CUDA_full_train'))
    assert s.first_completion(c) is None
    write('status.json', dict(completed_conditions=1, state='complete'))
    with pytest.raises(FileNotFoundError): s.first_completion(c)
    result = tmp_path / 'source.json'; result.write_text('{}')
    good = dict(passed=True, exit_code=0, resource_guard_aborted=False, report_path=str(result), report_sha256=s.sha(result))
    write('train.guard_receipt.json', good); write('replay.guard_receipt.json', dict(good, resource_guard_aborted=True))
    with pytest.raises(ValueError): s.first_completion(c)
    write('replay.guard_receipt.json', good)
    assert s.first_completion(c)['actual_original_CUDA_condition_complete']
    result.write_text('{"changed":true}')
    with pytest.raises(ValueError): s.first_completion(c)


def test_all_split_tails_labels_and_batches_are_required_not_a_subset():
    tree = ast.parse((s.ROOT / (s.MODULE.replace('.', '/') + '.py')).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    code = ast.unparse(functions['source_or_replay'])
    assert "('train', data.train)" in code and "('validation', data.val)" in code and "('test', data.test)" in code
    assert "score.tobytes() != reference.tobytes()" in code
    assert "detail['epochs_completed'] * len(loader)" in code
    assert 'saved[\'source_rows\']' in code and "saved['labels']" in code
    assert 'original.evaluate_saved' in code and 'first.compare_nested' in code
    calls = [ast.unparse(n.func) for n in ast.walk(functions['native_probe']) if isinstance(n, ast.Call)]
    assert 'loss.backward' in calls and not any(name.endswith('.step') for name in calls)
    assert tuple(s.PHASES) == ('CUDA_probe', 'CPU_reference', 'author', 'replay')
