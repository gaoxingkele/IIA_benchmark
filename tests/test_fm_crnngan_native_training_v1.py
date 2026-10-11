"""Test actual author-loop changes, optimizer gating and full frozen obligations."""
import ast
import copy
from types import SimpleNamespace
import time

import pytest

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.crnngan_reviewed_source_v1 import functions, instrumented_main
from scripts.flow_matching.run_crnngan_native_original_queue_v1 import verify_registration


def source_path():
    asset = next(a for a in read(ROOT / 'configs/acquisition/fm_adopted_gan_original_sources.v1.json')['assets']
                 if a['id'] == 'crnngan')
    return ROOT / asset['original_directory'] / 'rnn_gan.py'


def test_whole_author_functions_preserved_except_three_explicit_corrections():
    original = {n.name: n for n in ast.parse(source_path().read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef)}
    main, epoch = functions(source_path())
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'run_epoch' and len(n.args) >= 6 and isinstance(n.args[4], ast.Attribute)
                and n.args[4].attr == 'opt_g')
    call.args[4], call.args[5] = call.args[5], call.args[4]
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'i')
    condition = next(n for n in loop.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                     and isinstance(n.test.left, ast.Name) and n.test.left.id == 'i')
    condition.test.left.id = 'global_step'
    assert ast.dump(main) == ast.dump(original['main'])
    branch = next(n for n in ast.walk(epoch) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                  and any(isinstance(k, ast.Name) and k.id == 'datasetlabel' for k in ast.walk(n.test)))
    old_branch = next(n for n in ast.walk(original['run_epoch']) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                      and any(isinstance(k, ast.Name) and k.id == 'datasetlabel' for k in ast.walk(n.test)))
    branch.body = copy.deepcopy(old_branch.body)
    assert ast.dump(epoch) == ast.dump(original['run_epoch'])


@pytest.mark.parametrize('losses,expected', [
    ((1.0, 0.2), ('G', None)), ((0.2, 1.0), (None, 'D')),
    ((1.0, 0.7), ('G', 'D')), ((0.7, 1.0), ('G', 'D')),
    ((0.0, 1.0), (None, 'D')), ((1.0, 0.0), ('G', None)),
])
def test_actual_corrected_run_epoch_freezes_only_stronger_network(losses, expected):
    _, epoch = functions(source_path())
    unit = ast.parse('')
    unit.body = [epoch]
    class Options:
        FULL_TRACE = 1
        def __init__(self, **kwargs):
            pass
    namespace = dict(time=time, tf=SimpleNamespace(no_op=lambda: None, RunOptions=Options))
    exec(compile(ast.fix_missing_locations(unit), str(source_path()), 'exec'), namespace)
    class Loader:
        def rewind(self, part):
            self.position = 0
        def get_batch(self, *args, **kwargs):
            self.position += 1
            return [object(), object()] if self.position <= 2 else [None, None]
    class Session:
        def __init__(self):
            self.fetched = []
        def run(self, fetches, *args, **kwargs):
            self.fetched.append(tuple(fetches[2:]))
            return [*losses, None, None]
    model = SimpleNamespace(batch_size=20, songlength=100, g_loss='g_loss', d_loss='d_loss',
                            input_songdata=SimpleNamespace(name='song'), input_metadata=SimpleNamespace(name='meta'))
    session = Session()
    namespace['run_epoch'](session, model, Loader(), 'train', 'G', 'D')
    assert session.fetched == [('G', 'D'), expected]
    session = Session()
    namespace['run_epoch'](session, model, Loader(), 'validation', 'G', 'D')
    assert session.fetched == [('G', 'D'), ('G', 'D')]


def test_curriculum_ends_after_six_full_pretraining_epochs():
    main, _ = functions(source_path())
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'i')
    branch = next(n for n in loop.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and isinstance(n.test.left, ast.Name) and n.test.left.id == 'i')
    unit = ast.parse('')
    unit.body = [branch]
    code = compile(ast.fix_missing_locations(unit), 'actual_author_curriculum', 'exec')
    lengths = []
    for i in range(8):
        ns = dict(i=i, global_step=0, FLAGS=SimpleNamespace(pretraining_epochs=6), songlength_ceiling=100)
        exec(code, ns)
        lengths.append(ns['new_songlength'])
    assert lengths == [4, 8, 12, 16, 20, 24, 100, 100]


def test_complete_training_validation_sampling_and_final_test_remain_in_instrumented_source():
    tree = instrumented_main(source_path())
    names = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert names.count('_native_graph_seed') == 1
    assert names.count('_native_epoch_complete') == 1
    assert names.count('_native_test_complete') == 1
    assert names.count('get_all_stats') == 1
    assert names.count('sample') == 2
    assert 'run_epoch' in names
    compile(tree, str(source_path()), 'exec')


def test_all_registered_full_variants_and_seeds_are_bound_to_model_configs():
    q = read(ROOT / 'configs/experiments/fm_crnngan_native_released_queue.v1.json')
    verify_registration(q)
    assert len(q['jobs']) == 35
    assert not q['paper_historical_corpus_identity_certified']
    assert not q['author_equivalence_certified']
    assert q['full_training_completed'] is False
    assert q['flags']['exit_after'] == 0
    assert q['flags']['select_validation_percentage'] is None
    assert q['flags']['select_test_percentage'] is None
    changed = copy.deepcopy(q)
    changed['jobs'] = changed['jobs'][:-1]
    with pytest.raises(ValueError, match='seven'):
        verify_registration(changed)
    changed = copy.deepcopy(q)
    changed['flags']['max_epoch'] = 5
    with pytest.raises(ValueError, match='budget'):
        verify_registration(changed)


def test_corrected_classical_registration_matches_every_independent_parsed_raw_record():
    from scripts.flow_matching.run_crnngan_native_original_queue_v2 import verify_registration as verify_classical
    q = read(ROOT / 'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json')
    verify_classical(q)
    prior = read(ROOT / q['prior_attempt_queue'])
    assert [(j['id'], j['seed']) for j in q['jobs']] == [(j['id'], j['seed']) for j in prior['jobs']]
    assert {j['output_directory'] for j in q['jobs']}.isdisjoint(j['output_directory'] for j in prior['jobs'])
    assert q['flags'] == prior['flags']
    records = read(ROOT / 'projects/flow_matching_research/results/2026-10-11/adopted_gan_native_preparation_v2/crnngan_all_released_midi.json')['evidence']['records']
    classical = [r for r in records if r['path'].startswith('data/classical/')]
    accepted = [r for r in classical if r['parsed']]
    assert len(classical) == 3029
    assert len(accepted) == q['parsed_released_files'] == 3014
    assert sum(r['note_events'] for r in accepted) == q['full_note_events'] == 10323257
    assert not q['recovery_seeds_are_additional_independent_repeats']


def test_native_counts_are_derived_from_exact_author_genre_composer_membership():
    path = source_path().with_name('music_data_utils.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    original_read_data = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MusicDataLoader')
    original_read_data = next(n for n in original_read_data.body if isinstance(n, ast.FunctionDef) and n.name == 'read_data')
    assignment = next(n for n in original_read_data.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == 'genres')
    assert ast.dump(assignment.value) == ast.dump(ast.parse('sorted(sources.keys())', mode='eval').body)
    unit = ast.parse('')
    unit.body = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(k, ast.Name) and k.id == 'sources' for k in ast.walk(n.targets[0]))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), namespace)
    assert set(namespace['sources']) == {'classical'}
    assert len(namespace['sources']['classical']) == 154
    records = read(ROOT / 'projects/flow_matching_research/results/2026-10-11/adopted_gan_native_preparation_v2/crnngan_all_released_midi.json')['evidence']['records']
    selected = [r for r in records if r['path'].split('/')[1] in namespace['sources']
                and r['path'].split('/')[2] in namespace['sources'][r['path'].split('/')[1]]]
    assert len(selected) == 3029


def test_checkpoint_reader_rejects_missing_nonfinite_or_corrupted_tensor():
    import hashlib
    import numpy as np
    from scripts.flow_matching.audit_crnngan_checkpoint_variables_v1 import verify_tensors
    class Reader:
        def __init__(self, values):
            self.values = values
        def get_variable_to_shape_map(self):
            return {k: list(v.shape) for k, v in self.values.items()}
        def get_tensor(self, name):
            return self.values[name]
    values = {'weights': np.array([1., 2.], dtype=np.float32), 'learning_rate': np.array(.1, dtype=np.float32)}
    audit = dict(full_variable_count=2, tensors=[dict(name=k, shape=list(v.shape), dtype=str(v.dtype),
        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in values.items()])
    assert verify_tensors(Reader(values), audit) == 2
    with pytest.raises(ValueError, match='Every'):
        verify_tensors(Reader({'weights': values['weights']}), audit)
    for wrong in [np.array([1., 3.], dtype=np.float32), np.array([1., float('nan')], dtype=np.float32)]:
        with pytest.raises(ValueError, match='differs'):
            verify_tensors(Reader(dict(values, weights=wrong)), audit)
