import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching.audit_rcgan_random_order_chain_v1 import compare_state, extract_functions, RandomProtocolSession


def saved_state(state):
    return dict(algorithm=np.array(state[0]), keys=state[1].copy(), position=np.array(state[2]),
                has_gauss=np.array(state[3]), cached_gaussian=np.array(state[4]))


def test_all_rng_fields_are_required_including_gaussian_cache():
    rng = np.random.RandomState(1103)
    rng.normal(size=3)
    actual = rng.get_state()
    saved = saved_state(actual)
    compare_state(actual, saved, np)
    for key in ('position', 'has_gauss', 'cached_gaussian'):
        wrong = dict(saved)
        wrong[key] = np.array(saved[key].item()+1)
        with pytest.raises(ValueError):
            compare_state(actual, wrong, np)
    wrong = dict(saved)
    wrong['keys'] = saved['keys'].copy()
    wrong['keys'][0] ^= 1
    with pytest.raises(ValueError):
        compare_state(actual, wrong, np)


def test_ast_extraction_excludes_native_imports_and_requires_complete_functions(tmp_path):
    path = tmp_path / 'source.py'
    path.write_text('import unavailable_library\ndef sample_Z(x):\n    return x\n', encoding='utf-8')
    namespace = {}
    extract_functions(path, {'sample_Z'}, namespace)
    assert namespace['sample_Z'](7) == 7
    with pytest.raises(ValueError):
        extract_functions(path, {'sample_Z', 'train_epoch'}, {})


def test_random_only_session_cannot_be_mistaken_for_native_training():
    session = RandomProtocolSession(np)
    session.run('D', {'Z':np.ones((28, 28, 5))})
    session.run('G', {'Z':np.ones((28, 28, 5))})
    session.run(['D_loss','G_loss'], {'Z':np.ones((28, 28, 5)), 'X':np.arange(28)})
    assert session.counts == dict(D=1,G=1,loss=1)
    assert np.array_equal(session.loss_batch_ids, np.arange(28))
    with pytest.raises(ValueError):
        session.run('sigma_solver', {})


def test_source_has_no_tensorflow_import_or_synthetic_benchmark_claim():
    source = Path('scripts/flow_matching/audit_rcgan_random_order_chain_v1.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Import) and any(a.name == 'tensorflow' for a in n.names) for n in ast.walk(tree))
    assert 'tensorflow_model_updates_executed_by_this_audit=0' in source
    assert 'numerical_training_losses_reconstructed=False' in source
