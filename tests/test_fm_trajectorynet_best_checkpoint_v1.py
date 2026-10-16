"""Exercise actual original checkpoint selection and the isolated correction."""
import ast
import dis
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/runs/fm_foundation_baseline_sources/sources/trajectorynet/original/TrajectoryNet/main.py'
spec = importlib.util.spec_from_file_location('_trajectorynet_best_checkpoint_v1', ROOT / 'src/iia_benchmark/models/trajectorynet_best_checkpoint_v1.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def native_control_flow(tmp_path):
    captured = []
    values = iter([.4, .9, .2])

    class Loss(float):
        def item(self):
            return float(self)

    namespace = dict(__name__='_pinned_original_checkpoint_selection_fixture',
        torch=SimpleNamespace(save=lambda state, path: captured.append(state['state_dict'])),
        compute_loss=lambda *args: Loss(next(values)), count_nfe=lambda model: 0,
        utils=SimpleNamespace(makedirs=lambda p: Path(p).mkdir(exist_ok=True)),
        os=__import__('os'))
    original = ast.parse(PATH.read_text())
    functions = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ['train', 'train_eval']]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(PATH), 'exec'), namespace)
    module = SimpleNamespace(**namespace)
    return module, captured


def exercise(validation, captured, tmp_path, consume_return):
    step = [0]
    model = SimpleNamespace(eval=lambda: None, state_dict=lambda: step[0])
    args = SimpleNamespace(save=str(tmp_path), use_growth=False)
    logger = SimpleNamespace(info=lambda text: None)
    best = float('inf')
    for value in [100, 200, 300]:
        step[0] = value
        returned = validation(None, args, model, None, value, best, logger, None)
        if consume_return:
            best = returned
    return best


def test_actual_original_bug_and_corrected_best_checkpoint_state(tmp_path):
    native, original_saves = native_control_flow(tmp_path / 'original')
    exercise(native.train_eval, original_saves, tmp_path / 'original', consume_return=False)
    assert original_saves == [100, 200, 300]
    native, fixed_saves = native_control_flow(tmp_path / 'fixed')
    before = PATH.read_bytes()
    _, validation = s.corrected_functions(native)
    best = exercise(validation, fixed_saves, tmp_path / 'fixed', consume_return=True)
    assert fixed_saves == [100, 300] and best == .2
    assert PATH.read_bytes() == before


def test_every_original_numerical_statement_and_full_iteration_loop_is_retained(tmp_path):
    native, _ = native_control_flow(tmp_path)
    train, validation = s.corrected_functions(native)
    original = ast.parse(PATH.read_text())
    fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(1, args.niters + 1)']
    assert len(calls) == 1
    # Numeric APIs remain the same global objects; no replacement loss,
    # optimizer, model, adjoint, batch, or random generator is installed.
    assert train.__globals__['compute_loss'] is native.compute_loss
    assert train.__globals__['torch'] is native.torch
    assert train.__globals__['train_eval'] is validation
    assert train.__code__.co_filename.endswith('::best_checkpoint_state_correction_v1')
    assert 'backward' in train.__code__.co_names and 'step' in train.__code__.co_names
    original_ops = [(i.opname, i.argval) for i in dis.get_instructions(native.train)]
    corrected_ops = [(i.opname, i.argval) for i in dis.get_instructions(train)]
    assert len(original_ops) == len(corrected_ops)
    differences = [(old, new) for old, new in zip(original_ops, corrected_ops) if old != new]
    assert differences == [(('POP_TOP', None), ('STORE_FAST', 'best_loss'))]
