"""Full native trainer contract checks; no fixture is benchmark performance."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from iia_benchmark.models import dsb_singlecell_native_ipf_v1 as s

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/models/fm_dsb_singlecell_native_IPF_E4_CPU.v1.json'


def test_actual_original_inclusive_loop_is_used_without_replacement_optimizer_loop():
    profile = json.loads(PROFILE.read_text())
    assert s.require_full_budget(SimpleNamespace(**profile['native_arguments'])) == 400040
    source = (ROOT/profile['source_root']/'bridge/runners/ipf.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IPFSequential')
    phase = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ipf_step')
    assert any(isinstance(n,ast.For) and ast.unparse(n.iter)=='tqdm(range(self.num_iter + 1))' for n in ast.walk(phase))
    ours = ast.parse(Path(s.__file__).read_text())
    fn = next(n for n in ast.walk(ours) if isinstance(n,ast.FunctionDef) and n.name=='ipf_step')
    assert not any(isinstance(n,(ast.For,ast.While)) for n in fn.body)
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='super().ipf_step' for n in ast.walk(fn))
    assert 'class Accelerator' not in Path(s.__file__).read_text()


@pytest.mark.parametrize('field,value', [('num_iter',9999),('n_ipf',19),('batch_size',128),
    ('cache_npar',9999),('num_steps',19),('cache_refresh_stride',1000),('gamma_min',.01),('ema',True)])
def test_no_registered_scientific_budget_or_native_source_default_is_silently_reduced(field,value):
    profile = json.loads(PROFILE.read_text())
    args = dict(profile['native_arguments'],**{field:value})
    with pytest.raises(ValueError,match='Preserve registered'):
        s.require_full_budget(SimpleNamespace(**args))


def test_metadata_hooks_leave_source_train_and_sampling_equations_in_original_files():
    text = Path(s.__file__).read_text()
    assert 'def train(' not in text and 'def new_cacheloader(' not in text
    assert 'def record_langevin_seq(' not in text
    profile = json.loads(PROFILE.read_text())
    assert profile['parameters_per_network']==14261 and profile['total_parameters']==28522
    assert profile['historical_paper_equivalence_certified'] is False
    assert profile['total_known_models_including_deferred']==70
