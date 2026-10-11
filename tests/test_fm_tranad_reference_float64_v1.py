import ast
import copy
import json

import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_tranad_reference_float64_v1 import (
    ReferenceFloat64TranADAdapter, ReferenceFloat64TranADDetector, precision_method_ast)
from iia_benchmark.models import mtsad_detectors as original
from scripts.flow_matching.run_tranad_reference_float64_queue_v1 import ROOT, bind, validate_queue


@pytest.mark.parametrize('name', ['fit', 'score'])
def test_entire_original_algorithm_ast_changes_only_float_precision(name):
    source=(ROOT/'src/iia_benchmark/models/mtsad_detectors.py').read_text(encoding='utf-8')
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='TranADDetector')
    original_node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    changed=precision_method_ast(source,name)
    casts=[n for n in ast.walk(changed) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
           and n.value.id=='np' and n.attr=='float64']
    assert len(casts)==1;casts[0].attr='float32'
    doubles=[n for n in ast.walk(changed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=='to' and isinstance(n.func.value,ast.Call)
             and isinstance(n.func.value.func,ast.Attribute) and n.func.value.func.attr=='double']
    assert len(doubles)==(1 if name=='fit' else 0)
    for call in doubles:call.func.value=call.func.value.func.value
    assert ast.dump(changed)==ast.dump(original_node)


def test_float64_fit_uses_all_original_modules_and_resolves_real_scale_overflow():
    torch.set_num_threads(1)
    windows=np.random.default_rng(17).normal(size=(6,10,3)).astype(np.float32)
    adapter=ReferenceFloat64TranADAdapter(method='tranad',parameters=dict(window=10,d_ff=16,batch_size=4,dropout=0),
                                        epochs=1,seed=1103,device='cpu').fit(windows)
    assert all(p.dtype==torch.float64 and torch.isfinite(p).all() for p in adapter._model.parameters())
    moderate=adapter.score(windows);assert moderate.shape==(6,) and np.isfinite(moderate).all()
    large=windows[:2].copy();large[:,:,0]=4.28699778285568e14
    assert np.isfinite(adapter.score(large)).all()
    previous=original.TranADDetector(window=10,d_ff=16,batch_size=4,dropout=0,device='cpu')
    previous._device=torch.device('cpu');previous._model=previous._build(3)
    previous._model.load_state_dict({k:v.float() for k,v in adapter._model.state_dict().items()})
    assert not np.isfinite(previous.score(large)).all()
    assert np.allclose(previous.score(windows),moderate,rtol=1e-4,atol=1e-6)


def test_old_complete_score_checks_and_foreign_method_rejection_remain():
    from iia_benchmark.models.fm_legacy_window_adapter import LegacyWindowAdapter
    assert ReferenceFloat64TranADAdapter.score is LegacyWindowAdapter.score
    with pytest.raises(ValueError):
        ReferenceFloat64TranADAdapter(method='other',parameters={},epochs=1).fit(np.ones((2,10,3)))


def registration():
    return json.loads((ROOT/'configs/runtime/fm_tranad_reference_float64.v1.json').read_text(encoding='utf-8'))


def test_all_five_original_datasets_and_five_original_seeds_budgets_and_sources_retained():
    value=registration();queue=validate_queue(ROOT,value)
    assert len(queue['jobs'])==25
    assert {j['dataset'] for j in queue['jobs']}=={'psm','smap','msl','smd','swat'}
    assert {j['seed'] for j in queue['jobs']}=={1103,1104,1105,1106,1107}
    controller,namespace,current,factory=bind(value)
    assert controller.main.__globals__ is namespace and current==queue and callable(factory)


@pytest.mark.parametrize('change',['mutex','settings','source'])
def test_no_new_training_concurrency_weakened_guards_or_changed_orchestration(change):
    value=registration()
    if change=='mutex':value['resource_lock']+='changed'
    elif change=='settings':value['settings']['minimum_commit_headroom_bytes']-=1
    else:value['source_receipts'][0]['sha256']='changed'
    with pytest.raises(ValueError):bind(value)
