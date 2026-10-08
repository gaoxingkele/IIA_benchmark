import json
import importlib
import inspect
from pathlib import Path
import pytest

from scripts.flow_matching.model_registry import build_configured_method, verify_config


def test_config_controls_callable_and_parameters_without_fitting():
    method=build_configured_method({'entrypoint':'iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector',
                                    'parameters':{'objective':'target','window':50}}, {'window':12,'device':'cpu'})
    assert method.window==12 and method.objective=='target' and method._model is None


def test_source_only_status_cannot_be_silently_instantiated():
    with pytest.raises(ValueError,match='source_only'):
        build_configured_method({'reproduction_status':'source_only'})
    with pytest.raises(ValueError,match='local'):
        build_configured_method({'entrypoint':'os:system'})


def test_registered_objective_configs_have_code_and_citation():
    root=Path(__file__).resolve().parents[1]
    configs=list((root/'configs/models').glob('fm_objective_*.json'))
    assert len(configs)==8
    for path in configs:
        assert verify_config(path)['status']=='local_symbol_verified'
        method=build_configured_method(path,{'device':'cpu'})
        assert callable(method.fit) and callable(method.score) and callable(method.sample)


def test_all_new_profiles_have_real_symbols_and_valid_constructor_parameters():
    root=Path(__file__).resolve().parents[1]
    paths=[]
    for pattern in ('fm_native*.json','fm_comparator*.json','fm_supplemental*.json','fm_objective*.json'):
        paths.extend((root/'configs/models').glob(pattern))
    assert len(paths)>=66
    for path in paths:
        report=verify_config(path)
        config=json.loads(path.read_text(encoding='utf-8'))
        assert config.get('reproduction_status'),path.name
        if config.get('entrypoint'):
            module,symbol=config['entrypoint'].split(':')
            implementation=getattr(importlib.import_module(module),symbol)
            try:inspect.signature(implementation).bind(**config.get('parameters',{}))
            except TypeError as error:pytest.fail(f'{path.name}: {error}')
        else:assert report['status']=='explicitly_non_callable'
