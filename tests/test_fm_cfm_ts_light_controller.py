import json
from pathlib import Path

from scripts.flow_matching.run_cfm_ts_light_queue_v1 import load, ROOT


def runtime():
    from scripts.flow_matching.run_light_controller import digest
    queue='configs/experiments/fm_cfm_ts_original_queue.v1.json'
    return {'queue':queue,'queue_sha256':digest(ROOT / queue),'source_receipts':[]}


def test_controller_preserves_every_original_operation():
    from tests.test_fm_native_exact_recovery import instructions
    from scripts.flow_matching.run_cfm_ts_original_queue import main
    bound=load(runtime())
    assert instructions(bound.__code__) == instructions(main.__code__)


def test_model_and_budget_verification_functions_preserved():
    from tests.test_fm_native_exact_recovery import instructions
    from scripts.flow_matching.run_cfm_ts_original_job import verify,complete
    bound=load(runtime())
    assert instructions(bound.__globals__['verify'].__code__)==instructions(verify.__code__)
    assert instructions(bound.__globals__['complete'].__code__)==instructions(complete.__code__)


def test_changed_queue_is_rejected():
    import pytest
    config=runtime();config['queue_sha256']='0'*64
    with pytest.raises(ValueError,match='queue changed'):
        load(config)
