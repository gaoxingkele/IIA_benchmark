from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.flow_matching.crossad_psm1103_full_retry_v1 import CANONICAL, STATE, build_queue


def test_only_failed_PSM_output_changes_and_all139_slots_remain():
    original = json.loads(Path('configs/experiments/fm_crossad_complete_queue.v7.json').read_text())
    result = build_queue(original)
    assert len(result['jobs']) == 139
    expected = deepcopy(original)
    job = next(j for j in expected['jobs'] if j['id'] == CANONICAL)
    job['output_directory'] = STATE + '/jobs/' + CANONICAL
    assert result == expected


@pytest.mark.parametrize('field,value', [('train_epochs', 19), ('batch_size', 64), ('patience', 2)])
def test_shortened_or_changed_original_recipe_is_rejected(field, value):
    original = json.loads(Path('configs/experiments/fm_crossad_complete_queue.v7.json').read_text())
    next(j for j in original['jobs'] if j['id'] == CANONICAL)['train_parameters'][field] = value
    with pytest.raises(ValueError):
        build_queue(original)


def test_canonical_slot_cannot_be_removed():
    original = json.loads(Path('configs/experiments/fm_crossad_complete_queue.v7.json').read_text())
    original['jobs'] = [j for j in original['jobs'] if j['id'] != CANONICAL]
    with pytest.raises(ValueError):
        build_queue(original)
