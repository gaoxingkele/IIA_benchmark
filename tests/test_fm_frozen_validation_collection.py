import hashlib

import pytest

from scripts.flow_matching.frozen_validation_collection import (
    LEGACY, SUCCESSOR, validate_successor, verify_registered_successor,
)


def test_registered_successor_preserves_all_scientific_checks_and_frozen_bytes():
    before = LEGACY.read_bytes()
    verify_registered_successor()
    assert LEGACY.read_bytes() == before


@pytest.mark.parametrize('weakening', [
    ("config['cases'] = config['cases'][:1]", "config['cases'] = []"),
    ('replay.verify(config)', 'None'),
    ('[42, 1103, 1104, 1105, 1106, 1107]', '[42, 1103, 1104, 1106, 1106, 1107]'),
])
def test_weakened_or_incorrect_successor_cannot_replace_frozen_checks(weakening):
    legacy = LEGACY.read_bytes()
    source = SUCCESSOR.read_text(encoding='utf-8')
    old, new = weakening
    assert old in source
    changed = source.replace(old, new).encode('utf-8')
    with pytest.raises(ValueError, match='every original scientific check'):
        validate_successor(legacy, changed, hashlib.sha256(legacy).hexdigest())


def test_changed_frozen_historical_source_is_rejected():
    legacy = LEGACY.read_bytes()
    with pytest.raises(ValueError, match='Frozen historical validation source changed'):
        validate_successor(legacy + b'\n', SUCCESSOR.read_bytes(), hashlib.sha256(legacy).hexdigest())


def test_collection_rule_only_replaces_the_specific_verified_historical_fixture():
    from tests.conftest import pytest_ignore_collect
    assert pytest_ignore_collect(SUCCESSOR, None) is None
    assert pytest_ignore_collect(LEGACY, None) is True
