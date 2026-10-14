"""Keep a superseded frozen audit fixture as evidence, not duplicate live tests."""
import pytest

from scripts.flow_matching.frozen_validation_collection import LEGACY, verify_registered_successor


def pytest_ignore_collect(collection_path, config):
    if collection_path.resolve() != LEGACY.resolve():
        return None
    try:
        verify_registered_successor()
    except (ValueError, OSError, StopIteration) as error:
        raise pytest.UsageError(str(error)) from error
    return True
