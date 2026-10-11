import copy

import pytest

from scripts.flow_matching.register_spectral_table3_parser_recovery_v3 import require_science_unchanged


@pytest.mark.parametrize('field', ['seed', 'author_hyperparameters', 'raw_sha256', 'training_entry', 'paper_claims'])
def test_dependency_recovery_cannot_change_scientific_protocol(field):
    original = dict(id='original', seed=42, author_hyperparameters={'epochs': 1000, 'batch_size': 32},
        raw_sha256='full-native-data', training_entry='original.py', paper_claims={'marginal': .019},
        reproduction_status='pending')
    recovered = dict(original, id='attempt', reproduction_status='parser_fixed', environment_dependency_recovery='overlay')
    require_science_unchanged(original, recovered)
    altered = copy.deepcopy(recovered)
    altered[field] = 'different'
    with pytest.raises(ValueError, match='Only execution identity'):
        require_science_unchanged(original, altered)
