import pytest

from scripts.flow_matching.freeze_spectral_table3_environment_v2 import verify_additions


def test_parser_overlay_cannot_change_the_inherited_numerical_environment():
    inherited = {'torch': '2.8.0+cu126', 'numpy': '1.26.4', 'pip': '23.0.1'}
    added = {'omegaconf': '2.2.0', 'antlr4-python3-runtime': '4.8'}
    actual = dict(inherited, **added)
    verify_additions(inherited, actual, added)
    for name in inherited:
        changed = dict(actual, **{name: 'another-version'})
        with pytest.raises(ValueError, match='Inherited package differs'):
            verify_additions(inherited, changed, added)


@pytest.mark.parametrize('unexpected', ['numpy', 'torch', 'PyYAML'])
def test_extra_overlay_packages_are_not_silently_allowed(unexpected):
    inherited = {'numpy': '1.26.4'}
    added = {'omegaconf': '2.2.0', 'antlr4-python3-runtime': '4.8', unexpected: 'changed'}
    with pytest.raises(ValueError, match='Only the missing'):
        verify_additions(inherited, dict(inherited, **added), added)
