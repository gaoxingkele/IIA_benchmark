import ast
import copy
from pathlib import Path
import pytest

from scripts.flow_matching.audit_giflow_corrected_checkpoint_v1 import native_test_function_corrected, corrected_boundary
from scripts.flow_matching.audit_giflow_corrected_arrays_v1 import validate_selection


SOURCE = '''
def run_experiment():
    model.load_state_dict(torch.load(early_stopping.path, map_location=device, weights_only=True))
    model.eval()
    losses_mae = []
    losses_mse = []
    losses_mape = []
    with torch.no_grad():
        for batch in test_loader:
            num_steps = 20
            for _ in range(num_steps):
                model(batch)
        ema.restore()
        return losses_mae, losses_mse, losses_mape
'''


def result():
    return dict(job={'track': 'reviewed_corrected'}, selection=dict(test_used_in_validation=False,
                tested_weights_match_validation_checkpoint=True, actually_evaluated_weights_saved='evaluated_model.pt'),
                data_binding=dict(evaluation_targets_in_input=0,timestamp_intersections=0),
                epochs_completed=68, optimizer_updates=3264)


def test_native_corrected_block_is_whole_and_unmodified():
    generated = native_test_function_corrected(SOURCE).body[0]
    original = ast.parse(SOURCE).body[0]
    assert [ast.dump(n) for n in generated.body] == [ast.dump(n) for n in original.body[1:]]
    assert ast.unparse(generated.body[-1]).count('num_steps = 20') == 1


@pytest.mark.parametrize('changed',[SOURCE.replace('20','19'),SOURCE.replace('early_stopping.path','final_path'),
                                   SOURCE.replace('model.eval()','ema.apply_shadow()')])
def test_reduced_sampler_or_unselected_weights_rejected(changed):
    with pytest.raises(ValueError):
        native_test_function_corrected(changed)


def test_corrected_data_selection_accepts_actual_reviewed_branch():
    validate_selection(result())
    proof=corrected_boundary(dict(full_checkpoint_contents_and_reinference_passed=True),result())
    assert proof['tested_checkpoint_selected_on_validation']
    assert not proof['test_used_in_validation']
    assert proof['actual_full_training_epochs']==68


@pytest.mark.parametrize('field',['test_used_in_validation','tested_weights_match_validation_checkpoint'])
def test_wrong_selection_cannot_be_described_as_corrected(field):
    actual=result();actual['selection'][field]=not actual['selection'][field]
    with pytest.raises(ValueError):validate_selection(actual)
    with pytest.raises(ValueError):corrected_boundary(dict(full_checkpoint_contents_and_reinference_passed=True),actual)


@pytest.mark.parametrize('field',['evaluation_targets_in_input','timestamp_intersections'])
def test_leaking_data_binding_rejected(field):
    actual=result();actual['data_binding'][field]=1
    with pytest.raises(ValueError):validate_selection(actual)


def test_actual_full_corrected_source_is_supported():
    source=Path('experiments/runs/flow_matching_campaign/sources/giflow/corrected/main.py')
    native_test_function_corrected(source.read_text(encoding='utf-8'))
