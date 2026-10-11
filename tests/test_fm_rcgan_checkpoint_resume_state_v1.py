from types import SimpleNamespace
import numpy as np
import pytest
from scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1 import reachable_ops,assert_no_random_transition,compare_loss


class Op:
    def __init__(self,name,kind,inputs=(),controls=()):
        self.name=name;self.type=kind
        self.inputs=[SimpleNamespace(op=op) for op in inputs];self.control_inputs=list(controls)


def test_initializer_randomness_is_distinct_from_training_reachable_randomness():
    variable=Op('weights','VariableV2')
    initializer=Op('initializer','RandomNormal')
    update=Op('Adam','ApplyAdam',[variable])
    assert reachable_ops([update])=={update,variable}
    assert assert_no_random_transition([update])==2
    unsafe=Op('optimizer','NoOp',[variable],[initializer])
    with pytest.raises(ValueError,match='random operations'):
        assert_no_random_transition([unsafe])


def test_native_loss_shortest_trace_strings_must_round_trip_bitwise():
    value=np.float32(.78421634)
    compare_loss(float(value),float(str(value)),np)
    with pytest.raises(ValueError,match='bitwise'):
        compare_loss(float(value),float(value)+1e-4,np)
