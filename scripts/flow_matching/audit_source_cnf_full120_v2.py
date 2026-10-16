"""Full CNF replay bound to the accurate heldout-validation metadata v2."""
from types import FunctionType
from scripts.flow_matching import audit_source_cnf_full120_v1 as previous
from scripts.flow_matching import run_source_cnf_full120_v2 as corrected_execution

for _key,_value in vars(previous).items():
    if not _key.startswith('__'):globals()[_key]=_value
execution=corrected_execution
MODULE='scripts.flow_matching.audit_source_cnf_full120_v2'
CONFIG='configs/reproducibility/fm_source_cnf_full120_checkpoint_replay.v2.json'
for _key,_value in list(vars(previous).items()):
    if isinstance(_value,FunctionType) and _value.__module__==previous.__name__:
        globals()[_key]=FunctionType(_value.__code__,globals(),_key,_value.__defaults__,_value.__closure__)


def registration():
    c=FunctionType(previous.registration.__code__,globals())()
    c.update(state_root='experiments/runs/fm_source_cnf_full120_checkpoint_replay_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/source_cnf_full120_checkpoint_replay_v2')
    return c


if __name__=='__main__':main()
