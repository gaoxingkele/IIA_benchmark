import ast
from pathlib import Path
import pytest

from scripts.flow_matching import followup_giflow_seed_replay_verification_v1 as original
from scripts.flow_matching.followup_giflow_seed_replay_verification_v2 import bound_main_tree

def test_only_heavy_scheduler_import_binding_changes_full_followup_main():
    source=Path(original.__file__).read_text(encoding='utf-8')
    old=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    new=bound_main_tree(source).body[0]
    heavy={'scripts.flow_matching.heavy_resources','scripts.flow_matching.heavy_scheduler_v2'}
    oldbody=[ast.dump(n,include_attributes=False) for n in old.body if not(isinstance(n,ast.ImportFrom) and n.module in heavy)]
    newbody=[ast.dump(n,include_attributes=False) for n in new.body]
    inserted=[ast.dump(n,include_attributes=False) for n in ast.parse("api=resource_api()\nexclusive_lock=api['exclusive_lock']\nresource_slot=api['resource_slot']").body]
    # Remove only the three inserted bindings, preserving the later existing api
    # creation and every source check, terminal gate, budget, dispatch and guard.
    pos=newbody.index(inserted[0])
    assert newbody[pos:pos+3]==inserted
    assert newbody[:pos]+newbody[pos+3:]==oldbody
    assert not any(isinstance(n,ast.ImportFrom) and n.module in heavy for n in ast.walk(new))
    with pytest.raises(ValueError): bound_main_tree(source.replace('from scripts.flow_matching.heavy_resources import exclusive_lock','pass'))
