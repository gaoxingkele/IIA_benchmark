import ast
import copy
from pathlib import Path

import pytest

from scripts.flow_matching.run_maelnet_selected_exact_recovery_v2 import compare_recovery, selected_main_ast, original


def test_full_guarded_stage_controller_changes_only_existing_single_job_selector():
    source=ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    expected=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    actual=selected_main_ast()
    assignments=[n for n in ast.walk(actual) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='sys' and t.attr=='argv'
        for t in n.targets)]
    assert len(assignments)==1
    bound=assignments[0]
    assert ast.unparse(bound.value.right)=="['--job-id', registration['selected_job_id']]"
    bound.value=bound.value.left
    assert ast.dump(expected)==ast.dump(actual)


def test_full_original150_slots_four_stages_seed_and_outputs_preserved():
    old=dict(settings=dict(output_root='old_state',python='native',epochs=3),
        jobs=[dict(id=str(i),seed=i,output_directory='old/'+str(i),stages=['fast1','fast2','slow','test'],
            data='original',width=512) for i in range(150)])
    new=copy.deepcopy(old)
    new['settings']['output_root']='new_state'
    new['jobs'][106].update(id='106_retry',output_directory='new/106',recovery_original_job_id='106')
    compare_recovery(old,new,'106','106_retry')
    for key,value in [('seed',107),('stages',['fast1']),('data','other'),('width',64),('output_directory','old/106')]:
        bad=copy.deepcopy(new)
        bad['jobs'][106][key]=value
        with pytest.raises(ValueError): compare_recovery(old,bad,'106','106_retry')
    bad=copy.deepcopy(new)
    bad['jobs'][107]['seed']=999
    with pytest.raises(ValueError): compare_recovery(old,bad,'106','106_retry')
    bad=copy.deepcopy(new)
    bad['jobs'].pop()
    with pytest.raises(ValueError): compare_recovery(old,bad,'106','106_retry')
