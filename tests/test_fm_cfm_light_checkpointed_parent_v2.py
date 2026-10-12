import ast
import subprocess
import sys
from pathlib import Path
from types import CodeType

from scripts.flow_matching.run_cfm_checkpointed_light_parent_v2 import lightweight_namespace,allowed_idle


def test_all_source_metadata_validation_and_full_producer_code_is_preserved():
    def equal_code(a,b):
        for field in ('co_code','co_names','co_varnames','co_freevars','co_cellvars','co_argcount','co_kwonlyargcount'):
            assert getattr(a,field)==getattr(b,field)
        assert len(a.co_consts)==len(b.co_consts)
        for left,right in zip(a.co_consts,b.co_consts):
            if isinstance(left,CodeType):equal_code(left,right)
            else:assert left==right
    values=lightweight_namespace()
    for name,path in [('main','run_cfm_checkpointed_full_retry_v1.py'),
                      ('verify_registration','run_cfm_checkpointed_original_job_v1.py')]:
        source=ast.parse((Path('scripts/flow_matching')/path).read_text())
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name)
        namespace={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
        equal_code(values[name].__code__,namespace[name].__code__)


def test_cold_parent_loads_no_numerical_libraries():
    code="from scripts.flow_matching.run_cfm_checkpointed_light_parent_v2 import lightweight_namespace;lightweight_namespace();import sys;assert not any(n in sys.modules for n in ('numpy','scipy','torch'))"
    result=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_no_handoff_of_actual_training_or_existing_partial_scientific_output():
    state=dict(active_pid=None,retained_partial_epochs_credited=0,state='waiting_for_heavy_lock')
    assert allowed_idle('original',state,False,[])
    assert not allowed_idle('original',dict(state,active_pid=1),False,[])
    assert not allowed_idle('original',state,True,[])
    assert not allowed_idle('original',state,False,[dict(name='python.exe',exe='F:/native/python.exe')])
