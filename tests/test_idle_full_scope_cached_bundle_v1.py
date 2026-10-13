import copy
import pytest
from scripts.flow_matching import run_idle_full_scope_cached_bundle_v1 as current


@pytest.mark.parametrize('field,value',[('active_pid',8),('state','running'),('shared_lock_owned',True)])
def test_no_native_child_or_owned_slot_can_be_handed_off(field,value):
    state=dict(state='waiting_full_gpu_resources',active_pid=None,shared_lock_owned=False)
    assert current.idle_allowed('original',state,False,[])
    assert not current.idle_allowed('original',dict(state,**{field:value}),False,[])
    assert not current.idle_allowed('original',state,False,[dict(name='python.exe',exe='scientific_worker.exe')])


def config_fixture():
    return dict(original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273),
                source_receipts=[],roles={'table2':dict(entry_module='scripts.flow_matching.run_spectral_table2_gpu_memory_queue_v2',
                previous_owner=dict(command=['python','-m','scripts.flow_matching.run_spectral_table2_gpu_memory_queue_v2','--runtime-config','original.json']),
                original_arguments=['--runtime-config','original.json'],original_queue='queue.json',queue_items_key='jobs',required_original_slots=30)})


def test_frozen_command_arguments_and_full_original_queue_are_mandatory(monkeypatch):
    config=config_fixture()
    monkeypatch.setattr(current.cold.original,'read',lambda path:dict(jobs=list(range(30))))
    assert len(current.verify(config,'table2')[1]['jobs'])==30
    altered=copy.deepcopy(config);altered['roles']['table2']['original_arguments']=['--runtime-config','smoke.json']
    with pytest.raises(ValueError,match='command argument'):
        current.verify(altered,'table2')
    shortened=copy.deepcopy(config);shortened['roles']['table2']['required_original_slots']=1
    with pytest.raises(ValueError,match='complete queue'):
        current.verify(shortened,'table2')


def test_only_registered_host_modules_may_be_executed(monkeypatch):
    config=config_fixture();config['roles']['table2']['entry_module']='scripts.flow_matching.run_tsad_experiment'
    with pytest.raises(ValueError,match='host entry'):
        current.verify(config,'table2')
