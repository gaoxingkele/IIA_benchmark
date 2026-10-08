import json
from pathlib import Path

import pytest
from scripts.flow_matching import transition_light_controllers_v2 as handoff


class Process:
    def __init__(self, command=(), children=(), name='python.exe', exe='python.exe'):
        self.command, self.descendants, self.label, self.image = command, children, name, exe
    def cmdline(self): return self.command
    def children(self, recursive=True): return self.descendants
    def name(self): return self.label
    def exe(self): return self.image
    def create_time(self): return 123.0


def setup(tmp_path, monkeypatch, child=(), active=None, queue='queue.json'):
    binding={'status_path':'status.json','old_worker_token':'old_worker.py',
             'queue_argument':'--queue','queue_path':'queue.json'}
    (tmp_path/'status.json').write_text(json.dumps({'pid':42,'state':'waiting_training_results','active_pid':active}))
    process=Process(['python','old_worker.py','--queue',queue],child)
    monkeypatch.setattr(handoff.original,'ROOT',tmp_path)
    monkeypatch.setattr(handoff.psutil,'Process',lambda pid:process)
    return binding,process


def test_verified_idle_controller_without_children_can_handoff(tmp_path, monkeypatch):
    binding,process=setup(tmp_path,monkeypatch)
    assert handoff.idle_observation(binding)[0] is process


def test_exact_console_helper_without_children_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setenv('SystemRoot',str(tmp_path/'Windows'))
    helper=Process(name='conhost.exe',exe=str(tmp_path/'Windows/System32/conhost.exe'))
    binding,process=setup(tmp_path,monkeypatch,[helper])
    assert handoff.idle_observation(binding)[0] is process


@pytest.mark.parametrize('kind',['model','fake_console','console_with_child','active_job','wrong_queue'])
def test_model_unknown_child_active_job_or_wrong_queue_blocks(tmp_path,monkeypatch,kind):
    monkeypatch.setenv('SystemRoot',str(tmp_path/'Windows'))
    child=Process(name='conhost.exe',exe=str(tmp_path/'Windows/System32/conhost.exe'))
    if kind=='model': child=Process(name='python.exe')
    elif kind=='fake_console': child.image=str(tmp_path/'fake/conhost.exe')
    elif kind=='console_with_child': child.descendants=[Process()]
    binding,_=setup(tmp_path,monkeypatch,[child],active=7 if kind=='active_job' else None,
                    queue='other.json' if kind=='wrong_queue' else 'queue.json')
    assert handoff.idle_observation(binding) is None
