"""Pagefile planning is a counterfactual and must not grant experiment admission."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess

import pytest

from scripts.flow_matching.audit_windows_commit_admission_v1 import GIB,build_plan,eligibility


def configuration():
    return dict(automatic_managed=True,paging_files=[r'?:\pagefile.sys'],
        active_pagefiles=[dict(Name=r'C:\pagefile.sys',AllocatedBaseSize=4096)],
        F_disk=dict(drive_type=3,file_system='NTFS',free_bytes=120*GIB),
        target_file_exists=False,is_administrator=False)


def test_wildcard_preserves_actual_C_system_managed_and_no_input_mutation():
    c=configuration();before=deepcopy(c);p=build_plan(c)
    assert c==before
    assert p['proposed_paging_files']==[r'C:\pagefile.sys 0 0',r'F:\pagefile.sys 32768 32768']
    assert p['existing_C_pagefile_preserved'] and p['existing_C_volume_remains_system_managed']
    assert not p['proposed_automatic_managed']
    assert p['configuration_only_not_a_runtime_capacity_increase']
    assert p['no_reboot_or_worker_stop'] and p['administrator_required']


def test_explicit_existing_allocation_and_other_volumes_are_preserved():
    c=configuration();c.update(automatic_managed=False,paging_files=[r'C:\pagefile.sys 8192 16384',r'D:\pagefile.sys 0 0'])
    p=build_plan(c)
    assert p['proposed_paging_files'][:-1]==c['paging_files']
    assert not p['existing_C_volume_remains_system_managed']


@pytest.mark.parametrize('change',[
    dict(target_file_exists=True),
    dict(active_pagefiles=[dict(Name=r'F:\pagefile.sys')]),
    dict(paging_files=[r'F:\pagefile.sys 0 0']),
    dict(paging_files=[]),dict(active_pagefiles=[]),dict(automatic_managed=False),
    dict(active_pagefiles=[dict(Name=r'D:\pagefile.sys')]),
])
def test_refuse_existing_F_or_loss_of_C_or_unbound_wildcard(change):
    c=configuration();c.update(change)
    with pytest.raises(ValueError):build_plan(c)


@pytest.mark.parametrize('field,value',[('free_bytes',96*GIB-1),('drive_type',2),('file_system','exFAT')])
def test_refuse_insufficient_reserve_or_wrong_volume(field,value):
    c=configuration();c['F_disk'][field]=value
    with pytest.raises(ValueError):build_plan(c)


def test_only_reviewed_size_and_no_duplicate_existing_entries():
    with pytest.raises(ValueError):build_plan(configuration(),64)
    c=configuration();c['paging_files']*=2
    assert len(build_plan(c)['proposed_paging_files'])==2


def test_counterfactual_never_admits_model_and_preserves_physical_gate():
    s=dict(minimum_available_memory_bytes=24*GIB,minimum_commit_headroom_bytes=40*GIB)
    m=dict(physical_available_bytes=23*GIB,commit_available_bytes=24*GIB)
    before=deepcopy((s,m))
    assert not eligibility(s,m)['memory_gates_pass']
    p=eligibility(s,m,32*GIB)
    assert p['commit_gate_passes'] and not p['physical_gate_passes']
    assert not p['memory_gates_pass'] and not p['actual_experiment_admitted'] and p['projection_only']
    m['physical_available_bytes']=24*GIB
    p=eligibility(s,m,32*GIB)
    assert p['memory_gates_pass'] and not p['actual_experiment_admitted']
    assert s==before[0] and m['commit_available_bytes']==before[1]['commit_available_bytes']


@pytest.mark.skipif(os.name!='nt',reason='PowerShell parser requires Windows')
def test_configuration_script_parses_and_has_no_automatic_elevation_restart_or_worker_stop():
    p=Path(__file__).resolve().parents[1]/'scripts/flow_matching/configure_F_pagefile_reviewed_v1.ps1'
    command="param($path) $tokens=$null;$errors=$null;[void][System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors);if($errors.Count){throw ($errors|Out-String)}"
    # Use stdin to preserve literal paths; parsing does not execute the script.
    quoted=str(p).replace("'","''")
    command=command.replace('param($path)',"$path='"+quoted+"';")
    result=subprocess.run(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    source=p.read_text(encoding='utf-8').lower()
    for forbidden in ['restart-computer','stop-process','taskkill','shutdown.exe','start-process','-verb runas']:
        assert forbidden not in source
    assert "if (-not $apply)" in source and 'if (-not $admin)' in source


@pytest.mark.skipif(os.name!='nt',reason='PowerShell behavioral harness requires Windows')
@pytest.mark.parametrize('inject_failure',[False,True])
def test_privileged_configuration_and_rollback_with_simulated_CIM_and_registry(tmp_path,inject_failure):
    """Exercise both Apply outcomes without touching actual privileged settings."""
    source_path=Path(__file__).resolve().parents[1]/'scripts/flow_matching/configure_F_pagefile_reviewed_v1.ps1'
    source=source_path.read_text(encoding='utf-8')
    admin_line=next(x for x in source.splitlines() if x.startswith('$admin='))
    backup_line="$root='F:\\aicoding\\IIA_Data\\environments\\windows_commit_capacity'"
    assert source.count(admin_line)==source.count(backup_line)==1
    # Replace only the privilege observation and backup directory in this isolated
    # test copy. Every CIM/registry access is intercepted by the harness below.
    source=source.replace(admin_line,'$admin=$true').replace(backup_line,"$root='"+str(tmp_path/'backup').replace("'","''")+"'")
    copy=tmp_path/'isolated-script.ps1';copy.write_text(source,encoding='utf-8')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(build_plan(configuration())),encoding='utf-8')
    harness=r'''
$ErrorActionPreference='Stop'
$script:auto=$true;$script:files=@('?:\pagefile.sys');$script:writes=@();$script:injected=INJECT_FAILURE
function Get-CimInstance { param($ClassName,$Filter)
 switch($ClassName) {
 'Win32_ComputerSystem' {[pscustomobject]@{AutomaticManagedPagefile=$script:auto}}
 'Win32_PageFileUsage' {[pscustomobject]@{Name='C:\pagefile.sys';AllocatedBaseSize=4096;CurrentUsage=20}}
 'Win32_LogicalDisk' {[pscustomobject]@{DriveType=3;FileSystem='NTFS';FreeSpace=120GB}}
 default {throw 'Unexpected CIM class'}
 }
}
function Get-ItemProperty {param($LiteralPath,$Name) [pscustomobject]@{PagingFiles=$script:files}}
function Set-CimInstance {param($InputObject,$Property) $script:auto=$Property.AutomaticManagedPagefile;$script:writes+=@('CIM');$InputObject}
function Set-ItemProperty {param($LiteralPath,$Name,$Value)
 if($script:injected){$script:injected=$false;throw 'simulated registry failure'}
 $script:files=@($Value);$script:writes+=@('Registry')
}
function Test-Path {param($LiteralPath)
 if($LiteralPath -eq 'F:\pagefile.sys'){return $false}
 Microsoft.PowerShell.Management\Test-Path -LiteralPath $LiteralPath
}
$failed=$false
try { . 'SCRIPT_PATH' -PlanFile 'PLAN_PATH' -Apply -Confirm:$false | Out-Null }
catch {if($_.Exception.Message -ne 'simulated registry failure'){throw};$failed=$true}
[pscustomobject]@{Failed=$failed;AutomaticManaged=$script:auto;PagingFiles=@($script:files);Writes=@($script:writes)}|ConvertTo-Json -Compress
'''
    harness=harness.replace('INJECT_FAILURE','$true' if inject_failure else '$false').replace('SCRIPT_PATH',str(copy).replace("'","''")).replace('PLAN_PATH',str(plan).replace("'","''"))
    r=subprocess.run(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',harness],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    actual=json.loads(r.stdout)
    assert actual['Failed']==inject_failure
    if inject_failure:
        assert actual['AutomaticManaged'] and actual['PagingFiles']==[r'?:\pagefile.sys']
        assert actual['Writes']==['CIM','CIM','Registry']
        assert not list((tmp_path/'backup').rglob('after.json'))
    else:
        assert not actual['AutomaticManaged']
        assert actual['PagingFiles']==[r'C:\pagefile.sys 0 0',r'F:\pagefile.sys 32768 32768']
        assert actual['Writes']==['CIM','Registry']
        after=json.loads(next((tmp_path/'backup').rglob('after.json')).read_text(encoding='utf-8-sig'))
        assert not after['RestartIssued'] and not after['WorkersStopped'] and not after['RuntimeCapacityIncreaseClaimed']
    before=json.loads(next((tmp_path/'backup').rglob('before.json')).read_text(encoding='utf-8-sig'))
    assert before['AutomaticManagedPagefile'] and before['PagingFiles']==[r'?:\pagefile.sys']
