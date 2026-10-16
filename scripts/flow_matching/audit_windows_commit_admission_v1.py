"""Read-only Windows commit-capacity diagnosis and a reviewable pagefile plan.

Never allocates pressure, changes Windows settings, lowers experiment gates,
stops workers, or restarts Windows. Projected capacity is not actual admission.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.cached_resource_snapshot_v1 import PerformanceInfo,windows_api

GIB=2**30
PROFILES=[('TranAD_author3270','configs/experiments/fm_tranad_author_all_entities.v1.json','settings'),
    ('CFM_full400_epoch280_resume','configs/runtime/fm_cfm_node1106_epoch280_resume.v1.json','resource_policy'),
    ('matched_TAB577_resource_recovery','configs/reproducibility/fm_common105_TAB577_resource_cell_recovery.2026-10-16_v1.json','settings'),
    ('EB_CITE150_native_replay','configs/reproducibility/fm_otcfm_EB_CITE_full150_checkpoint_replay.v1.json','settings')]


def memory_snapshot():
    class Status(ctypes.Structure):
        _fields_=[('length',w.DWORD),('load',w.DWORD)]+[(k,ctypes.c_ulonglong) for k in
            ['totalphys','availphys','totalpage','availpage','totalvirtual','availvirtual','extended']]
    a=Status();a.length=ctypes.sizeof(a)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(a)):raise ctypes.WinError()
    p=PerformanceInfo();p.cb=ctypes.sizeof(p)
    if not windows_api()(ctypes.byref(p),p.cb):raise ctypes.WinError()
    commit=(p.CommitLimit-p.CommitTotal)*p.PageSize
    if abs(commit-a.availpage)>512*2**20:raise ValueError('Independent live readings differ; capture again without changing gates')
    return dict(physical_total_bytes=p.PhysicalTotal*p.PageSize,physical_available_bytes=p.PhysicalAvailable*p.PageSize,
        commit_total_bytes=p.CommitTotal*p.PageSize,commit_limit_bytes=p.CommitLimit*p.PageSize,commit_available_bytes=commit,
        second_API_commit_available_bytes=a.availpage,independent_API_readings_agree=True)


def system_configuration():
    command=r"""$ErrorActionPreference='Stop';[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$registry='HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$disk=Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='F:'"
[pscustomobject]@{
 automatic_managed=(Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile
 paging_files=@((Get-ItemProperty -LiteralPath $registry -Name PagingFiles).PagingFiles)
 active_pagefiles=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
 F_disk=[pscustomobject]@{drive_type=$disk.DriveType;file_system=$disk.FileSystem;free_bytes=[long]$disk.FreeSpace}
 is_administrator=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
 target_file_exists=(Test-Path -LiteralPath 'F:\pagefile.sys')
} | ConvertTo-Json -Depth 5 -Compress
"""
    return json.loads(subprocess.check_output(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',command],encoding='utf-8'))


def build_plan(configuration,additional_GiB=32):
    if additional_GiB!=32:raise ValueError('This reviewed plan adds exactly32GiB on F,never replaces C')
    disk=configuration['F_disk']
    if disk['drive_type']!=3 or disk['file_system']!='NTFS' or disk['free_bytes']<96*GIB:
        raise ValueError('F must be fixed NTFS with32GiB allocation plus64GiB free reserve')
    if configuration['target_file_exists'] or any(x['Name'].lower()==r'f:\pagefile.sys' for x in configuration['active_pagefiles']):
        raise ValueError('Never overwrite or resize an existing F pagefile')
    old=list(configuration['paging_files']);proposed=[]
    for value in old:
        if re.match(r'(?i)^f:\\pagefile\.sys(?:\s|$)',value):raise ValueError('F pagefile already configured')
        if value.lower().startswith('?:'):
            if not configuration['automatic_managed'] or not configuration['active_pagefiles']:
                raise ValueError('Wildcard requires actual active system-managed volume evidence')
            for active in configuration['active_pagefiles']:
                proposed.append(active['Name']+' 0 0')
        else:proposed.append(value)
    if not proposed:raise ValueError('Existing boot-volume pagefile must be explicitly preserved')
    c_entries=[x for x in proposed if re.match(r'(?i)^c:\\pagefile\.sys(?:\s|$)',x)]
    if not c_entries:raise ValueError('This host-specific plan requires preserving the existing C pagefile')
    proposed=list(dict.fromkeys(proposed));proposed.append(r'F:\pagefile.sys 32768 32768')
    return dict(schema_version=1,target=r'F:\pagefile.sys',additional_pagefile_bytes=32*GIB,initial_MB=32768,maximum_MB=32768,
        reserve_free_bytes=64*GIB,before=configuration,proposed_paging_files=proposed,proposed_automatic_managed=False,
        existing_C_pagefile_preserved=True,existing_C_volume_remains_system_managed=all(x.split()[-2:]==['0','0'] for x in c_entries),
        configuration_only_not_a_runtime_capacity_increase=True,administrator_required=True,
        no_reboot_or_worker_stop=True,no_threshold_budget_precision_or_tolerance_changes=True)


def eligibility(settings,memory,additional_bytes=0):
    physical=memory['physical_available_bytes']>=settings['minimum_available_memory_bytes']
    commit=memory['commit_available_bytes']+additional_bytes>=settings['minimum_commit_headroom_bytes']
    return dict(physical_gate_passes=physical,commit_gate_passes=commit,memory_gates_pass=physical and commit,
        actual_experiment_admitted=False,projection_only=bool(additional_bytes),
        commit_available_holding_charge_fixed=memory['commit_available_bytes']+additional_bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    configuration=system_configuration();memory=memory_snapshot();plan=build_plan(configuration)
    records=[]
    for name,path,field in PROFILES:
        c=read(ROOT/path);settings=c[field]
        records.append(dict(profile=name,configuration=path,configuration_sha256=sha(ROOT/path),settings=settings,
            observed=eligibility(settings,memory),counterfactual_with32GiB_pagefile_holding_usage_fixed=eligibility(settings,memory,32*GIB)))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for filename,value in [('reviewable_pagefile_plan.json',plan),('read_only_capacity_diagnosis.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),memory=memory,system_configuration=configuration,profiles=records,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        Windows_settings_changed=False,numerical_processes_stopped=False,all_original_experiments_complete=False))]:
        with (out/filename).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(out),commit_limit_GiB=memory['commit_limit_bytes']/GIB,commit_available_GiB=memory['commit_available_bytes']/GIB,
        administrator=configuration['is_administrator'],settings_changed=False)))


if __name__=='__main__':main()
