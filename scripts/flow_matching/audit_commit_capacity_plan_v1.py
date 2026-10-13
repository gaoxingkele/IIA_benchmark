"""Read Windows commit capacity and frozen experiment admission thresholds.

This module never changes paging settings, launches models or restarts Windows.
The plan is a capacity estimate at the measured load, not benchmark performance.
"""
import argparse
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from scripts.flow_matching.cached_resource_snapshot_v1 import PerformanceInfo,windows_api,resource_snapshot

ROOT=Path(__file__).resolve().parents[2]
GIB=2**30


def projection(limit,used,additional,required):
    if min(limit,used,additional,required)<0 or used>limit:raise ValueError('Valid nonnegative measured capacity required')
    available=limit-used
    return dict(current_available_bytes=available,current_admission_met=available>=required,
        proposed_available_bytes=available+additional,proposed_admission_met=available+additional>=required,
        proposal_is_not_applied=True)


def capture(config):
    shell=r'''$taskPlanAudit=[ordered]@{Computer=(Get-CimInstance Win32_ComputerSystem|Select-Object TotalPhysicalMemory,AutomaticManagedPagefile);PageFiles=@(Get-CimInstance Win32_PageFileUsage|Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage);Disks=@(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"|Select-Object DeviceID,Size,FreeSpace);IsAdministrator=([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)};$taskPlanAudit|ConvertTo-Json -Depth 4'''
    system=json.loads(subprocess.check_output(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',shell],encoding='utf-8'))
    info=PerformanceInfo();info.cb=ctypes.sizeof(info)
    if not windows_api()(ctypes.byref(info),info.cb):raise ctypes.WinError(ctypes.get_last_error())
    limit,used=info.CommitLimit*info.PageSize,info.CommitTotal*info.PageSize
    physical=resource_snapshot()['physical_available_bytes'];rows=[]
    for role in config['admission_roles']:
        value=json.loads((ROOT/role['configuration']).read_text())
        for key in role['threshold_path']:value=value[key]
        required=int(value)
        rows.append(dict(role=role['name'],configuration=role['configuration'],threshold_path=role['threshold_path'],required_headroom_bytes=required,
            **projection(limit,used,config['proposed_initial_MiB']*2**20,required)))
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),windows_settings=system,
        current_commit_limit_bytes=limit,current_commit_used_bytes=used,current_commit_available_bytes=limit-used,
        current_physical_available_bytes=physical,admission_roles=rows,
        proposed_pagefile={'path':config['proposed_pagefile_path'],'initial_MiB':config['proposed_initial_MiB'],'maximum_MiB':config['proposed_maximum_MiB'],
            'retain_C_system_managed':True,'requires_privileged_user_action':True,'activation_requires_runtime_reverification':True},
        automatic_growth_reference_commit_fraction=.9,
        guard_abort_commit_fraction_at_16GiB_reserve=(limit-16*GIB)/limit,
        diagnosis_boundary='Measured present commit margin is below frozen admission thresholds. Projection assumes C remains unchanged and F adds the stated backing capacity, with unchanged commit use. It does not prove physical/GPU capacity or predict timing of Windows automatic growth.',
        windows_configuration_modified=False,registry_modified=False,experiments_stopped=False,reboot_performed=False,
        experiment_budgets_guards_or_tolerances_changed=False,all_original_experiments_complete=False,
        official_sources=config['official_sources'])


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text());report=capture(config)
    with (ROOT/config['output_report']).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(dict(commit_limit_GiB=report['current_commit_limit_bytes']/GIB,available_commit_GiB=report['current_commit_available_bytes']/GIB,
        administrator=report['windows_settings']['IsAdministrator'],system_modified=False))


if __name__=='__main__':main()
