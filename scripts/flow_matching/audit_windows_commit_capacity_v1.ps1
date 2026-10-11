param(
    [string]$PlanPath = 'configs/runtime/fm_windows_commit_capacity_plan.v1.json',
    [switch]$ApplyPlan
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$resolvedPlan = Join-Path $repoRoot $PlanPath
$plan = Get-Content -LiteralPath $resolvedPlan -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.target_pagefile -ne 'D:\pagefile.sys' -or $plan.preserve_existing_system_managed_pagefile -ne 'C:\pagefile.sys' -or
    $plan.initial_size_mib -ne 65536 -or $plan.maximum_size_mib -ne 65536 -or
    $plan.minimum_target_free_space_after_bytes -ne 137438953472 -or $plan.automatic_restart -or $plan.automatic_process_stop) {
    throw 'Only the exact reviewed additional D: 64GiB pagefile plan is supported.'
}
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if ($ApplyPlan -and -not $admin) {
    throw 'Administrator PowerShell and SeCreatePagefilePrivilege are required. No system setting changed.'
}

function Get-CapacitySnapshot {
    $computer = Get-CimInstance Win32_ComputerSystem
    $registry = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management' -Name PagingFiles,ExistingPageFiles
    return [ordered]@{
        captured_utc = [DateTime]::UtcNow.ToString('o')
        administrator = $admin
        total_physical_memory_bytes = $computer.TotalPhysicalMemory
        automatic_managed_pagefile = $computer.AutomaticManagedPagefile
        pagefile_usage = @(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
        pagefile_settings = @(Get-CimInstance Win32_PageFileSetting | Select-Object Name,InitialSize,MaximumSize)
        registry_paging_files = @($registry.PagingFiles)
        registry_existing_pagefiles = @($registry.ExistingPageFiles)
        disks = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace)
    }
}

$before = Get-CapacitySnapshot
$targetDisk = @($before.disks | Where-Object { $_.DeviceID -eq 'D:' })
if ($targetDisk.Count -ne 1 -or
    $targetDisk[0].FreeSpace - [int64]$plan.initial_size_mib * 1MB -lt [int64]$plan.minimum_target_free_space_after_bytes) {
    throw 'The proposed pagefile must leave at least128GiB free on D:. No system setting changed.'
}
$targetPresent = @($before.pagefile_settings | Where-Object { $_.Name -ieq $plan.target_pagefile }).Count -gt 0 -or
    @($before.pagefile_usage | Where-Object { $_.Name -ieq $plan.target_pagefile }).Count -gt 0 -or
    (Test-Path -LiteralPath $plan.target_pagefile)
if ($ApplyPlan -and $targetPresent) { throw 'An existing D: pagefile/configuration is preserved; this add-only plan cannot overwrite it.' }
if ($ApplyPlan -and (-not $before.automatic_managed_pagefile -or $before.pagefile_settings.Count -ne 0 -or
    $before.pagefile_usage.Count -ne 1 -or $before.pagefile_usage[0].Name -ine $plan.preserve_existing_system_managed_pagefile)) {
    throw 'Current pagefile settings differ from the reviewed automatic C:-only baseline. No system setting changed.'
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$auditDirectory = Join-Path $repoRoot (Join-Path $plan.audit_root $stamp)
if (Test-Path -LiteralPath $auditDirectory) { throw 'Preserve every previous capacity-audit/administration record.' }
New-Item -ItemType Directory -Path $auditDirectory | Out-Null
$before | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $auditDirectory 'before.json') -Encoding UTF8
Copy-Item -LiteralPath $resolvedPlan -Destination (Join-Path $auditDirectory 'reviewed_plan.json')

if ($ApplyPlan) {
    # These are pagefile configuration operations, never filesystem deletion or native-process control.
    # The original C: pagefile remains system-managed; only the additional D: pagefile has a fixed64GiB size.
    try {
        $computer = Get-WmiObject Win32_ComputerSystem -EnableAllPrivileges
        $computer.AutomaticManagedPagefile = $false
        $null = $computer.Put()
        $null = Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
            Name = $plan.preserve_existing_system_managed_pagefile; InitialSize = 0; MaximumSize = 0
        } -EnableAllPrivileges
        $null = Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
            Name = $plan.target_pagefile; InitialSize = [uint32]$plan.initial_size_mib; MaximumSize = [uint32]$plan.maximum_size_mib
        } -EnableAllPrivileges
        $after = Get-CapacitySnapshot
        $configured = @($after.pagefile_settings | Where-Object { $_.Name -ieq $plan.target_pagefile })
        if ($configured.Count -ne 1 -or $configured[0].InitialSize -ne 65536 -or $configured[0].MaximumSize -ne 65536) {
            throw 'New pagefile configuration verification failed.'
        }
        $after | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $auditDirectory 'after_configuration.json') -Encoding UTF8
    } catch {
        $applyFailure = $_.Exception.Message
        # Restore only the exact prior registry configuration and automatic-management policy.
        # Never delete a pagefile, stop a process, or request a restart.
        try {
            $computer = Get-WmiObject Win32_ComputerSystem -EnableAllPrivileges
            $computer.AutomaticManagedPagefile = [bool]$before.automatic_managed_pagefile
            $null = $computer.Put()
            Set-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management' `
                -Name PagingFiles -Value ([string[]]$before.registry_paging_files)
            $rollback = Get-CapacitySnapshot
            $rollback | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $auditDirectory 'configuration_rollback.json') -Encoding UTF8
        } catch {
            throw "Apply failed: $applyFailure. Automatic configuration rollback also failed; inspect backup $auditDirectory. No restart or process stop was requested."
        }
        throw "Apply failed: $applyFailure. Previous automatic-management and registry configuration restored; inspect $auditDirectory. No restart requested."
    }
}

$receipt = [ordered]@{
    captured_utc = [DateTime]::UtcNow.ToString('o')
    audit_directory = $auditDirectory
    audit_only = -not [bool]$ApplyPlan
    administrator = $admin
    additional_pagefile_configuration_written = [bool]$ApplyPlan
    new_pagefile_runtime_allocation_proven = $false
    reboot_may_be_required_before_runtime_capacity_changes = $true
    reboot_requested = $false
    original_native_processes_stopped_or_modified = $false
    original_experimental_resource_guards_changed = $false
    raw_data_books_or_dataset_references_modified = $false
    old_configuration_backup = (Join-Path $auditDirectory 'before.json')
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $auditDirectory 'receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json -Depth 5
