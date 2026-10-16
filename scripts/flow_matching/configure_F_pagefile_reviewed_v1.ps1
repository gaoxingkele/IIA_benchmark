[CmdletBinding(SupportsShouldProcess=$true,ConfirmImpact='Medium')]
param(
    [Parameter(Mandatory=$true)][string]$PlanFile,
    [switch]$Apply
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$registry='HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$plan=Get-Content -LiteralPath $PlanFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.target -ne 'F:\pagefile.sys' -or $plan.initial_MB -ne 32768 -or $plan.maximum_MB -ne 32768) {
    throw 'Only the reviewed additional fixed32GiB F pagefile plan is supported.'
}
$computer=Get-CimInstance Win32_ComputerSystem
$beforeFiles=@((Get-ItemProperty -LiteralPath $registry -Name PagingFiles).PagingFiles)
$sameFiles=(ConvertTo-Json -InputObject $beforeFiles -Compress) -eq (ConvertTo-Json -InputObject @($plan.before.paging_files) -Compress)
if (-not $sameFiles -or $computer.AutomaticManagedPagefile -ne $plan.before.automatic_managed) {
    throw 'Pagefile configuration changed after review. Capture a fresh plan; do not overwrite it.'
}
$active=@(Get-CimInstance Win32_PageFileUsage)
$disk=Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='F:'"
if ($disk.DriveType -ne 3 -or $disk.FileSystem -ne 'NTFS' -or [long]$disk.FreeSpace -lt 96GB) {
    throw 'F must be fixed NTFS with32GiB allocation plus64GiB free reserve.'
}
if ((Test-Path -LiteralPath 'F:\pagefile.sys') -or @($active | Where-Object Name -EQ 'F:\pagefile.sys').Count) {
    throw 'An F pagefile already exists. This script will not overwrite it.'
}
$proposed=@()
foreach ($value in $beforeFiles) {
    if ($value -match '(?i)^F:\\pagefile\.sys(?:\s|$)') { throw 'F pagefile is already configured.' }
    if ($value.StartsWith('?:')) {
        if (-not $computer.AutomaticManagedPagefile -or -not $active.Count) { throw 'Cannot preserve an unbound wildcard.' }
        foreach ($page in $active) { $proposed+=($page.Name+' 0 0') }
    } else { $proposed+=$value }
}
if (-not $proposed.Count) { throw 'Preserve the existing boot-volume pagefile first.' }
if (-not @($proposed | Where-Object { $_ -match '(?i)^C:\\pagefile\.sys(?:\s|$)' }).Count) {
    throw 'This host-specific plan must preserve the existing C pagefile.'
}
$proposed=@($proposed | Select-Object -Unique)+@('F:\pagefile.sys 32768 32768')
if ((ConvertTo-Json -InputObject $proposed -Compress) -ne (ConvertTo-Json -InputObject @($plan.proposed_paging_files) -Compress)) {
    throw 'Recomputed safe configuration differs from the reviewed plan.'
}
$admin=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $Apply) {
    [pscustomobject]@{Mode='read_only_review';IsAdministrator=$admin;PlannedPagingFiles=$proposed;SettingsChanged=$false;RestartIssued=$false} | ConvertTo-Json -Depth 4
    return
}
if (-not $admin) { throw 'Administrator PowerShell is required. No settings were changed.' }
if (-not $PSCmdlet.ShouldProcess('Windows paging configuration','Preserve C/system pages and add fixed32GiB F pagefile for subsequent startup')) { return }
$root='F:\aicoding\IIA_Data\environments\windows_commit_capacity'
if (-not (Test-Path -LiteralPath $root)) { New-Item -ItemType Directory -Path $root | Out-Null }
$backup=Join-Path $root ('configuration-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
New-Item -ItemType Directory -Path $backup | Out-Null
$original=[ordered]@{AutomaticManagedPagefile=$computer.AutomaticManagedPagefile;PagingFiles=$beforeFiles;ActivePageFiles=@($active | Select-Object Name,AllocatedBaseSize,CurrentUsage)}
$original | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $backup 'before.json') -Encoding UTF8
try {
    Set-CimInstance -InputObject $computer -Property @{AutomaticManagedPagefile=$false} | Out-Null
    Set-ItemProperty -LiteralPath $registry -Name PagingFiles -Value ([string[]]$proposed)
    $actual=@((Get-ItemProperty -LiteralPath $registry -Name PagingFiles).PagingFiles)
    if ((ConvertTo-Json -InputObject $actual -Compress) -ne (ConvertTo-Json -InputObject $proposed -Compress)) { throw 'Configuration verification failed.' }
    if ((Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile) { throw 'Per-volume configuration was not persisted.' }
    $result=[ordered]@{SettingsChanged=$true;PlannedPagingFiles=$actual;Backup=$backup;RuntimeCapacityIncreaseClaimed=$false;RestartIssued=$false;WorkersStopped=$false;NextStep='Recheck actual commit limit. WMI configuration may require a later safe restart; do not restart ongoing experiments.'}
    $result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backup 'after.json') -Encoding UTF8
    $result | ConvertTo-Json -Depth 4
} catch {
    $failure=$_
    try {
        Set-CimInstance -InputObject (Get-CimInstance Win32_ComputerSystem) -Property @{AutomaticManagedPagefile=[bool]$original.AutomaticManagedPagefile} | Out-Null
        Set-ItemProperty -LiteralPath $registry -Name PagingFiles -Value ([string[]]$original.PagingFiles)
    } catch { throw ('Configuration failed; automatic restore also failed. Backup: '+$backup+'. Restore error: '+$_.Exception.Message) }
    throw $failure
}
