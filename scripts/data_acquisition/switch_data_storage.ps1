param(
    [string]$RepositoryRoot = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$PythonExecutable = 'python'
)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\')
$configuration = Get-Content -LiteralPath (Join-Path $workspace 'configs\storage\data_storage.v1.json') -Raw | ConvertFrom-Json
$source = [IO.Path]::GetFullPath((Join-Path $workspace 'data\public_datasets'))
$destination = [IO.Path]::GetFullPath($configuration.mounts[0].physical_path)
$stateDirectory = [IO.Path]::GetFullPath($configuration.migration_state_directory)
$retired = $source + '.retired_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$dataBoundary = [IO.Path]::GetFullPath((Join-Path $workspace 'data')).TrimEnd('\') + '\'
$reportPath = [IO.Path]::GetFullPath((Join-Path $workspace $configuration.migration_report))
$reportBoundary = [IO.Path]::GetFullPath((Join-Path $workspace 'docs\reports')).TrimEnd('\') + '\'
if (-not $reportPath.StartsWith($reportBoundary, [StringComparison]::OrdinalIgnoreCase)) { throw 'Migration report must remain under workspace docs/reports.' }
if (-not $source.StartsWith($dataBoundary, [StringComparison]::OrdinalIgnoreCase) -or
    -not $retired.StartsWith($dataBoundary, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source and retired directory must remain inside the intended workspace data directory.'
}
if ($destination.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Physical destination must be outside the repository.'
}
if (-not (Test-Path -LiteralPath $destination -PathType Container)) { throw 'Verified destination is missing.' }
if (Test-Path -LiteralPath $retired) { throw 'Retired directory already exists; preserve it.' }
$sourceItem = Get-Item -LiteralPath $source -Force
if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Source is already a mount; no cutover performed.' }
$dataItem = Get-Item -LiteralPath (Split-Path $source -Parent) -Force
if ($dataItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Source parent may not redirect outside the workspace.' }
$statusFile = Join-Path $stateDirectory 'status.json'
$copyReport = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
if ($copyReport.status -ne 'verified_ready_for_cutover' -or -not $copyReport.audit.complete -or $copyReport.errors.Count) {
    throw 'All copied files must pass the migration audit before cutover.'
}
if ([IO.Path]::GetFullPath($copyReport.source) -ne $source -or
    [IO.Path]::GetFullPath($copyReport.destination) -ne $destination) { throw 'Audited paths do not match configured paths.' }
$copyEvidence = Join-Path $stateDirectory 'copy_completion_status.json'
if (-not (Test-Path -LiteralPath $copyEvidence)) { Copy-Item -LiteralPath $statusFile -Destination $copyEvidence }

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class IIAMigrationProcess {
    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("ntdll.dll")] static extern int NtSuspendProcess(IntPtr handle);
    [DllImport("ntdll.dll")] static extern int NtResumeProcess(IntPtr handle);
    public static int Change(uint pid, bool suspend) {
        IntPtr handle = OpenProcess(0x0800, false, pid);
        if (handle == IntPtr.Zero) return -1;
        try { return suspend ? NtSuspendProcess(handle) : NtResumeProcess(handle); }
        finally { CloseHandle(handle); }
    }
}
'@
$suspended = New-Object 'System.Collections.Generic.List[uint32]'
$resumeFailures = New-Object 'System.Collections.Generic.List[uint32]'
$junctionCreated = $false
$started = (Get-Date).ToUniversalTime().ToString('o')
try {
    $processes = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"
    foreach ($process in $processes) {
        if ($process.CommandLine -and $process.CommandLine.Contains($workspace) -and
            $process.CommandLine -notmatch 'migrate_data_storage.py') {
            if ([IIAMigrationProcess]::Change($process.ProcessId, $true) -ne 0) { throw 'Could not briefly suspend an active repository workflow.' }
            $suspended.Add($process.ProcessId)
        }
    }
    # Recheck every source/destination signature with writers briefly quiesced.
    & $PythonExecutable (Join-Path $PSScriptRoot 'migrate_data_storage.py') --source $source --destination $destination --state $stateDirectory --audit-only
    if ($LASTEXITCODE -ne 0) { throw 'Source or destination changed during migration; original data retained.' }
    Move-Item -LiteralPath $source -Destination $retired
    try {
        New-Item -ItemType Junction -Path $source -Target $destination | Out-Null
        $junction = Get-Item -LiteralPath $source -Force
        if (-not ($junction.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            [IO.Path]::GetFullPath($junction.Target[0]) -ne $destination) { throw 'Junction verification failed.' }
        $junctionCreated = $true
    } catch {
        if (Test-Path -LiteralPath $source) {
            $failedLink = Get-Item -LiteralPath $source -Force
            if ($failedLink.Attributes -band [IO.FileAttributes]::ReparsePoint) { [IO.Directory]::Delete($source) }
            else { throw 'Unexpected source path created; both copies preserved for recovery.' }
        }
        Move-Item -LiteralPath $retired -Destination $source
        throw
    }
} finally {
    foreach ($processId in $suspended) {
        if ([IIAMigrationProcess]::Change($processId, $false) -ne 0 -and (Get-Process -Id $processId -ErrorAction SilentlyContinue)) {
            $resumeFailures.Add($processId)
        }
    }
}
if ($resumeFailures.Count) { throw ('Workflow resume failed: ' + ($resumeFailures -join ',')) }
if (-not $junctionCreated) { throw 'Cutover was not verified; source cleanup forbidden.' }

# The only recursive deletion is the explicitly checked, retired D-drive copy.
$retiredItem = Get-Item -LiteralPath $retired -Force
if ([IO.Path]::GetFullPath($retiredItem.FullName) -ne $retired -or
    -not $retired.StartsWith($dataBoundary, [StringComparison]::OrdinalIgnoreCase) -or
    ($retiredItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unsafe retired-copy cleanup target.' }
$cleanupErrors = @()
Remove-Item -LiteralPath $retired -Recurse -Force -ErrorAction SilentlyContinue -ErrorVariable cleanupErrors
$removed = -not (Test-Path -LiteralPath $retired)
$ledger = Join-Path $stateDirectory 'verified_files.jsonl'
$report = [ordered]@{
    schema_version = 1
    status = $(if ($removed) { 'completed' } else { 'cutover_complete_cleanup_pending' })
    source_logical_path = $source
    physical_data_directory = $destination
    compatibility = 'Configured Windows junction; frozen registrations and queue hashes preserved'
    sha256_verified_file_count = $copyReport.audit.file_count
    data_bytes = $copyReport.audit.bytes
    copy_completion_evidence = $copyEvidence
    per_file_sha256_ledger = $ledger
    ledger_sha256 = (Get-FileHash -LiteralPath $ledger -Algorithm SHA256).Hash.ToLower()
    source_copy_removed = $removed
    retired_directory = $retired
    suspended_then_resumed_process_ids = @($suspended.ToArray())
    workflow_resume_failures = @($resumeFailures.ToArray())
    cleanup_errors = @($cleanupErrors | ForEach-Object { $_.ToString() })
    cutover_started_at = $started
    completed_at = (Get-Date).ToUniversalTime().ToString('o')
    boundary = 'Storage relocation only; raw files, incomplete downloads and experimental definitions preserved.'
}
New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force | Out-Null
[IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
Write-Output ($report | ConvertTo-Json -Depth 8)
if (-not $removed) { exit 1 }
