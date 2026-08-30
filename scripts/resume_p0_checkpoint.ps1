param(
    [ValidateSet("casim", "cone", "bip", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $RepoRoot

python scripts\paper_exact.py check --require-local --full-hash
python -m pytest -q tests\test_paper_exact.py

if ($Target -in @("casim", "all")) {
    & .\tmp\codeocean_envs\casim\Scripts\python.exe `
        experiments\paper_harness\p0_paper_exact\paper_grid.py `
        run-casim --workers 8 --repetitions 10
}

if ($Target -in @("cone", "all")) {
    & .\tmp\codeocean_envs\cone\Scripts\python.exe `
        experiments\paper_harness\p0_paper_exact\paper_grid.py `
        run-cone --workers 6
}

if ($Target -in @("bip", "all")) {
    python scripts\paper_exact.py run-author `
        --paper-id faulwasser2025_uncertainty_reduction `
        --engine native-windows
}
