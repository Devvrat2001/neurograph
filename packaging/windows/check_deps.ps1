#requires -version 5.1
# Throwaway helper: list runtime DLL dependencies of the installer payload.
$ErrorActionPreference = "Stop"

$dumpbin = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $dumpbin) {
    $dumpbin = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1
}
if (-not $dumpbin) { throw "dumpbin.exe not found under VS 2022 installs" }

$repoRoot = Resolve-Path "$PSScriptRoot\..\.."
$bin      = Join-Path $repoRoot "build\Release"

foreach ($f in @("neurograph_workbench.exe", "neurograph_train.exe", "neurograph_predict.exe", "neurograph_rt.dll")) {
    Write-Host ""
    Write-Host "== $f ==" -ForegroundColor Cyan
    & $dumpbin.FullName /DEPENDENTS (Join-Path $bin $f) |
        Select-String '\.dll' |
        Where-Object { $_ -notmatch 'Dump of file' } |
        ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
}
