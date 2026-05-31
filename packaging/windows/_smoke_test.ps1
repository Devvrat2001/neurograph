#requires -version 5.1
<#
.SYNOPSIS
  P1-10 -- Smoke test silent install / uninstall of the NeuroGraph MSI.

.DESCRIPTION
  Runs `msiexec /i ... /quiet`, verifies every payload file lands at
  C:\Program Files\NeuroGraph\, then runs `msiexec /x ... /quiet` and
  verifies the install dir is gone. Mirrors the equivalent step in
  .github/workflows/release-windows.yml so devs can rehearse the
  acceptance criterion locally before pushing a release tag.

  Must be run from an elevated PowerShell because msiexec /i for a
  per-machine MSI requires admin.
#>
param(
  [string]$MsiPath = $null
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $MsiPath) {
  $repoRoot = Resolve-Path "$PSScriptRoot\..\.."
  $candidate = Get-ChildItem "$repoRoot\build\installer\NeuroGraph-*.msi" |
               Sort-Object LastWriteTime -Descending |
               Select-Object -First 1
  if (-not $candidate) { throw "No MSI found under build/installer/. Run build_installer.ps1 first." }
  $MsiPath = $candidate.FullName
}
if (-not (Test-Path $MsiPath)) { throw "MSI not found: $MsiPath" }

# Refuse to run unelevated -- otherwise msiexec silently fails with 1603.
$current = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($current)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw "This script must run elevated (Run as Administrator). msiexec /i for a per-machine MSI requires admin."
}

$installLog   = Join-Path $env:TEMP "neurograph-msi-install.log"
$uninstallLog = Join-Path $env:TEMP "neurograph-msi-uninstall.log"
$installDir   = "C:\Program Files\NeuroGraph"

if (Test-Path $installDir) {
  Write-Warning "$installDir already exists -- a prior install may not have been cleaned up. Continuing anyway."
}

Write-Host "==> Silent install $MsiPath" -ForegroundColor Cyan
$p = Start-Process msiexec.exe `
       -ArgumentList @("/i", $MsiPath, "/quiet", "/norestart", "/l*v", $installLog) `
       -Wait -PassThru
if ($p.ExitCode -ne 0) {
  Write-Host "Install failed (exit $($p.ExitCode)). Last 60 lines of log:" -ForegroundColor Red
  Get-Content $installLog -Tail 60
  throw "msiexec /i exited $($p.ExitCode) -- see $installLog"
}

Write-Host "==> Verify payload at $installDir" -ForegroundColor Cyan
$expected = @(
  "neurograph_workbench.exe",
  "neurograph_predict.exe",
  "neurograph_rt.dll",
  "spdlog.dll", "fmt.dll", "libsodium.dll"
)
foreach ($f in $expected) {
  $path = Join-Path $installDir $f
  if (-not (Test-Path $path)) { throw "Missing after install: $path" }
  Write-Host ("    OK " + $f)
}

# Locate ProductCode so we can /x without pinning a GUID. Get-WmiObject
# Win32_Product is slow but reliable; the ProductCode also lives under
# HKLM\Software\Microsoft\Windows\CurrentVersion\Uninstall but the
# WMI path is the canonical MSI lookup.
Write-Host "==> Resolving ProductCode" -ForegroundColor Cyan
$product = Get-WmiObject Win32_Product -Filter "Name = 'NeuroGraph'" |
           Select-Object -First 1
if (-not $product) { throw "NeuroGraph not registered in MSI database after install" }
Write-Host ("    ProductCode = " + $product.IdentifyingNumber)

Write-Host "==> Silent uninstall" -ForegroundColor Cyan
$p = Start-Process msiexec.exe `
       -ArgumentList @("/x", $product.IdentifyingNumber, "/quiet", "/norestart", "/l*v", $uninstallLog) `
       -Wait -PassThru
if ($p.ExitCode -ne 0) {
  Write-Host "Uninstall failed (exit $($p.ExitCode)). Last 60 lines of log:" -ForegroundColor Red
  Get-Content $uninstallLog -Tail 60
  throw "msiexec /x exited $($p.ExitCode) -- see $uninstallLog"
}
if (Test-Path $installDir) {
  throw "Install dir still present after uninstall: $installDir"
}

Write-Host "==> Smoke test passed" -ForegroundColor Green
