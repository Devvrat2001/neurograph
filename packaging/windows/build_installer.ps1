#requires -version 5.1
<#
.SYNOPSIS
  P1-10 -- Build NeuroGraph.msi from a CMake Release build.

.DESCRIPTION
  Stages the runtime payload (workbench, predict, neurograph_rt, vcpkg
  Release DLLs, license RTF) into packaging/windows/staging/ and then
  invokes the WiX 5 toolchain to compile + link installer.wxs into
  build/installer/NeuroGraph-<version>.msi.

  This script is the *only* place the installer payload list lives --
  the WiX source globs whatever ends up in the staging directory. Add
  a file by editing this script, not the .wxs.

.PARAMETER BuildDir
  CMake build directory. Defaults to <repo>/build.

.PARAMETER Configuration
  CMake configuration name. Defaults to Release. Debug builds carry
  *D-suffix DLL names that don't match what the workbench DLL
  imports, so building a Debug-payload MSI is generally wrong.

.PARAMETER Version
  Semver string for the MSI's ProductVersion. Defaults to
  1.0.0. CI overrides this with a real release tag.

.PARAMETER Manufacturer
  Display name shown in Programs and Features and as the
  SmartScreen "Publisher" once the MSI is signed. Defaults to
  "NeuroGraph Inc." for local builds.

.NOTES
  Requires:
    * cmake build complete in $BuildDir/$Configuration/
    * `wix` global tool installed (`dotnet tool install --global wix`)
    * WixToolset.UI.wixext extension added (we add it on first run)
#>

param(
  [string]$BuildDir      = (Resolve-Path "$PSScriptRoot\..\..\build" -ErrorAction SilentlyContinue),
  [string]$Configuration = "Release",
  [string]$Version       = "1.0.0",
  [string]$Manufacturer  = "NeuroGraph Inc."
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Resolve absolute paths up front so wix.exe sees stable inputs.
$RepoRoot      = Resolve-Path "$PSScriptRoot\..\.."
$PackagingDir  = "$RepoRoot\packaging\windows"
$StagingDir    = "$PackagingDir\staging"
$OutputDir     = "$RepoRoot\build\installer"
$WxsFile       = "$PackagingDir\installer.wxs"
$LicenseRtf    = "$PackagingDir\LICENSE.rtf"
$ConfigBinDir  = "$BuildDir\$Configuration"
$VcpkgBinDir   = "$BuildDir\vcpkg_installed\x64-windows\bin"

if (-not (Test-Path $ConfigBinDir)) {
  throw "CMake $Configuration build not found at $ConfigBinDir. Run: cmake --build $BuildDir --config $Configuration --parallel"
}
if (-not (Test-Path $VcpkgBinDir)) {
  throw "vcpkg Release DLL dir not found at $VcpkgBinDir. Did vcpkg manifest mode finish bootstrapping?"
}

# ---------------------------------------------------------------------------
# 1. Stage the payload
# ---------------------------------------------------------------------------
Write-Host "==> Staging payload at $StagingDir" -ForegroundColor Cyan
if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

# Targets we author (workbench, predict CLI, C ABI runtime DLL).
$ProductFiles = @(
  "neurograph_workbench.exe",
  "neurograph_predict.exe",
  "neurograph_rt.dll"
)
foreach ($f in $ProductFiles) {
  $src = Join-Path $ConfigBinDir $f
  if (-not (Test-Path $src)) {
    throw "Missing build artifact $src -- build the corresponding CMake target first."
  }
  Copy-Item $src $StagingDir
}

# vcpkg Release runtime DLLs the workbench / predict / rt import.
# Determined by `dumpbin /DEPENDENTS` on each payload binary (see
# packaging/windows/check_deps.ps1). Re-check this list whenever
# you add a new vcpkg dep.
#
# Notes:
#   * libsodium is pulled in by the workbench's license/anti-tamper
#     path, not by predict.exe or neurograph_rt.dll.
#   * d3dcompiler_47.dll and d3d11.dll are Windows system DLLs that
#     ship in System32 on Windows 10/11 -- we deliberately do NOT
#     bundle them.
#   * libcurl.dll / zlib1.dll appear in build/Release/ from the
#     blanket POST_BUILD copy but no shipped binary actually
#     imports them, so they're omitted to keep the MSI lean.
$VcpkgDlls = @("spdlog.dll", "fmt.dll", "libsodium.dll")
foreach ($f in $VcpkgDlls) {
  $src = Join-Path $VcpkgBinDir $f
  if (-not (Test-Path $src)) {
    throw "Missing vcpkg DLL $src -- re-run cmake configure to refresh vcpkg_installed."
  }
  Copy-Item $src $StagingDir
}

Write-Host "    Staged files:"
Get-ChildItem $StagingDir | ForEach-Object { Write-Host ("      " + $_.Name) }

# ---------------------------------------------------------------------------
# 2. Make sure the WiX UI extension is registered for this user
# ---------------------------------------------------------------------------
Write-Host "==> Ensuring WixToolset.UI.wixext is installed" -ForegroundColor Cyan
& wix extension add -g WixToolset.UI.wixext 2>&1 | Out-Null
# Idempotent: if already added, wix prints a warning and returns 0.

# ---------------------------------------------------------------------------
# 3. Build the MSI
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$MsiPath = "$OutputDir\NeuroGraph-$Version.msi"
$PdbPath = "$OutputDir\NeuroGraph-$Version.wixpdb"

Write-Host "==> Building $MsiPath" -ForegroundColor Cyan
& wix build `
    -arch x64 `
    -ext WixToolset.UI.wixext `
    -d "Version=$Version" `
    -d "Manufacturer=$Manufacturer" `
    -d "PayloadDir=$StagingDir" `
    -d "LicenseRtf=$LicenseRtf" `
    -out $MsiPath `
    -pdb $PdbPath `
    $WxsFile

if ($LASTEXITCODE -ne 0) { throw "wix build failed (exit $LASTEXITCODE)" }

$size = (Get-Item $MsiPath).Length
Write-Host ("==> Built $MsiPath ({0:N1} MB)" -f ($size / 1MB)) -ForegroundColor Green

if ($size -gt 50MB) {
  Write-Warning "MSI size $([math]::Round($size/1MB,1)) MB exceeds the P1-10 acceptance budget of 50 MB."
}
