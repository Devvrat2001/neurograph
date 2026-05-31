#requires -version 5.1
<#
.SYNOPSIS
  P1-10 -- Sign the NeuroGraph MSI with a code-signing certificate.

.DESCRIPTION
  Wraps signtool.exe. Reads the certificate location and password from
  environment variables so we never bake secrets into the script:

    NEUROGRAPH_SIGN_CERT_PATH  -- path to .pfx, OR
    NEUROGRAPH_SIGN_CERT_HASH  -- SHA1 thumbprint of an installed cert
    NEUROGRAPH_SIGN_CERT_PASS  -- password for the .pfx (only used with
                                  CERT_PATH)
    NEUROGRAPH_SIGN_TS_URL     -- RFC 3161 timestamp authority URL.
                                  Defaults to DigiCert.

  When neither CERT_PATH nor CERT_HASH is set, this script logs a
  warning and exits 0. That is intentional -- local dev builds and CI
  forks without secret access should produce an unsigned MSI rather
  than fail the build. Production CI gates the release on
  signing being available.

.PARAMETER MsiPath
  Path to the .msi to sign. Defaults to the most recent
  NeuroGraph-*.msi in build/installer/.
#>

param(
  [string]$MsiPath = $null
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Default to the newest MSI in the build/installer directory.
if (-not $MsiPath) {
  $repoRoot = Resolve-Path "$PSScriptRoot\..\.."
  $candidate = Get-ChildItem "$repoRoot\build\installer\NeuroGraph-*.msi" `
               -ErrorAction SilentlyContinue |
               Sort-Object LastWriteTime -Descending |
               Select-Object -First 1
  if (-not $candidate) {
    throw "No MsiPath given and no NeuroGraph-*.msi found under build/installer/."
  }
  $MsiPath = $candidate.FullName
}
if (-not (Test-Path $MsiPath)) { throw "MSI not found: $MsiPath" }

$certPath = $env:NEUROGRAPH_SIGN_CERT_PATH
$certHash = $env:NEUROGRAPH_SIGN_CERT_HASH
$certPass = $env:NEUROGRAPH_SIGN_CERT_PASS
$tsUrl    = if ($env:NEUROGRAPH_SIGN_TS_URL) { $env:NEUROGRAPH_SIGN_TS_URL }
            else { "http://timestamp.digicert.com" }

if (-not $certPath -and -not $certHash) {
  Write-Warning "No signing cert configured (NEUROGRAPH_SIGN_CERT_PATH / NEUROGRAPH_SIGN_CERT_HASH unset). Skipping signature -- MSI will trigger SmartScreen 'Unknown publisher' on install."
  exit 0
}

# Locate signtool.exe (ships with the Windows SDK).
$signtool = Get-ChildItem `
    "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" `
    -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending |
    Select-Object -First 1
if (-not $signtool) {
  throw "signtool.exe not found. Install the Windows 10/11 SDK."
}

Write-Host "==> Signing $MsiPath" -ForegroundColor Cyan
$args = @(
  "sign",
  "/fd", "SHA256",            # file digest algorithm
  "/td", "SHA256",            # timestamp digest algorithm
  "/tr", $tsUrl,
  "/d",  "NeuroGraph Workbench Installer"
)
if ($certPath) {
  $args += @("/f", $certPath)
  if ($certPass) { $args += @("/p", $certPass) }
} else {
  # Use a cert from the local user / machine store by SHA1 thumbprint.
  $args += @("/sha1", $certHash)
}
$args += $MsiPath

& $signtool.FullName @args
if ($LASTEXITCODE -ne 0) {
  throw "signtool sign failed (exit $LASTEXITCODE)"
}

# Verify the signature is well-formed and the timestamp authority
# countersigned successfully -- this catches bad timestamp servers
# and corrupted certs without waiting for QA on a clean VM.
& $signtool.FullName verify /pa /v $MsiPath
if ($LASTEXITCODE -ne 0) {
  throw "signtool verify failed (exit $LASTEXITCODE)"
}

Write-Host "==> Signed and verified" -ForegroundColor Green
