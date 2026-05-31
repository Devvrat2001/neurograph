# Convenience wrapper that adds wix.exe to PATH and runs the build.
# Used by the local-dev workflow when wix isn't on the user's PATH yet.
$env:PATH = "$env:USERPROFILE\.dotnet\tools;$env:PATH"
& "$PSScriptRoot\build_installer.ps1" @args
exit $LASTEXITCODE
