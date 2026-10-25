param([Parameter(Mandatory = $true)][string]$DataPath)
$ErrorActionPreference = 'Stop'

$dataRoot = (Resolve-Path -LiteralPath $DataPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $dataRoot 'worlds') -PathType Container)) {
    throw 'Choose the Foundry user Data directory containing worlds and modules.'
}
$sourceRoot = Join-Path $PSScriptRoot 'auto-wall-live'
$targetRoot = [IO.Path]::GetFullPath((Join-Path $dataRoot 'modules/auto-wall-live'))
if ((Test-Path -LiteralPath $targetRoot) -and (Get-Item -LiteralPath $targetRoot).LinkType) {
    throw 'The module destination is a link. Choose a regular module directory.'
}
New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null
$recordPath = Join-Path $targetRoot 'INSTALL-MANIFEST.json'
$previousFiles = if (Test-Path -LiteralPath $recordPath) {
    @((Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json).files)
} else { @() }
$files = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File
$manifest = @()
foreach ($file in $files) {
    $relative = $file.FullName.Substring($sourceRoot.Length + 1)
    $destination = [IO.Path]::GetFullPath((Join-Path $targetRoot $relative))
    if (-not $destination.StartsWith($targetRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'A module path escaped the destination.'
    }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    $installedHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($sourceHash -ne $installedHash) { throw "Installed file hash mismatch: $relative" }
    $manifest += [pscustomobject]@{path=$relative.Replace('\', '/'); sha256=$installedHash; bytes=$file.Length}
}
# Retire only files owned by the previous installation, with unchanged bytes.
# This removes the old detector on upgrade without touching unknown user files.
foreach ($previous in $previousFiles) {
    if ($previous.path.Replace('\', '/') -in $manifest.path) { continue }
    $obsoletePath = [IO.Path]::GetFullPath((Join-Path $targetRoot $previous.path))
    if (-not $obsoletePath.StartsWith($targetRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'An obsolete module path escaped the destination.'
    }
    if (-not (Test-Path -LiteralPath $obsoletePath)) { continue }
    $obsoleteFile = Get-Item -LiteralPath $obsoletePath
    $parent = $obsoleteFile.Directory
    while ($parent -and $parent.FullName -ne $targetRoot) {
        if ($parent.LinkType) { throw 'An obsolete module path contains a directory link.' }
        $parent = $parent.Parent
    }
    if ($obsoleteFile.PSIsContainer -or $obsoleteFile.LinkType -or
        (Get-FileHash -LiteralPath $obsoletePath -Algorithm SHA256).Hash -ne $previous.sha256) {
        throw "Obsolete module file was changed locally; preserve it before updating: $($previous.path)"
    }
    Remove-Item -LiteralPath $obsoletePath -Force
}
$moduleVersion = (Get-Content -LiteralPath (Join-Path $sourceRoot 'module.json') -Raw | ConvertFrom-Json).version
$record = [ordered]@{module='auto-wall-live'; version=$moduleVersion; installed_at_utc=[DateTime]::UtcNow.ToString('o'); files=$manifest}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding utf8
Write-Output "Installed and verified $($files.Count) files at $targetRoot."
Write-Output 'No world data or module activation settings were changed. Reload the Foundry client to use the update. For a first installation, restart Foundry when convenient, then enable Auto-Wall Live in Manage Modules.'
