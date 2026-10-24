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
$record = [ordered]@{module='auto-wall-live'; version='0.1.0'; installed_at_utc=[DateTime]::UtcNow.ToString('o'); files=$manifest}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $targetRoot 'INSTALL-MANIFEST.json') -Encoding utf8
Write-Output "Installed and verified $($files.Count) files at $targetRoot."
Write-Output 'No world data or module activation settings were changed. Restart Foundry when convenient to discover the new module, then enable Auto-Wall Live in Manage Modules.'
