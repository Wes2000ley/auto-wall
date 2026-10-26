$ErrorActionPreference = 'Stop'
$installer = Join-Path $PSScriptRoot '../install.ps1'
$temporaryParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar)
$testRoot = Join-Path $temporaryParent ('foundry-module-install-check-' + [guid]::NewGuid().ToString())
$checks = 0
function Assert-Install($condition, [string]$message) {
    if (-not $condition) { throw $message }
    $script:checks++
    Write-Output "PASS $message"
}
try {
    New-Item -ItemType Directory -Path (Join-Path $testRoot 'worlds') -Force | Out-Null
    & $installer -DataPath $testRoot
    $manualRecord = Get-Content -LiteralPath (Join-Path $testRoot 'modules/auto-wall-live/INSTALL-MANIFEST.json') -Raw | ConvertFrom-Json
    Assert-Install ($manualRecord.module -eq 'auto-wall-live') 'Default installer still selects Auto-Wall Live'
    & $installer -DataPath $testRoot -ModuleId map-enhancer
    $moduleRoot = Join-Path $testRoot 'modules/map-enhancer'
    $recordPath = Join-Path $moduleRoot 'INSTALL-MANIFEST.json'
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    Assert-Install ($record.module -eq 'map-enhancer') 'Explicit module selection installs Map Enhancer'
    $pins = Get-Content -LiteralPath (Join-Path $moduleRoot 'asset-pins.json') -Raw | ConvertFrom-Json
    foreach ($pin in $pins.files.PSObject.Properties) {
        $path = Join-Path $moduleRoot $pin.Name
        Assert-Install ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $pin.Value.sha256) "Installed bytes match the pin: $($pin.Name)"
    }
    $unknown = Join-Path $moduleRoot 'user-notes.txt'
    Set-Content -LiteralPath $unknown -Value 'User file: retain this' -Encoding utf8
    $obsolete = Join-Path $moduleRoot 'obsolete-worker.js'
    Set-Content -LiteralPath $obsolete -Value 'Owned obsolete runtime' -Encoding utf8
    $record.files += [pscustomobject]@{path='obsolete-worker.js'; sha256=(Get-FileHash -LiteralPath $obsolete -Algorithm SHA256).Hash}
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding utf8
    & $installer -DataPath $testRoot -ModuleId map-enhancer
    Assert-Install (-not (Test-Path -LiteralPath $obsolete)) 'Upgrade retires unchanged owned obsolete files'
    Assert-Install ((Get-Content -LiteralPath $unknown -Raw).Trim() -eq 'User file: retain this') 'Upgrade preserves unowned user files'
    Set-Content -LiteralPath $obsolete -Value 'User modified the old runtime' -Encoding utf8
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding utf8
    $refused = $false
    try { & $installer -DataPath $testRoot -ModuleId map-enhancer }
    catch { $refused = $_.Exception.Message -like '*changed locally*' }
    Assert-Install $refused 'Upgrade refuses to delete modified obsolete files'
    Assert-Install ((Get-Content -LiteralPath $obsolete -Raw).Trim() -eq 'User modified the old runtime') 'Refused upgrade retains the changed file'
    Write-Output "$checks installer checks passed."
} finally {
    # Delete only this run's verified, uniquely named test directory.
    $resolved = [IO.Path]::GetFullPath($testRoot)
    if (-not $resolved.StartsWith($temporaryParent + [IO.Path]::DirectorySeparatorChar) -or
        -not ([IO.Path]::GetFileName($resolved) -match '^foundry-module-install-check-[0-9a-f-]{36}$')) {
        throw 'Refusing to clean up an unexpected test directory.'
    }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
