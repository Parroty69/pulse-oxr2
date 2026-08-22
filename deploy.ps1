$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
& $Python (Join-Path $RepoRoot "scripts/deploy.py") @args
exit $LASTEXITCODE
