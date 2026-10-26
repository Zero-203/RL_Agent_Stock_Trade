#Requires -Version 5.1
<#
.SYNOPSIS
	One-click environment setup for the FinRL group project (Python 3.9 + pip).

.DESCRIPTION
	Steps:
	  1. Locate a Python 3.9 interpreter (py launcher -> PATH -> common install dirs).
	  2. Create/refresh .venv in the repository root.
	  3. Upgrade pip, install pinned dependencies from requirements-lock.txt.
	  4. Install the local finrl package in editable mode.
	  5. Install ElegantRL from GitHub (original env used a now-deleted local copy;
		 pip lock only records versions, so ElegantRL is installed from source here).
	  6. Smoke-test: import finrl and print key package versions.

.USAGE
	powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
	powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -SkipElegantRL
	powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -PythonPath "C:\Python39\python.exe"
#>
param(
	[string]$PythonPath = "",
	[switch]$SkipElegantRL
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LockFile = Join-Path $RepoRoot "requirements-lock.txt"
$VenvDir  = Join-Path $RepoRoot ".venv"
$VenvPy   = Join-Path $VenvDir "Scripts\python.exe"

function Write-Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

function Find-Python39 {
	if ($PythonPath -ne "") {
		if (-not (Test-Path $PythonPath)) { throw "PythonPath not found: $PythonPath" }
		return $PythonPath
	}
	$candidates = @()
	if (Get-Command py -ErrorAction SilentlyContinue) {
		$candidates += "py -3.9"
	}
	$candidates += "python"
	$localPy39 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python39\python.exe"
	if (Test-Path $localPy39) { $candidates += "`"$localPy39`"" }

	foreach ($c in $candidates) {
		try {
			$ver = & cmd /c "$c -c ""import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')""" 2>$null
			if ($ver -match "^3\.9") {
				if ($c -like "py *") { return (& py -3.9 -c "import sys;print(sys.executable)") }
				return (& $c -c "import sys;print(sys.executable)")
			}
		} catch { }
	}
	throw @"
Python 3.9 not found.
Install Python 3.9.x from https://www.python.org/downloads/release/python-3913/
(tick 'Add python.exe to PATH'), then re-run this script,
or pass -PythonPath "C:\path\to\python.exe".
"@
}

Write-Step "Locating Python 3.9"
$py39 = Find-Python39
Write-Host "Using: $py39"
& $py39 --version

if (Test-Path $VenvPy) {
	Write-Step ".venv already exists, reusing"
} else {
	Write-Step "Creating .venv"
	& $py39 -m venv $VenvDir
}

Write-Step "Upgrading pip"
& $VenvPy -m pip install --upgrade pip

Write-Step "Installing pinned dependencies from requirements-lock.txt"
& $VenvPy -m pip install -r $LockFile

Write-Step "Installing finrl (editable, no deps)"
& $VenvPy -m pip install -e $RepoRoot --no-deps

if ($SkipElegantRL) {
	Write-Step "Skipping ElegantRL (-SkipElegantRL)"
} else {
	Write-Step "Installing ElegantRL from GitHub"
	& $VenvPy -m pip install "git+https://github.com/AI4Finance-Foundation/ElegantRL.git"
}

Write-Step "Smoke test"
& $VenvPy -c "import finrl, stable_baselines3, torch, pandas, numpy; print('finrl OK at', finrl.__file__); print('sb3', stable_baselines3.__version__, '| torch', torch.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__)"

Write-Host "`nDone. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Note: TA-Lib is not required by the default tutorial pipeline (stockstats computes indicators)."
