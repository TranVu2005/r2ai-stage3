$ErrorActionPreference = 'Stop'
$r2aiRepo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $r2aiRepo
$r2aiGitRoot = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0 -or (Resolve-Path -LiteralPath $r2aiGitRoot).Path -ne (Resolve-Path -LiteralPath $r2aiRepo).Path) { throw 'Run setup in the standalone R2AI checkout' }
if ((uv --version) -ne 'uv 0.11.28') { throw 'This environment was verified with uv 0.11.28' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    uv venv --python 3.12.13 .venv
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
}
uv pip install --python .venv/Scripts/python.exe --constraint configs/environment.constraints.txt --torch-backend cu128 --link-mode copy -e ".[probe,ml,dev]" --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Dependency dry-run failed' }
uv pip install --python .venv/Scripts/python.exe --constraint configs/environment.constraints.txt --torch-backend cu128 --link-mode copy "torch==2.11.0+cu128"
if ($LASTEXITCODE -ne 0) { throw 'CUDA torch install failed' }
uv pip install --python .venv/Scripts/python.exe --constraint configs/environment.constraints.txt --torch-backend cu128 --link-mode copy -e ".[probe,ml,dev]"
if ($LASTEXITCODE -ne 0) { throw 'Editable install failed' }
.venv/Scripts/python.exe -B -c "from r2ai import paths; import torch, sys; assert sys.version_info[:3] == (3,12,13); assert torch.__version__ == '2.11.0+cu128'; assert torch.version.cuda == '12.8'; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
