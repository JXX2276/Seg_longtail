param(
    [string]$Config = "configs/teacher_500.yaml"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$env:HF_HOME = Join-Path $Root ".cache/huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $Root ".cache/huggingface/hub"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
$env:YOLO_OFFLINE = "true"

function Invoke-Stage {
    param([string[]]$Arguments)
    & .\.venv\Scripts\seg-longtail.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Teacher stage failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-Stage @("convert", "--config", $Config)
Invoke-Stage @(
    "propose", "--config", $Config,
    "--input", "workspace/teacher/index.jsonl",
    "--output", "workspace/teacher/candidates.jsonl",
    "--device", "cuda:0"
)
Invoke-Stage @(
    "verify", "--config", $Config,
    "--input", "workspace/teacher/candidates.jsonl",
    "--output", "workspace/teacher/verified.jsonl",
    "--device", "cuda:0"
)
Invoke-Stage @(
    "segment", "--config", $Config,
    "--input", "workspace/teacher/verified.jsonl",
    "--output", "workspace/teacher/segmented.jsonl",
    "--device", "cuda:0"
)
Invoke-Stage @(
    "export", "--config", $Config,
    "--input", "workspace/teacher/segmented.jsonl",
    "--output-dir", "workspace/teacher/generated_masks"
)

& .\.venv\Scripts\python.exe scripts\build_training_dataset.py
if ($LASTEXITCODE -ne 0) { throw "build_training_dataset.py failed" }

Write-Host "Teacher pipeline complete: workspace/dataset"
