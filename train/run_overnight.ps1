# Overnight queue for the controlled retraining experiment.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File train\run_overnight.ps1
# Safe to re-run: every run resumes from its last completed epoch; finished runs are skipped.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
New-Item -ItemType Directory -Force logs | Out-Null

# keep the machine awake while this script runs
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]0x80000001L -bor [uint32]0x00000040L) | Out-Null

$queue = @(
  @{arm="purged";         seed=0},
  @{arm="random_control"; seed=0},
  @{arm="purged";         seed=1},
  @{arm="random_control"; seed=1},
  @{arm="purged";         seed=2},
  @{arm="random_control"; seed=2}
)
foreach ($r in $queue) {
  $tag = "$($r.arm)_seed$($r.seed)"
  if (Test-Path "data\retrain\${tag}_test.parquet") { Write-Host "skip $tag (done)"; continue }
  Write-Host "=== $tag  $(Get-Date)"
  python train\train_leakage_experiment.py --arm $r.arm --seed $r.seed *>> "logs\$tag.log"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR in $tag (exit code $LASTEXITCODE) - stopping queue"
    break
  }
  python analysis\06_evaluate_retraining.py *>> "logs\evaluate.log"
}
Write-Host "queue finished $(Get-Date)"
