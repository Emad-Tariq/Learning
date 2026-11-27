param(
    [string]$DataPath = (Join-Path $env:USERPROFILE 'Downloads\reid_student_data_main'),
    [int]$Workers = 8,
    [switch]$Regenerate
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location -LiteralPath $projectPath
$runPath = 'Models\Abhay\Model11\local_runs\pc_retrain_20261126'
$requiredFiles = @('baseline_models.joblib','verifier.joblib','calibrators.joblib','metrics.json')
$missing = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $runPath $_)) })
if (-not $Regenerate -and $missing.Count -eq 0) {
    $csvFiles = @('submission_mlp_blend.csv','submission_baseline.csv')
    $outputsExist = @($csvFiles | Where-Object { Test-Path -LiteralPath (Join-Path $runPath $_) }).Count -eq 2
    if ($outputsExist) {
        $latestModel = ($requiredFiles | ForEach-Object { (Get-Item -LiteralPath (Join-Path $runPath $_)).LastWriteTimeUtc } | Sort-Object -Descending | Select-Object -First 1)
        $oldestCsv = ($csvFiles | ForEach-Object { (Get-Item -LiteralPath (Join-Path $runPath $_)).LastWriteTimeUtc } | Sort-Object | Select-Object -First 1)
        if ($oldestCsv -ge $latestModel) {
            Write-Host "Already generated: $projectPath\$runPath\submission_mlp_blend.csv"
            Write-Host 'Use -Regenerate to rerun inference. No work is needed to reuse the existing CSV.'
            return
        }
    }
}
if ($missing.Count -gt 0) {
    & python -u Models\Abhay\Model11\run.py --data $DataPath --output $runPath --workers 4
    if ($LASTEXITCODE -ne 0) { throw 'Training stopped. Completed checkpoints and matching cache are retained.' }
} else {
    Write-Host 'Found complete trained checkpoints. Skipping retraining and validation.'
}
& python -u Models\Abhay\Model11\predict_submission.py --data $DataPath --sample (Join-Path $DataPath 'sample_submission.csv') --run $runPath --workers $Workers --descriptor-cache 384
if ($LASTEXITCODE -ne 0) { throw 'Prediction stopped. Matching cache is retained for the next run.' }
Write-Host "Submission: $projectPath\$runPath\submission_mlp_blend.csv"
