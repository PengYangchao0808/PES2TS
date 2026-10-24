param(
    [Parameter(Mandatory=$true)][string]$OutCsv,
    [string]$Sentinel = "",
    [int]$IntervalSeconds = 2,
    [int]$DurationSeconds = 1800,
    [double]$MinHeadroomGB = 8.0
)
$ErrorActionPreference = "Stop"
$outDir = Split-Path -Parent $OutCsv
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
$summaryPath = [System.IO.Path]::ChangeExtension($OutCsv, ".summary.json")
"ts,committed_gb,limit_gb,headroom_gb,min_headroom_gb,below_gate" |
    Set-Content -Encoding UTF8 $OutCsv

$start = Get-Date
$minHeadroom = [double]::PositiveInfinity
$samples = 0
$belowGate = 0
$stopReason = "duration"
while ($true) {
    if ((Get-Date) - $start -ge [TimeSpan]::FromSeconds($DurationSeconds)) {
        $stopReason = "duration"
        break
    }
    if ($Sentinel -and -not (Test-Path -LiteralPath $Sentinel)) {
        $stopReason = "sentinel-removed"
        break
    }
    try {
        $samplesNow = (Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' -ErrorAction Stop).CounterSamples
        $committed = [double]$samplesNow[0].CookedValue
        $limit = [double]$samplesNow[1].CookedValue
        $headroom = $limit - $committed
    } catch {
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    $headroomGB = $headroom / 1GB
    if ($headroomGB -lt $minHeadroom) { $minHeadroom = $headroomGB }
    $isBelow = $headroomGB -lt $MinHeadroomGB
    if ($isBelow) { $belowGate++ }
    "{0},{1:N3},{2:N3},{3:N3},{4:N3},{5}" -f `
        (Get-Date).ToString("o"), ($committed / 1GB), ($limit / 1GB), `
        $headroomGB, $minHeadroom, $isBelow |
        Add-Content -Encoding UTF8 $OutCsv
    $samples++
    Start-Sleep -Seconds $IntervalSeconds
}
$summary = [ordered]@{
    out_csv = $OutCsv
    sentinel = $Sentinel
    started_at = $start.ToString("o")
    ended_at = (Get-Date).ToString("o")
    stop_reason = $stopReason
    samples = $samples
    duration_seconds = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    min_headroom_gb = if ($samples -gt 0) { [math]::Round($minHeadroom, 3) } else { $null }
    min_headroom_gate_gb = $MinHeadroomGB
    below_gate_samples = $belowGate
    passed = ($samples -gt 0 -and $belowGate -eq 0)
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $summaryPath
$summary | ConvertTo-Json -Depth 4
if ($summary.passed) { exit 0 } else { exit 2 }
