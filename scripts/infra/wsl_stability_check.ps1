<#
.SYNOPSIS
    PES2TS WSL 稳定运行 —— Windows 宿主侧 preflight / 告警检查（PowerShell 5.1 兼容）。

.DESCRIPTION
    在启动任何 WSL 重活（大数据扫描、数据审计、V5.6 训练）之前运行一次；也可由计划任务
    每 5 分钟运行作为巡检。检查项：
      1. Windows 提交内存（commit）余量 / 使用率
      2. 页面文件分配与峰值
      3. vmmemWSL 工作集（WSL VM 占用的宿主内存）
      4. C: / E: 磁盘余量（页面文件与 ext4.vhdx 所在盘）
      5. ext4.vhdx 体积
      6. 最近 N 分钟内的 Resource-Exhaustion-Detector 2004 低虚拟内存事件
      7. WSL 发行版当前状态

    退出码：0=OK，1=WARN，2=CRITICAL（可阻止重活启动）。

.PARAMETER Heavy
    重活模式：使用更严格阈值（commit 余量、vmmemWSL、2004 事件），适用于数据扫描与训练。

.PARAMETER Json
    以 JSON 输出（便于 CI / 其他脚本消费）。

.PARAMETER AutoShutdown
    CRITICAL 且 WSL 正在运行时执行 `wsl --shutdown`，释放 vmmem 后再退出（默认只建议不执行）。

.PARAMETER EventWindowMinutes
    回看低内存事件的窗口分钟数，默认 15。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\infra\wsl_stability_check.ps1 -Heavy

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\infra\wsl_stability_check.ps1 -Json -EventWindowMinutes 30
#>
[CmdletBinding()]
param(
    [switch]$Heavy,
    [switch]$Json,
    [switch]$AutoShutdown,
    [int]$EventWindowMinutes = 15
)

$ErrorActionPreference = "Continue"
$now = Get-Date
$issues = New-Object System.Collections.ArrayList
$result = [ordered]@{
    checked_at = $now.ToString("s")
    mode = $(if ($Heavy) { "heavy" } else { "normal" })
    level = "OK"
    exit_code = 0
    metrics = [ordered]@{}
    issues = @()
    vmmem_gb = $null
    wsl_state = "unknown"
}

function Add-Issue([string]$level, [string]$code, [string]$message) {
    [void]$issues.Add([ordered]@{ level = $level; code = $code; message = $message })
}

# ---------- 1. OS 内存 / 提交内存 ----------
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
if ($null -ne $os) {
    $result.metrics.phys_total_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $result.metrics.phys_free_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
}
# 提交内存以 PerfOS_Memory 为准（CommittedBytes / CommitLimit 是事件 2004 判定的真实口径）；
# Win32_OperatingSystem 的 TotalVirtualMemorySize/FreeVirtualMemory 仅作兜底。
$memPerf = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
$commitLimitGB = 0.0
$commitFreeGB = 0.0
$commitUsedPct = 0.0
if ($null -ne $memPerf -and $memPerf.CommitLimit -gt 0) {
    $commitLimitGB = [math]::Round($memPerf.CommitLimit / 1GB, 1)
    $committedGB = [math]::Round($memPerf.CommittedBytes / 1GB, 1)
    $commitFreeGB = [math]::Round(($memPerf.CommitLimit - $memPerf.CommittedBytes) / 1GB, 1)
    $commitUsedPct = [math]::Round($memPerf.CommittedBytes / $memPerf.CommitLimit * 100, 1)
    $result.metrics.commit_source = "PerfOS_Memory"
    $result.metrics.committed_gb = $committedGB
    $result.metrics.available_gb = [math]::Round($memPerf.AvailableBytes / 1GB, 1)
} elseif ($null -ne $os) {
    $commitLimitGB = [math]::Round($os.TotalVirtualMemorySize / 1MB, 1)
    $commitFreeGB = [math]::Round($os.FreeVirtualMemory / 1MB, 1)
    if ($os.TotalVirtualMemorySize -gt 0) {
        $commitUsedPct = [math]::Round((1 - $os.FreeVirtualMemory / $os.TotalVirtualMemorySize) * 100, 1)
    }
    $result.metrics.commit_source = "Win32_OperatingSystem"
}
$result.metrics.commit_limit_gb = $commitLimitGB
$result.metrics.commit_free_gb = $commitFreeGB
$result.metrics.commit_used_pct = $commitUsedPct

# ---------- 2. 页面文件 ----------
$pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
if ($null -ne $pf) {
    $result.metrics.pagefiles = @()
    foreach ($p in @($pf)) {
        $result.metrics.pagefiles += [ordered]@{
            name = $p.Name
            allocated_gb = [math]::Round($p.AllocatedBaseSize / 1024, 1)
            current_usage_gb = [math]::Round($p.CurrentUsage / 1024, 1)
            peak_usage_gb = [math]::Round($p.PeakUsage / 1024, 1)
        }
    }
    $pfTotalGB = 0.0
    foreach ($p in @($pf)) { $pfTotalGB += ($p.AllocatedBaseSize / 1024) }
    $result.metrics.pagefile_total_gb = [math]::Round($pfTotalGB, 1)
}

# ---------- 3. vmmemWSL ----------
$vmmem = Get-Process vmmemWSL -ErrorAction SilentlyContinue
if ($null -ne $vmmem) {
    $vmmemGB = [math]::Round(($vmmem | Measure-Object WorkingSet64 -Sum).Sum / 1GB, 1)
    $result.vmmem_gb = $vmmemGB
} else {
    $result.vmmem_gb = 0.0
}

# ---------- 4. 磁盘 ----------
$disks = @{}
foreach ($d in (Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" -ErrorAction SilentlyContinue)) {
    $disks[$d.DeviceID] = [math]::Round($d.FreeSpace / 1GB, 1)
}
$result.metrics.disk_free_gb = $disks

# ---------- 5. ext4.vhdx ----------
$vhdxPath = $null
$lxss = Get-ChildItem "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss" -ErrorAction SilentlyContinue
foreach ($k in @($lxss)) {
    $p = Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue
    if ($null -ne $p -and $null -ne $p.BasePath) {
        $bp = $p.BasePath -replace '^\\\\\?\\', ''
        $candidate = Join-Path $bp "ext4.vhdx"
        if (Test-Path $candidate) { $vhdxPath = $candidate; break }
    }
}
if ($null -ne $vhdxPath) {
    $vhdx = Get-Item $vhdxPath -ErrorAction SilentlyContinue
    if ($null -ne $vhdx) {
        $result.metrics.vhdx_path = $vhdxPath
        $result.metrics.vhdx_size_gb = [math]::Round($vhdx.Length / 1GB, 1)
    }
}

# ---------- 6. WSL 状态 ----------
# 不用 `wsl -l -v`（PS 5.1 下其输出为 UTF-16 且带 NUL，难以稳定解析），
# 以 vmmemWSL 进程是否存在判断 WSL VM 是否在运行。
$wslState = "Stopped"
if ($null -ne $vmmem) { $wslState = "Running" }
$result.wsl_state = $wslState

# ---------- 7. 低内存事件（Event 2004） ----------
$eventCount = 0
$eventLastVmmemGB = $null
$eventLastTime = $null
$start = $now.AddMinutes(-1 * $EventWindowMinutes)
try {
    $events = Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 2004; StartTime = $start } -ErrorAction SilentlyContinue
    if ($null -ne $events) {
        $events = @($events)
        $eventCount = $events.Count
        $last = $events | Sort-Object TimeCreated -Descending | Select-Object -First 1
        if ($null -ne $last) {
            $eventLastTime = $last.TimeCreated.ToString("s")
            if ($last.Message -match 'vmmemWSL\s*\(\d+\)\s*consumed\s*(\d+)\s*bytes') {
                $eventLastVmmemGB = [math]::Round([int64]$Matches[1] / 1GB, 1)
            }
        }
    }
} catch {
    Add-Issue "WARN" "EVENT_QUERY_FAIL" ("无法读取事件日志: " + $_.Exception.Message)
}
$result.metrics.event_2004_count = $eventCount
$result.metrics.event_2004_last_time = $eventLastTime
$result.metrics.event_2004_last_vmmem_gb = $eventLastVmmemGB
$result.metrics.event_window_minutes = $EventWindowMinutes

# ---------- 阈值判定 ----------
$commitFree = $result.metrics.commit_free_gb
$commitPct = $result.metrics.commit_used_pct
$vmmemGB = $result.vmmem_gb
$cFree = $null; $eFree = $null
if ($disks.ContainsKey("C:")) { $cFree = $disks["C:"] }
if ($disks.ContainsKey("E:")) { $eFree = $disks["E:"] }
$vhdxGB = $result.metrics.vhdx_size_gb

if ($Heavy) {
    $commitFreeWarn = 20.0; $commitFreeCrit = 12.0
    $vmmemWarn = 14.0; $vmmemCrit = 20.0
    $pctWarn = 80.0; $pctCrit = 90.0
} else {
    $commitFreeWarn = 15.0; $commitFreeCrit = 8.0
    $vmmemWarn = 20.0; $vmmemCrit = 26.0
    $pctWarn = 85.0; $pctCrit = 92.0
}

if ($null -ne $commitFree) {
    if ($commitFree -lt $commitFreeCrit) { Add-Issue "CRITICAL" "COMMIT_FREE_CRIT" ("commit 余量 {0} GB < {1} GB" -f $commitFree, $commitFreeCrit) }
    elseif ($commitFree -lt $commitFreeWarn) { Add-Issue "WARN" "COMMIT_FREE_WARN" ("commit 余量 {0} GB < {1} GB" -f $commitFree, $commitFreeWarn) }
}
if ($null -ne $commitPct) {
    if ($commitPct -ge $pctCrit) { Add-Issue "CRITICAL" "COMMIT_PCT_CRIT" ("commit 使用率 {0}% >= {1}%" -f $commitPct, $pctCrit) }
    elseif ($commitPct -ge $pctWarn) { Add-Issue "WARN" "COMMIT_PCT_WARN" ("commit 使用率 {0}% >= {1}%" -f $commitPct, $pctWarn) }
}
if ($null -ne $vmmemGB) {
    if ($vmmemGB -ge $vmmemCrit) { Add-Issue "CRITICAL" "VMMEM_CRIT" ("vmmemWSL {0} GB >= {1} GB" -f $vmmemGB, $vmmemCrit) }
    elseif ($vmmemGB -ge $vmmemWarn) { Add-Issue "WARN" "VMMEM_WARN" ("vmmemWSL {0} GB >= {1} GB" -f $vmmemGB, $vmmemWarn) }
}
if ($null -ne $cFree) {
    if ($cFree -lt 10.0) { Add-Issue "CRITICAL" "C_FREE_CRIT" ("C: 剩余 {0} GB < 10 GB（页面文件无法扩张）" -f $cFree) }
    elseif ($cFree -lt 20.0) { Add-Issue "WARN" "C_FREE_WARN" ("C: 剩余 {0} GB < 20 GB（页面文件扩张受限）" -f $cFree) }
}
if ($null -ne $eFree) {
    if ($eFree -lt 25.0) { Add-Issue "CRITICAL" "E_FREE_CRIT" ("E: 剩余 {0} GB < 25 GB（ext4.vhdx 生长空间）" -f $eFree) }
    elseif ($eFree -lt 60.0) { Add-Issue "WARN" "E_FREE_WARN" ("E: 剩余 {0} GB < 60 GB（ext4.vhdx 生长空间）" -f $eFree) }
}
if ($eventCount -gt 0) {
    $msg = ("最近 {0} 分钟出现 {1} 次低虚拟内存事件 2004（最近 vmmemWSL={2} GB）" -f $EventWindowMinutes, $eventCount, $eventLastVmmemGB)
    if ($Heavy -or $eventCount -ge 2) { Add-Issue "CRITICAL" "EVENT_2004" $msg }
    else { Add-Issue "WARN" "EVENT_2004" $msg }
}

# ---------- 汇总 ----------
$level = "OK"
$exitCode = 0
foreach ($i in $issues) {
    if ($i.level -eq "CRITICAL") { $level = "CRITICAL"; $exitCode = 2; break }
    elseif ($i.level -eq "WARN" -and $level -ne "CRITICAL") { $level = "WARN"; $exitCode = 1 }
}
$result.level = $level
$result.exit_code = $exitCode
$result.issues = @($issues)

# ---------- 告警行为 ----------
if ($level -eq "CRITICAL" -and $AutoShutdown -and $wslState -eq "Running") {
    Write-Warning "[guard] CRITICAL: 正在执行 wsl --shutdown 释放 vmmemWSL ..."
    wsl --shutdown 2>$null
    Start-Sleep -Seconds 3
    $result.metrics.auto_shutdown_performed = $true
}

# ---------- 输出 ----------
if ($Json) {
    $result | ConvertTo-Json -Depth 6
} else {
    Write-Host ""
    Write-Host ("=== PES2TS WSL 稳定运行检查 ({0}) mode={1} ===" -f $result.checked_at, $result.mode) -ForegroundColor Cyan
    Write-Host ("物理内存 : {0} GB 总 / {1} GB 空闲" -f $result.metrics.phys_total_gb, $result.metrics.phys_free_gb)
    Write-Host ("提交内存 : 限额 {0} GB / 空闲 {1} GB / 使用 {2}%" -f $result.metrics.commit_limit_gb, $result.metrics.commit_free_gb, $result.metrics.commit_used_pct)
    if ($result.metrics.Contains("pagefile_total_gb")) {
        Write-Host ("页面文件 : 共 {0} GB" -f $result.metrics.pagefile_total_gb)
    }
    Write-Host ("vmmemWSL : {0} GB  (WSL 状态: {1})" -f $result.vmmem_gb, $result.wsl_state)
    $diskParts = @()
    foreach ($k in $disks.Keys) { $diskParts += ("{0} {1} GB" -f $k, $disks[$k]) }
    Write-Host ("磁盘余量 : {0}" -f ($diskParts -join " / "))
    if ($result.metrics.Contains("vhdx_size_gb")) {
        Write-Host ("ext4.vhdx: {0} GB  ({1})" -f $result.metrics.vhdx_size_gb, $result.metrics.vhdx_path)
    }
    Write-Host ("2004 事件: 最近 {0} 分钟 {1} 次" -f $EventWindowMinutes, $eventCount)
    Write-Host ""
    if ($issues.Count -eq 0) {
        Write-Host ("[OK] 可以启动 WSL 重活。" -f $level) -ForegroundColor Green
    } else {
        Write-Host ("[判定] {0}（exit={1}）" -f $level, $exitCode) -ForegroundColor $(if ($level -eq "CRITICAL") { "Red" } else { "Yellow" })
        foreach ($i in $issues) {
            $color = if ($i.level -eq "CRITICAL") { "Red" } else { "Yellow" }
            Write-Host ("  [{0}] {1}: {2}" -f $i.level, $i.code, $i.message) -ForegroundColor $color
        }
        Write-Host ""
        Write-Host "处置建议："
        if ($level -eq "CRITICAL") {
            Write-Host "  1) 立即停止新的 WSL 重活；执行: wsl --shutdown"
            Write-Host "  2) 检查页面文件（建议移到 D: 固定 24-32 GB）与 C: 空间"
            Write-Host "  3) 释放后再运行本脚本，直到判定 OK 再继续"
        } else {
            Write-Host "  1) 避免再启动新的重活；优先完成收敛/落盘"
            Write-Host "  2) 阶段结束后执行 wsl --shutdown 释放 vmmemWSL"
        }
    }
    Write-Host ""
}

exit $exitCode
