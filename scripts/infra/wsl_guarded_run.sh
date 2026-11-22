#!/bin/bash
# PES2TS WSL 受限运行器：在 systemd cgroup 里运行命令，给单次重活设内存/CPU 上限。
#
# 目的：防止“一次数据扫描把 vmmemWSL 顶到 25-30 GB → 宿主提交内存耗尽 →
# DSH/其他进程 OOM”的 2026-11-22 事故重演。内存超限时由 cgroup 杀掉本任务
# （退出码 137），而不是拖垮整个 Windows 宿主。
#
# 用法（Windows 侧）：
#   wsl -e bash /mnt/e/.../scripts/infra/wsl_guarded_run.sh --mem 12G --cpu 800 -- \
#       /root/miniforge3/envs/pes2ts-gpu/bin/python scripts/v55/audit_data_assets.py
#
# 参数：
#   --mem   <SIZE>  单任务内存硬上限，例如 12G / 8192M（默认 12G；MemoryMax）
#   --high  <SIZE>  内存高水位（默认 = 0.9*--mem，达到后内核开始回收而不直接杀）
#   --cpu   <PCT>   CPU 配额（如 800 = 8 核；不填则不限制）
#   --name  <NAME>  systemd 单元名后缀（便于在 systemctl 里定位）
#   --       参数结束符，其后是要执行的命令
#
# 退出码：透传被运行命令的退出码；137 = 被 cgroup OOM 终止（应缩小数据分片重跑）。
set -u

MEM="12G"
HIGH=""
CPU=""
NAME="pes2ts-task"
while [ $# -gt 0 ]; do
  case "$1" in
    --mem)  MEM="${2:?--mem 需要值}"; shift 2 ;;
    --high) HIGH="${2:?--high 需要值}"; shift 2 ;;
    --cpu)  CPU="${2:?--cpu 需要值}"; shift 2 ;;
    --name) NAME="${2:?--name 需要值}"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "用法: $0 [--mem 12G] [--high 10G] [--cpu 800] [--name TAG] -- <command...>" >&2
  exit 64
fi

# 默认 MemoryHigh = 0.9 * MemoryMax
if [ -z "$HIGH" ]; then
  case "$MEM" in
    *G|*g) HIGH="$(awk -v m="${MEM%[Gg]}" 'BEGIN{printf "%dG", (m*0.9)+0.5}')" ;;
    *M|*m) HIGH="$(awk -v m="${MEM%[Mm]}" 'BEGIN{printf "%dM", (m*0.9)+0.5}')" ;;
    *)     HIGH="$MEM" ;;
  esac
fi

mem_line() {
  free -m | awk '/^Mem:/{printf "used=%sMB buff/cache=%sMB available=%sMB", $3, $6, $7}'
}

echo "[guard] $(date '+%Y-%m-%d %H:%M:%S') mem_before: $(mem_line)"
echo "[guard] limits: MemoryMax=$MEM MemorySwapMax=0 MemoryHigh=$HIGH CPUQuota=${CPU:-<none>}"
echo "[guard] task=$NAME :: $*"

RC=0
if command -v systemd-run >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  PROPS=(--property="MemoryMax=$MEM" --property="MemorySwapMax=0" --property="MemoryHigh=$HIGH")
  if [ -n "$CPU" ]; then PROPS+=(--property="CPUQuota=${CPU}%"); fi
  systemd-run --scope --quiet --unit="pes2ts-${NAME}-$(date +%s)" "${PROPS[@]}" -- "$@"
  RC=$?
else
  echo "[guard] WARN: systemd-run 不可用（systemd 未启用？），退化为 ulimit -v 兜底" >&2
  case "$MEM" in
    *G|*g) VKB=$(awk -v m="${MEM%[Gg]}" 'BEGIN{printf "%d", m*1024*1024}') ;;
    *M|*m) VKB=$(awk -v m="${MEM%[Mm]}" 'BEGIN{printf "%d", m*1024}') ;;
    *)     VKB="" ;;
  esac
  if [ -n "$VKB" ] && ulimit -v "$VKB" 2>/dev/null; then
    echo "[guard] fallback ulimit -v ${VKB} KB"
  fi
  "$@"
  RC=$?
fi

echo "[guard] $(date '+%Y-%m-%d %H:%M:%S') mem_after : $(mem_line)  exit=$RC"
if [ "$RC" -eq 137 ]; then
  echo "[guard] 任务被 cgroup OOM 终止（exit=137）：请缩小 --mem 对应的数据分片/批大小后重跑。" >&2
fi
exit "$RC"
