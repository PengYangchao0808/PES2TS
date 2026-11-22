#!/bin/bash
# PES2TS WSL 受限运行器（稳定性方案 v1 的便捷封装）
#
# 在 cgroup-v2 scope 里运行命令，给单次重活设硬内存上限，防止一次数据扫描把
# vmmemWSL 顶到 25-30 GB、进而耗尽宿主 commit（2026-11-22 两次 renderer OOM 的根因）。
#
# 默认委托给既有、带 2s 内存采样与 gate 的 scripts/infra/cgroup_scope.sh
# （S4 infra plan §5.1 产物）；指定 --cpu 时退化为直接 systemd-run（无采样）。
#
# 用法（Windows 侧）：
#   wsl -e bash /mnt/e/.../scripts/infra/wsl_guarded_run.sh --mem 12G \
#       --name v55-audit -- /root/miniforge3/envs/pes2ts-gpu/bin/python scripts/v55/audit_data_assets.py
#
# 参数：
#   --mem   <SIZE>  单任务内存硬上限（MemoryMax），默认 12G
#   --high  <SIZE>  内存高水位（MemoryHigh），默认 = 0.9*--mem
#   --cpu   <PCT>   CPU 配额（如 800）。指定时直接 systemd-run（无 cgroup 采样 CSV）
#   --outdir <DIR>  cgroup 采样输出目录；默认 outputs/ops/wsl_stability/cgroup/<name>-<ts>
#   --name  <NAME>  任务名（用于 unit 名与 outdir），默认 pes2ts-task
#   --             参数结束符，其后为要执行的命令
#
# 退出码：透传被运行命令的退出码；137 = 在 scope 内被 cgroup OOM 终止；
#        91/90 = cgroup_scope.sh 的 gate（oom_kill>0 / swap 长期>0）。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCOPE_SH="$SCRIPT_DIR/cgroup_scope.sh"

MEM="12G"
HIGH=""
CPU=""
OUTDIR=""
NAME="pes2ts-task"
while [ $# -gt 0 ]; do
  case "$1" in
    --mem)    MEM="${2:?--mem 需要值}"; shift 2 ;;
    --high)   HIGH="${2:?--high 需要值}"; shift 2 ;;
    --cpu)    CPU="${2:?--cpu 需要值}"; shift 2 ;;
    --outdir) OUTDIR="${2:?--outdir 需要值}"; shift 2 ;;
    --name)   NAME="${2:?--name 需要值}"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "用法: $0 [--mem 12G] [--high 10G] [--cpu 800] [--outdir DIR] [--name TAG] -- <command...>" >&2
  exit 64
fi

if [ -z "$HIGH" ]; then
  case "$MEM" in
    *G|*g) HIGH="$(awk -v m="${MEM%[Gg]}" 'BEGIN{printf "%dG", (m*0.9)+0.5}')" ;;
    *M|*m) HIGH="$(awk -v m="${MEM%[Mm]}" 'BEGIN{printf "%dM", (m*0.9)+0.5}')" ;;
    *)     HIGH="$MEM" ;;
  esac
fi
if [ -z "$OUTDIR" ]; then
  OUTDIR="$ROOT_DIR/outputs/ops/wsl_stability/cgroup/${NAME}-$(date +%Y%m%dT%H%M%S)"
fi

mem_line() {
  free -m | awk '/^Mem:/{printf "used=%sMB buff/cache=%sMB available=%sMB", $3, $6, $7}'
}

echo "[guard] $(date '+%Y-%m-%d %H:%M:%S') task=$NAME mem_before: $(mem_line)"
echo "[guard] MemoryMax=$MEM MemoryHigh=$HIGH CPUQuota=${CPU:-<none>} outdir=$OUTDIR"
echo "[guard] cmd: $*"

RC=0
if [ -z "$CPU" ] && [ -x "$SCOPE_SH" ]; then
  # 复用 S4 infra 的 cgroup_scope.sh：硬上限 + 2s 采样 + oom/swap gate + 汇总 JSON
  CGROUP_SCOPE_UNIT="pes2ts-${NAME}-$$" "$SCOPE_SH" "$MEM" "$HIGH" "$OUTDIR" "$@"
  RC=$?
elif command -v systemd-run >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  PROPS=(--property="MemoryMax=$MEM" --property="MemorySwapMax=0" --property="MemoryHigh=$HIGH")
  if [ -n "$CPU" ]; then PROPS+=(--property="CPUQuota=${CPU}%"); fi
  echo "[guard] 直接 systemd-run（无 cgroup 采样；如需采样请去掉 --cpu 改用 cgroup_scope.sh）"
  systemd-run --scope --quiet --unit="pes2ts-${NAME}-$(date +%s)" "${PROPS[@]}" -- "$@"
  RC=$?
else
  echo "[guard] WARN: systemd-run 不可用，退化为 ulimit -v 兜底" >&2
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
  echo "[guard] 任务被 cgroup OOM 终止（exit=137）：请缩小数据分片/批大小或提高 --mem 后重跑。" >&2
elif [ "$RC" -eq 91 ] || [ "$RC" -eq 90 ]; then
  echo "[guard] cgroup gate 未通过（exit=$RC，见 $OUTDIR/cgroup_summary.json）。" >&2
fi
exit "$RC"
