#!/bin/bash
# PES2TS WSL guest 侧内存回收：sync + drop_caches，并打印回收前后内存。
#
# 用法（Windows 侧）：
#   wsl -e bash /mnt/e/Calculations/AI4S_ML_Studys/PES2TS/infra/runtime/wsl_reclaim.sh
#
# 说明：
#   - 该脚本只回收 guest 页缓存/可回收 slab；匿名内存需进程退出才能归还。
#   - 彻底把 vmmemWSL 占用的宿主内存还给 Windows，仍建议在阶段结束后执行
#     `wsl --shutdown`（见 docs/infra/PES2TS_WSL稳定运行与内存治理方案_v1_20261122.md）。
#   - 需要 root（WSL 默认登录用户即 root）。
set -u

GREEN=""; NC=""
if [ -t 1 ]; then GREEN="\033[32m"; NC="\033[0m"; fi

mem_line() {
  free -m | awk '/^Mem:/{printf "used=%sMB free=%sMB buff/cache=%sMB available=%sMB\n", $3, $4, $6, $7}'
}

echo "== PES2TS WSL reclaim $(date '+%Y-%m-%d %H:%M:%S %z') =="
echo "before: $(mem_line)"

if [ "$(id -u)" -ne 0 ]; then
  echo "WARN: 非 root，无法写 /proc/sys/vm/drop_caches；仅执行 sync。" >&2
fi

sync
if [ -w /proc/sys/vm/drop_caches ]; then
  echo 3 > /proc/sys/vm/drop_caches && echo "drop_caches: 1+2+3 已执行"
else
  echo "WARN: /proc/sys/vm/drop_caches 不可写，跳过 drop_caches" >&2
fi

echo "after : $(mem_line)"
echo ""
echo "${GREEN}提示${NC}: 若要立即把内存还给 Windows，请在 Windows 侧执行: wsl --shutdown"
echo "      然后用 infra/runtime/wsl_stability_check.ps1 复检 vmmemWSL。"
