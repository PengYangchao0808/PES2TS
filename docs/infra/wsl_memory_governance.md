# WSL 内存治理运维手册（S4 infra，plan §5.1 / §5.4 / §5.5 / §10）

> 状态：随 S4 infra 工具链落地（`scripts/infra/`）。
> 依据：`outputs/pesn/v53mp/reports/S4_infra_优化方案_20261023.md` §5.1、§5.4、§5.5、§7、§10。
> 适用机器：WSL2（内核 5.15，**无 PSI**：`/proc/pressure/memory` 不存在；**无 cgroup v2 `memory.peak`**）。

---

## 1. 当前 `.wslconfig` vs 推荐值

| 项 | 当前 | 推荐（`config/wslconfig.recommended.ini`） | 理由 |
|---|---|---|---|
| `memory` | **56GB** | **40GB** | 56 GB 把宿主机挤干 → 两次 host OOM（2026-10-22）。40 GB 给 Windows 留 ~24 GB（宿主 64 GB 假设；宿主更小按 70% 折算） |
| `swap` | 0 | **8GB** | 只作安全气垫，不是容量。swap 被训练真实使用 = 失败（见 §3 swap 判定门） |
| `swapFile` | `E:\wsl_swap\swap.vhdx` | 不变 | 放快速本地盘 |
| `autoMemoryReclaim` | `gradual` | 不变（`[experimental]` 段） | 逐步归还内存给宿主 |

**应用方式**：把 `config/wslconfig.recommended.ini` 内容拷到 Windows 侧 `%UserProfile%\.wslconfig`，然后在 PowerShell 执行 `wsl --shutdown` 重启 VM。不重启不生效。

**为什么 56 GB 会打死宿主**：2026-10-22 两次 VM 死机时，guest cgroup 限 50 GiB、`oom_kill=0`、guest anon 37.0 GiB——即 **宿主机先耗尽**，任何 guest 内部的 cgroup 上限都无法预防。因此第一层治理必须在 Windows 侧（`.wslconfig memory=40GB`），第二层才是 guest 内的 per-run scope。

---

## 2. 两层内存治理

```
Windows 64 GB
└─ WSL VM  ≤40 GB（.wslconfig，第一层）
   └─ per-run transient scope（systemd-run --scope，第二层）
      ├─ 训练期（S3 现状）：MemoryMax=34G / MemoryHigh=32G —— 单路
      ├─ I1 之后（并发）：MemoryMax=6G / MemoryHigh 按 run —— 一个 run 回归只杀自己
      └─ 采样：cgroup_scope.sh 每 2 s 落 memory.current / swap.current / anon / file / oom*
```

- **单路训练（现状）**：`MemoryMax=34G / MemoryHigh=32G`。20k 单 run 稳态 anon 25.9–26.0 GB，终评瞬时峰值 30.75 GiB，都在 34 G 之下。
- **I1 之后每 run 一个 `MemoryMax=6G` scope**：loader v2（memmap 列式索引）落地后单 run anon 目标 ≤2.5 GB（50k）/ ≤4 GB（100k），6 G 上限足够；某一路 OOM 只会 OOM-kill 自己的 scope，不拖垮 VM，更不拖垮其他 run。

### 2.1 用 `cgroup_scope.sh` 启动一次受控 run

```bash
bash scripts/infra/cgroup_scope.sh 34G 32G outputs/pesn/v53mp/infra_bench/probe \
    python3 -c "import time; time.sleep(20)"
```

- 首选 `systemd-run --scope -p MemoryMax=... -p MemoryHigh=...`（与 `scripts/campaigns/r3_v53mp_cgroup_run.sh` 同机制）；systemd 不可用时回退到手动 cgroup v2 目录（`CGROUP_V2_PARENT`，默认 `/sys/fs/cgroup`）。
- **采样器每 2 s 写 `<outdir>/cgroup_sample.csv`（与 `r3_v53mp_cgroup_run.sh` 同格式，前 4 列逐字节一致：`ts,memory_current_bytes,anon_bytes,file_bytes`；第 5 列 `memory_swap_current_bytes` 是 §5.1 swap 门的附加列，旧 4 列 CSV 仍可解析）+ `.peak`（ts,peak,anon）+ `.events`（memory.events 快照）**。
- 退出时打印 summary（peak anon、max swap.current、swap 持续秒数、oom_kill）并写 `<outdir>/cgroup_summary.json`。
- **退出码**：命令本身的退出码（137 = scope 内 OOM kill）；命令成功但 swap 持续 >60 s → **90**；命令成功但 `oom_kill>0` → **91**（plan §5.1 swap 判定门）。systemd 不可用时用 `--fallback` 显式切换手动 cgroup-v2 目录（`CGROUP_V2_PARENT`，默认 `/sys/fs/cgroup`）。
- 可调环境变量：`CGROUP_SCOPE_INTERVAL`（采样周期，默认 2 s）、`CGROUP_SCOPE_UNIT`。systemd 路径固定 `-p MemorySwapMax=0`（与 r3 脚本一致）——scope 内结构性禁止换页，溢出表现为 in-scope oom_kill，`靠 swap 跑完` 不可能发生。

### 2.2 监控（本机无 PSI，用这三样 + Windows 侧）

guest 内每 2 s 采（`cgroup_scope.sh` 已自动做）：

| 文件 | 内容 |
|---|---|
| `/sys/fs/cgroup/.../memory.current` | 当前占用 |
| `memory.stat` 的 `anon` / `file` | 匿名页（真正的工作集）vs 页缓存（可回收）——判"内存高"到底是数据还是缓存 |
| `memory.events` 的 `oom` / `oom_kill` | OOM 计数（内核 5.15 无 `memory.peak`，所以峰值必须自己采样） |

Windows 侧（guest 看不到宿主余量，需另采）：`wsl --status`，或 PowerShell `Get-Counter '\Memory\Available MBytes'`，与 guest 采样同一时间轴对齐——两次 host OOM 的教训是**只盯 guest 指标必然漏掉宿主悬崖**。

---

## 3. Swap 判定门（硬门，plan §5.1）

> `memory.swap.current` 在训练期**持续 >0 = 失败**。不接受"靠 swap 跑完"。

- `.wslconfig swap=8GB` 只是气垫：允许偶发的、瞬时的换页不杀死 run；但 `cgroup_scope.sh` 会统计 swap>0 的**持续秒数**，>60 s 直接退出码 90。
- 战役级 run（`r3_v53mp_cgroup_run.sh`）另设 `MemorySwapMax=0`——训练 scope 内根本不许换页，溢出即 in-scope OOM kill，比静默降速到不可用要好。
- 判定到 swap 持续 >0 时：先查是不是 `.wslconfig` 没重启生效（`wsl --status`），再查是不是并发路数超了（每路 6 G × 路数 + 缓存 > VM 上限）。

---

## 4. 数据搬 ext4（plan §5.4；`migrate_data_to_ext4.py`）

50k/100k 表合计 80/160 GB，`/mnt/e` 只剩 ~71 GB 且是 drvfs（9p）：npz 随机读 5.62 ms、大文件 mmap 不可靠。搬到 ext4（本机 `/` 有 ~824 GB）后 npz 2.06 ms、manifest 1.5 min → 20 s。

四阶段（每阶段一个显式 flag；默认是 dry-run plan）：

```bash
# 1) plan（只读）：文件数、总字节、src/dst 文件系统类型、1.2× 空间检查、目标提议
python3 -m scripts.infra.migrate_data_to_ext4

# 2) copy（可断点续跑：dst 已存在且 sidecar sha 相同的文件直接跳过）
python3 -m scripts.infra.migrate_data_to_ext4 --copy --target /pes2ts_data/v53mp

# 3) verify：对每个 dst 文件强制全量 rehash，与 src sidecar sha 对比
python3 -m scripts.infra.migrate_data_to_ext4 --verify --target /pes2ts_data/v53mp

# 4) commit：manifest 双写（repo reports/ + dst 根），随后 src 树逐文件换成 symlink
python3 -m scripts.infra.migrate_data_to_ext4 --commit --target /pes2ts_data/v53mp
```

守门（任何字节移动前 fail-closed）：

- dst 必须是 **ext4**（`statfs` f_type == 0xEF53；`/mnt/e` 是 9p → 拒绝）；
- dst 剩余空间 ≥ **1.2 × 树大小**；
- `--copy/--verify/--commit` 必须显式 `--target`（detector 只在 plan 里提议）；
- `--commit` 在同一次调用里没有通过的 `--verify` 会自己先跑一遍 verify——**symlink 永远不先于校验**。

commit 产物：

- `outputs/pesn/v53mp/reports/data_migration_manifest_<date>.json`（repo 侧）+ `<dst>/data_migration_manifest_<date>.json`（dst 侧），逐文件 `{path, sha256, size}`；
- `outputs/pesn/v53mp/data/<file>` 变为指向 ext4 副本的 symlink（repo 内留 manifest + symlink，plan §5.4 原文要求）。

回滚：删 symlink、把 dst 文件拷回原位即可——manifest 里有每个文件的 sha256，可以逐文件验证回拷。

---

## 5. Sidecar 与线程卫生（plan §5.5）

- **sha256 sidecar**：`<file>.sha256sidecar.json`，键为 `(path, size, mtime_ns, inode)`；任何一项变化即静默失效并重算；`--reverify` 强制全量：
  ```bash
  python3 -m scripts.infra.sha_sidecar verify outputs/pesn/v53mp/data/*.npz
  python3 -m scripts.infra.sha_sidecar verify --reverify outputs/pesn/v53mp/data/*.csv
  ```
  一份 JSON 对应一个目标文件，mmap 安全、并行安全（无共享索引可写坏）。
- **线程卫生 A/B**（预期 5–15%，**必须实测**，不得当作已知收益）：
  ```bash
  bash scripts/infra/thread_caps.sh python3 <cmd>     # OMP/MKL/OPENBLAS/NUMEXPR=1
  ```
  对应 B3 台阶：`infra_bench --thread-caps`。

---

## 6. B0 画像 / 基准（plan §7、§10-B0）

```bash
# B0：当前 HEAD，3 reps，独立 bench root（不写 campaign ledger/不消耗 attempt cap）
python3 -m scripts.infra.infra_bench --tag B0 \
    --run-plan outputs/pesn/v53mp/run_plan_v53mp_20k.jsonl --run-id v53mp_20k_A_s17

# 无 GPU 环境/CI：走 dry-run（伪造 stub 计时，验证 bench-root/解析/报告/判定链路）
python3 -m scripts.infra.infra_bench --tag B0 --dry-run --reps 2
```

**bench-root 机制**（不改 `r3_v53mp_run.py`）：每个 rep 一个自洽根
`outputs/pesn/v53mp/infra_bench/<tag>_<ts>/rep<k>/root/`，内含从冻结计划
拷出的 `run_plan_v53mp_20k.jsonl`（A_s17 行 + `n_updates=--updates`、
`smoke_updates=null`，五 sha 逐字原样）+ 按 driver plan-init 同一 scheme 重写的
`.pin.json`（`{sha256, campaign_mode}`）。config/tables/split/r3-ckpt 全部指向
**原始路径**（yaml 不拷贝不改——`config_sha256` 绑定必须继续对原文件校验）。
启动前两道验收：`r3_v53mp_run.py --output-root <bench_root> gate-check`
子进程 + 进程内跑 driver 自己的 `read_run_plan` / `verify_plan_pin` /
`assert_stage_binding` / five-sha 复算。随后每个 rep 经
`r3_v53mp_run.py --output-root <rep_root> run` 执行（包在 `cgroup_scope.sh`
与 nvidia-smi 采样器里）——plan pin、五 sha、PHASE_CAPS 全部按 rep 根生效，
campaign 侧 `run_records.jsonl` 永远不被触碰。

两处 driver 契约偏差（写进 `report.json`，不隐藏）：

- `_parse_run_id` 只认 `v53mp_<tier>_<arm>_s<seed>`，所以 bench 行沿用
  `v53mp_20k_A_s17`（隔离靠 rep 根，不靠改 run_id；`bench_` 前缀会被
  driver 拒收）；
- tier 20k 的 driver-effective 训练长度由冻结 config schedule 决定
  （freeze 400 + joint 15000 = **15400 updates**），计划行 `n_updates`
  只进 ledger/打印，不进 trainer。

**派生 bench config（闭合 15400-vs-2000 缺口）**：`--updates` 与
driver-effective 不同时，infra_bench 按 bench 派生一份
`config_bench.yaml`（写入 `<rep_root>/`）：载入**原始** yaml（永不写回），
整份复制 `training_contract.schedule`，仅改写
`joint_updates_base_2k = (updates − freeze) / factor(tier)`
（factor = TIER_LIMITS[tier]//2000，20k 即 10；freeze 保持 400；λ、seeds、
val_every、batch、lr 等全部逐字保留）。拒绝条件（fail-closed）：
`--updates ≤ freeze`（`UPDATES_LE_FREEZE`）、`updates − freeze` 不能被
factor 整除（`UPDATES_NOT_DIVISIBLE`）。bench 计划行的 `config_sha256`
随即绑定**派生 yaml 的 sha**（五 sha 绑定全部内化在 bench root 内：pin
盖的是含该行的 plan 字节，`validate_bench_root`/driver 的 five-sha 复算
都对派生文件校验），driver 也以 `--config <bench yaml>` 启动。
`--updates == driver-effective` 时直接用原始 config（无警告路径）。
派生文件带 `infra_bench:` provenance 段（记录 derived_from / 输入 sha）。
`report.json` 记录 `{config_source: "bench_derived" | "original",
original_config_sha256, bench_config_sha256, driver_effective_updates}`
；`driver_effective_updates` 始终按**实际传给 driver 的 config** 重算。

产物：`outputs/pesn/v53mp/infra_bench/<tag>_<ts>/report.{json,md}`，含 per-rep
与 median/P95 ms/update、anon 峰值、VRAM 峰值、启动秒（spawn →
train_log 首行）、swap 峰值、oom_kill、gate_check 结果，以及机器可读 verdict
（B0 门：median ms/update ∈ 78±15%、anon ∈ 26 GB±15%、swap=0、oom_kill=0；
warmup 排除只作用于 ms/update 统计，不影响启动测量）。

B1–B4 每级只改一个变量（plan §7 阶梯），**一次只上一级**；B1/B2 的验收是
"数据与指标逐位不变"，B3/B4 才是"变快"。

---

## 7. 故障速查

| 症状 | 处理 |
|---|---|
| VM 整个重启、guest 无 oom_kill | 宿主 OOM：查 Windows 侧可用内存采样；确认 `.wslconfig memory≤40GB` 且 `wsl --shutdown` 过 |
| scope 内 oom_kill>0（退出码 91 或 137） | 单 run 超 MemoryMax：先确认是否 I1 后还在用 34G 旧值；再查该 tier 的 anon 预算（plan §2 表） |
| 退出码 90 | swap 持续 >60 s：见 §3 |
| 搬迁被拒 NOT_EXT4 | 目标不是 ext4（drvfs/9p/tmpfs 都不行）——换挂载点 |
| 搬迁被拒 INSUFFICIENT_FREE_SPACE | dst 剩余 < 1.2× 树大小；清理或换盘 |
| sidecar verify 全 FAIL | 文件确实变了（mtime/inode 漂移）：先 `git status`/manifest 对照，确认无损后再 `update` 重建 sidecar |
