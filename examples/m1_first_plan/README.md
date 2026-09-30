# M1 首个可执行计划

该计划由 scan-ready 的 `RXN_0000007104` 端点数据生成，保留 `truth_assisted_p1` 来源标记，并从白名单转换器排除旧导出中的 IRC 字段。端点 multiplicity 在此示例中显式输入为 1，正式计算前仍须化学复核。计划以 H–N 距离为单驱动坐标，包含 9 个扫描目标值；N–H 断键及 C=N 键级变化均作为观察量保留，可转换为 ACP 静态请求。它尚未提交 ACP、未执行 xTB，也不代表有效反应路径或 TS。

`ACP_job_create_preflight.json` 是可审阅的 ACP V1 `PESsearch` job-create 请求体，资源草案为 2 核、4 GB；任务预算仍在 `remark` 中保留，ACP 端尚无对应的 CPU-hour/wall-time 强制限额。请求没有指定输出目录，由 ACP 管理 `WORK/RESULT`。该文件没有提交。
