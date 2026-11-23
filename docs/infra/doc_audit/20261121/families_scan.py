# -*- coding: utf-8 -*-
"""版本家族 / 取代链识别与证据核验（只读）。

产物：
  families.csv                  家族 × 成员 × 角色 × 引用数 × 状态证据
  families_evidence_check.json  每条证据 needles 是否在源文件中命中及行号
  families_report.txt           人读摘要（FAMILIES.md 的数据底稿）

用法：python .slim/deepwork/20261121_docaudit/disposition/families_scan.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docscan_lib as L  # noqa: E402

R = L.ROOT

# 家族定义：FAMILIES.md 的机器可核验底稿。
# 每个成员：path + role（current/superseded/historical/partial/contradictory/unknown）
# 每条 evidence：(源文件, 必须命中的文本, 说明)。检不到即报警，防止"凭印象"。
FAMILIES = [
    {
        "id": "F01", "title": "单端点 IRC 生成控制器·产品设计",
        "note": "v1/v2 已迁入 docs/superseded/；v3 为当前产品设计",
        "members": [
            ("docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v1_20261020.md", "superseded"),
            ("docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v2_20261020.md", "superseded"),
            ("docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md", "current"),
        ],
        "evidence": [
            ("docs/superseded/README.md", "单端点 IRC 生成控制器 产品设计 v1", "superseded 索引表登记原路径与取代者"),
            ("docs/superseded/README.md", "产品设计 v3", "索引表取代者列"),
            ("docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md", "取代 v1/v2", "v3 状态头声明取代关系"),
            ("docs/README.md", "P3 建 `docs/superseded/` 并迁入 **4 份**", "README 2026-11-20 归档执行记录"),
        ],
        "verdict": "①可立即归位已完成（v1/v2 已在 superseded）；v3 留 plans/（Active-Proposed）",
    },
    {
        "id": "F02", "title": "V5.1-R3 端点对称特征与损失重训（R3 前身）",
        "note": "v1/v2 已被 R3 反应意图方案取代并迁入 superseded",
        "members": [
            ("docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v1_20261020.md", "superseded"),
            ("docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v2_20261020.md", "superseded"),
            ("docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md", "current"),
        ],
        "evidence": [
            ("docs/superseded/README.md", "V5.1-R3 端点对称特征与损失重训方案 v1", "superseded 索引表"),
            ("docs/superseded/README.md", "[R3 v3]", "取代者=R3 v3"),
            ("docs/README.md", "迁入 **4 份**自身已声明 Superseded 的文档", "归档执行记录"),
        ],
        "verdict": "①已归位；R3 v3 留 plans/（Active-Proposed）",
    },
    {
        "id": "F03", "title": "V5.1-R3 数据库+特征+损失三位一体修复（执行细化）",
        "note": "文件名 _v1 但状态仍 Active-Proposed，且 superseded/README 明确说明'不收录'",
        "members": [
            ("docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md", "current"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md", "Proposed v1.1", "文档状态头"),
            ("docs/README.md", "Active-Proposed（执行细化）", "README 状态列登记为执行细化"),
            ("docs/superseded/README.md", "仍在 `docs/README.md` 与台账中登记为 Proposed 执行细化", "明确未收录原因"),
        ],
        "verdict": "②留原地；文件名 _v1 与 v3 家族并列但并非被取代 → 需所有者确认是否并入 R3 v3 家族",
    },
    {
        "id": "F04", "title": "V5–V7 三阶段研究方案（v1.2 / v2.0 / v3.0 三代）",
        "note": "v2.0 被 v3.0 取代（两代都在 historical 门禁名单内，不得移动）",
        "members": [
            ("docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md", "superseded"),
            ("docs/plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md", "superseded"),
            ("docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md", "current"),
        ],
        "evidence": [
            ("docs/README.md", "Superseded（2026-10-20）", "README 对 v1.2 的状态列"),
            ("docs/README.md", "Superseded by v3.0", "README 对 v2.0 的状态列"),
            ("docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md", "状态：**Proposed v3.2**（取代 v2.0", "v3.0 头部状态行"),
            ("infra/governance/docs_consistency_config.json", "PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md", "historical 门禁名单：v2.0 移动即红"),
            ("infra/governance/docs_consistency_config.json", "PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md", "historical 门禁名单：v1.2 移动即红"),
        ],
        "verdict": "②仅加横幅/已在原地；①不可迁（门禁钉死）：v1.2、v2.0",
    },
    {
        "id": "F05", "title": "PES-N 未来开发计划（v2 → V5–V7 三代）",
        "note": "PES-N v2 状态 Superseded/Historical；继任链为 V5-V7 v1.2→v2.0→v3.0",
        "members": [
            ("docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md", "superseded"),
            ("docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md", "superseded"),
            ("docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md", "current"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md", "Superseded / Historical", "状态头"),
            ("docs/README.md", "历史（已被 V5–V7 v2.0 取代；内容 v2.5）", "README 权威文档表状态"),
            ("docs/README.md", "Superseded by v3.0", "v2.0 状态列"),
        ],
        "verdict": "②PES-N v2 正文有横幅但被 26 处引用、未在门禁名单 → 若迁 superseded 需先改 26 处链接，建议仅加横幅",
    },
    {
        "id": "F06", "title": "GeometryPolicyContract v1 / v2",
        "note": "v1 的 §4.2/§5.2 被 v2 取代（部分取代，v1 仍在 historical 门禁名单内）",
        "members": [
            ("docs/contracts/PES2TS_GeometryPolicyContract_v1.md", "partial"),
            ("docs/contracts/PES2TS_GeometryPolicyContract_v2.md", "current"),
        ],
        "evidence": [
            ("docs/contracts/PES2TS_GeometryPolicyContract_v1.md", "**SUPERSEDED（2026-10-21）**", "v1 头部横幅：§4.2/§5.2 被 v2 取代"),
            ("docs/README.md", "Superseded（§4.2/§5.2，由 [GeometryPolicyContract v2]", "README 状态列"),
        ],
        "verdict": "②部分取代：v1 留原地 + 横幅（门禁 historical 钉死，不得移动）；v2 为当前合同",
    },
    {
        "id": "F07", "title": "ReactionIntent Contract v1 / v2",
        "note": "v1 §4/§6.2 被 v2 取代（部分取代，v1 在 historical 门禁名单内）",
        "members": [
            ("docs/contracts/PES2TS_ReactionIntent_Contract_v1.md", "partial"),
            ("docs/contracts/PES2TS_ReactionIntent_Contract_v2.md", "current"),
        ],
        "evidence": [
            ("docs/contracts/PES2TS_ReactionIntent_Contract_v1.md", "**SUPERSEDED（2026-10-21）**", "v1 头部横幅：§4/§6.2 被 v2 取代"),
            ("docs/README.md", "Superseded（§4 Stop/q 条款由", "README 状态列"),
        ],
        "verdict": "②部分取代：v1 原地保留 + 横幅；v2 当前 Active",
    },
    {
        "id": "F08", "title": "FullPathNavigationContract v1 → MoveProcessContract v1（任务合同换代）",
        "note": "FullPathNavigation v1 被 ADR-0031/MoveProcessContract v1 整体取代",
        "members": [
            ("docs/contracts/PES2TS_FullPathNavigationContract_v1.md", "superseded"),
            ("docs/contracts/PES2TS_MoveProcessContract_v1.md", "current"),
        ],
        "evidence": [
            ("docs/contracts/PES2TS_FullPathNavigationContract_v1.md", "**SUPERSEDED（2026-10-21，ADR-0031 / MoveProcessContract v1）**", "v1 头部横幅"),
            ("docs/README.md", "Active-Normative（唯一 Move/Stop/TS/q 任务合同，ADR-0030）", "被取代时的旧状态"),
            ("docs/README.md", "SUPERSEDED by ADR-0031 / MoveProcessContract v1", "README 状态列"),
        ],
        "verdict": "②v1 门禁钉死（historical+required_files），仅加横幅；MoveProcessContract v1 当前权威",
    },
    {
        "id": "F09", "title": "V4.2–V4.7 十小时战役（方案 + 合同 + 报告）",
        "note": "三件套同属 Executed-Frozen 负结果证据，报告只读",
        "members": [
            ("docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md", "historical"),
            ("docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md", "historical"),
            ("docs/reports/PES2TS_实现报告_20261018.md", "historical"),
            ("docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md", "historical"),
        ],
        "evidence": [
            ("docs/README.md", "Executed-Frozen", "README 状态列（方案/合同）"),
            ("docs/README.md", "V4.7 负结果与工程修复记录；只读", "实现报告状态列"),
            ("docs/README.md", "23 个正式 run；10-18 判定负结果，保留证据", "方案状态列"),
        ],
        "verdict": "②全部原地保留（报告只读 + 冻结证据）；不迁 superseded",
    },
    {
        "id": "F10", "title": "V5.4.1 Embedding-50K（方案 + ADR + 报告 + registry）",
        "note": "Executed-Frozen；E*=E0 留任；方案文件名 v1.0 但内容已 v1.2",
        "members": [
            ("docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md", "current"),
            ("docs/design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md", "current"),
            ("docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md", "current"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md", "状态：**Executed-Frozen**（2026-10-29", "方案状态头"),
            ("docs/README.md", "文件名保留 v1.0", "README 说明文件名与内容版本不一致（内容 v1.1.1）"),
            ("docs/README.md", "转 Executed-Frozen", "README 状态同步记录"),
        ],
        "verdict": "②原地保留（门禁 required_files）；文件名 v1.0 vs 内容 v1.2 → 命名异常 P2",
    },
    {
        "id": "F11", "title": "V5.4.2 Backbone-50K（方案 + ADR + 报告）",
        "note": "WP7 已 EXECUTED（2026-10-31），但方案状态头仍 Active-Proposed → 状态滞后",
        "members": [
            ("docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md", "contradictory"),
            ("docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md", "current"),
            ("docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md", "current"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md", "状态：**Active-Proposed**", "方案状态头（未随执行更新）"),
            ("docs/README.md", "V5.4.2 WP7 EXECUTED", "README 2026-10-31 执行记录"),
            ("docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md", "no_backbone_winner", "执行报告结论"),
        ],
        "verdict": "③需所有者裁决：状态自相矛盾（计划 Active-Proposed vs 报告已 EXECUTED）",
    },
    {
        "id": "F12", "title": "V5.4.3 双 Decoder-50K（方案+清单+ADR+报告）",
        "note": "18-run 已执行完毕（2026-11-06），文档保持 Frozen-Preregistered/Proposed 不再改写（台账明示）",
        "members": [
            ("docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md", "current"),
            ("docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md", "current"),
            ("docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md", "current"),
            ("docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md", "current"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md", "Frozen-Preregistered v1.0", "方案状态头"),
            ("docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md", "状态：**Proposed**", "ADR 状态头"),
            ("docs/plans/PES2TS_文档修订台账.md", "V5.4.3 全阶段执行完毕（Executed-Frozen；双轨冠军确证）", "台账 2026-11-06 登记"),
            ("docs/plans/PES2TS_文档修订台账.md", "保持 `Frozen-Preregistered v1.0`", "台账明示不改文"),
        ],
        "verdict": "②原地保留（不改冻结原文是明示规则）；docs/README 缺 2026-11-06 条目 → 索引滞后 P1",
    },
    {
        "id": "F13", "title": "V5.4.3 之前的 V5.4 剩余架构工作方案",
        "note": "v1.0，头部写'建议状态'，0 引用，未在台账/README 登记",
        "members": [
            ("docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md", "unknown"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md", "建议状态：**Active-Proposed**", "'建议状态'非正式状态声明"),
        ],
        "verdict": "③需所有者裁决：0 引用 + 非正式状态",
    },
    {
        "id": "F14", "title": "V5.3-MP 大规模 MOVE–PROCESS（计划 + S0–S4 报告）",
        "note": "S0–S3/S4 已执行；计划状态头仍 Proposed v1.1，仅以'执行状态'行旁注",
        "members": [
            ("docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md", "contradictory"),
            ("docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md", "historical"),
            ("docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md", "historical"),
            ("docs/reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md", "historical"),
            ("docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md", "historical"),
            ("docs/reports/PES2TS_V53MP_S4_50k_单路正式训练报告_20261027.md", "historical"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md", "状态：**Proposed v1.1**", "计划状态头未更新"),
            ("docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md", "执行状态（2026-10-23）：S0/S1/S2/S3 训练全部完成", "头部执行状态旁注"),
            ("docs/README.md", "V5.3-MP S1 2k 冒烟完成（当前）", "README 阶段条目"),
        ],
        "verdict": "③需所有者裁决：状态头 Proposed vs 已执行；报告只读不动",
    },
    {
        "id": "F15", "title": "30h R3→V5.2→V5.3 分层训练与验收计划",
        "note": "同一文件内 L3 EXECUTED-FROZEN/Stop 已取代 与 L5 '待冻结的执行计划' 自相矛盾",
        "members": [
            ("docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md", "contradictory"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md", "EXECUTED-FROZEN / Stop 语义已取代（2026-10-21）", "头部横幅"),
            ("docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md", "状态：**待冻结的执行计划**", "同一文件 L5 的另一状态声明"),
            ("docs/README.md", "Executed-Frozen / Stop 语义已被 ADR-0030 取代", "README 状态列"),
        ],
        "verdict": "③需所有者裁决：同文件两处状态声明矛盾（且门禁 historical 钉死不可移动）",
    },
    {
        "id": "F16", "title": "V5.0–V5.2 十小时冲刺（方案 + 执行报告）",
        "note": "V5.1a 已执行且 rejected_gate；方案 Executed-Frozen/Historical",
        "members": [
            ("docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md", "historical"),
            ("docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md", "historical"),
        ],
        "evidence": [
            ("docs/README.md", "Executed-Frozen（V5.1a）/ Historical", "README 状态列"),
            ("docs/README.md", "V5.1a 已执行并 `rejected_gate`", "README 2026-10-18 条"),
        ],
        "verdict": "②原地保留（门禁 historical 钉死；报告只读）",
    },
    {
        "id": "F17", "title": "V5.1 修复冲刺与扩量准入 / V5.1-R2 损失尺度修复",
        "note": "R2 被 R3 取代（门禁钉死）；V5.1 修复冲刺 v1 状态缺失、仅 1 引用",
        "members": [
            ("docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md", "unknown"),
            ("docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md", "superseded"),
        ],
        "evidence": [
            ("docs/README.md", "Superseded by R3", "README 对 R2 的状态列"),
            ("infra/governance/docs_consistency_config.json", "PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md", "historical 门禁名单"),
        ],
        "verdict": "②R2 原地+横幅（门禁钉死）；③V5.1 修复冲刺需裁决（无状态头、1 引用）",
    },
    {
        "id": "F18", "title": "V52R 全路径导航与端点停止修复（方案 + 输入充分性）",
        "note": "D2-G STOP 关闭线；后续由 V5.3-MP 接续",
        "members": [
            ("docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md", "historical"),
            ("docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md", "historical"),
            ("docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md", "historical"),
        ],
        "evidence": [
            ("docs/README.md", "Closed — D2-G STOP", "README 状态列"),
            ("AGENT.md", "V5.2 / V5.2-R 已关闭（D2-G STOP，T1 不授权，负结果归档）", "AGENT.md 执行投影中的关闭记录"),
        ],
        "verdict": "②原地保留（负结果证据；报告只读）",
    },
    {
        "id": "F19", "title": "EGNN→BC→IQL→SAC 训练方案三代（v2 / v3 / IRC 直接位移 v1）",
        "note": "三份均自声明被取代，但均不在门禁名单、有 2–7 处引用 → 迁 superseded 需先改链",
        "members": [
            ("docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md", "superseded"),
            ("docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md", "superseded"),
            ("docs/plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md", "superseded"),
        ],
        "evidence": [
            ("docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md", "已被取代", "头部横幅"),
            ("docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md", "已被取代", "头部横幅"),
            ("docs/plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md", "已被取代", "头部横幅"),
        ],
        "verdict": "③需所有者裁决：可迁 superseded（自声明）但有引用；建议先加'已被取代'指针再迁",
    },
    {
        "id": "F20", "title": "训练/测试集划分与数据来源方案（同文件名 v1→v1.1）",
        "note": "文件名 v1_20261120，内容已到 v1.1；docs/README 2026-11-20 条登记",
        "members": [
            ("docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md", "current"),
        ],
        "evidence": [
            ("docs/README.md", "v1.1（2026-11-20 晚）", "README 条目记录内容升 v1.1"),
            ("docs/plans/PES2TS_文档修订台账.md", "升 **v1.1**", "台账 2026-11-20 增量登记"),
        ],
        "verdict": "②原地保留；文件名 v1 vs 内容 v1.1 → 命名异常 P2",
    },
    {
        "id": "F21", "title": "仓库归档与清理方案 v1（治理/操作规则）",
        "note": "2026-11-20 归档治理的来源文档，operational 非规范",
        "members": [
            ("docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md", "current"),
        ],
        "evidence": [
            ("docs/README.md", "PES2TS 仓库归档与清理方案 v1", "README 顶部条目"),
            ("docs/plans/PES2TS_文档修订台账.md", "新增 [PES2TS 仓库归档与清理方案 v1]", "台账 2026-11-20 登记"),
        ],
        "verdict": "②原地保留（docs/infra 操作规则来源）",
    },
]

VERSION_STRIP_RE = re.compile(r"_v\d+(?:\.\d+)*")
DATE_STRIP_RE = re.compile(r"_20\d{6}")


def check_evidence() -> list[dict]:
    results = []
    for fam in FAMILIES:
        for src, needle, note in fam["evidence"]:
            path = R / src
            rec = {"family": fam["id"], "source": src, "needle": needle, "note": note,
                   "exists": path.is_file(), "line": None, "ok": False}
            if path.is_file():
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    lines = []
                for i, ln in enumerate(lines, 1):
                    if needle in ln:
                        rec.update(ok=True, line=i)
                        break
                if not rec["ok"]:
                    # 兜底：全文字符串包含（跨行 needle 情况）
                    rec["ok"] = needle in "\n".join(lines)
            results.append(rec)
    return results


def family_stats(docs: dict[str, L.Doc]) -> list[dict]:
    rows = []
    for fam in FAMILIES:
        for path, role in fam["members"]:
            d = docs.get(path)
            rows.append({
                "family": fam["id"], "family_title": fam["title"], "path": path, "role": role,
                "exists": bool(d),
                "bucket": d.bucket if d else "",
                "action": d.action if d else "",
                "inbound_refs": d.inbound_refs if d else "",
                "gate_locked": ";".join(d.gate_locked) if d else "",
                "head_status": ";".join(d.status) if d else "",
                "self_declared_superseded": int(d.self_declared_superseded) if d else "",
                "verdict": fam["verdict"],
            })
    return rows


def autodetect_uncovered(docs: list[L.Doc]) -> list[dict]:
    """发现未被 FAMILIES 覆盖的多版本文件名族（同一 basename 去版本后的组）。"""
    covered = {p for fam in FAMILIES for p, _ in fam["members"]}
    groups: dict[str, list[str]] = {}
    for d in docs:
        if not d.path.startswith("docs/"):
            continue
        name = Path(d.path).name
        base = DATE_STRIP_RE.sub("", VERSION_STRIP_RE.sub("", Path(d.path).stem))
        if base != Path(d.path).stem:
            groups.setdefault(base, []).append(d.path)
    out = []
    for base, members in groups.items():
        if len(members) > 1 and not all(m in covered for m in members):
            out.append({"base": base, "members": sorted(members),
                        "covered": [m for m in members if m in covered]})
    return sorted(out, key=lambda x: -len(x["members"]))


def main() -> None:
    docs = L.build()
    by_path = {d.path: d for d in docs}
    ev = check_evidence()
    rows = family_stats(by_path)
    uncovered = autodetect_uncovered(docs)

    (L.OUT_DIR / "families.csv").write_text("", encoding="utf-8")  # 占位，下行重写
    with (L.OUT_DIR / "families.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = [e for e in ev if not e["ok"]]
    payload = {
        "families": len(FAMILIES),
        "members": len(rows),
        "evidence_total": len(ev),
        "evidence_failed": bad,
        "uncovered_multi_version_groups": uncovered,
        "per_family": [
            {"id": f["id"], "title": f["title"], "verdict": f["verdict"],
             "members": len(f["members"]),
             "evidence": [e for e in ev if e["family"] == f["id"]]}
            for f in FAMILIES
        ],
    }
    (L.OUT_DIR / "families_evidence_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"families={len(FAMILIES)} members={len(rows)} evidence={len(ev)} failed={len(bad)}"]
    for f in FAMILIES:
        lines.append(f"{f['id']} {f['title']} | members={len(f['members'])} | {f['verdict']}")
    lines.append(f"uncovered multi-version groups: {len(uncovered)}")
    for u in uncovered[:20]:
        lines.append(f"  - {u['base']}: {u['members']}")
    (L.OUT_DIR / "families_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    if bad:
        print("EVIDENCE MISSING:")
        for b in bad:
            print("  ", b)
    print("uncovered groups:", len(uncovered))


if __name__ == "__main__":
    main()
