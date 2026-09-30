import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repo = process.cwd();
const threadId = "01a0ed78-73eb-7833-b3c1-4a6c0733a397";
const outputDir = path.join(repo, "outputs", threadId);
const workbookPath = path.join(outputDir, "Demo24_双人化学复核.xlsx");
const imageDir = path.join(outputDir, "reaction_images");
const previewDir = "C:/Users/彭扬超/.codex/visualizations/2026/09/29/01a0ed78-73eb-7833-b3c1-4a6c0733a397";
const imageManifest = JSON.parse(await fs.readFile(path.join(imageDir, "manifest.json"), "utf8"));
if (imageManifest.length !== 24 || imageManifest.some((item) => !item.mapping_exact)) {
  throw new Error("Expected 24 RDKit cards with exact R/P atom-map sets; render the reaction cards first.");
}
console.log(`RDKit cards ready: ${imageManifest.length}; exact map sets: ${imageManifest.filter((item) => item.mapping_exact).length}`);

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') quoted = false;
      else value += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") {
      row.push(value);
      value = "";
    } else if (char === "\n") {
      row.push(value.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      value = "";
    } else value += char;
  }
  if (value.length || row.length) {
    row.push(value.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows.shift();
  return rows.filter((r) => r.length === headers.length).map((r) =>
    Object.fromEntries(headers.map((header, i) => [header, r[i]])));
}

const csv = await fs.readFile(path.join(repo, "data/manifests/pes2ts_demo24_candidates_v1.csv"), "utf8");
const candidates = parseCsv(csv);
const manifest = JSON.parse(await fs.readFile(path.join(repo, "data/manifests/demo24_reaction_case_manifest_v1.json"), "utf8"));
if (candidates.length !== 24 || manifest.n_cases !== 24) throw new Error("Expected exactly 24 reviewed cases");
const stratumLabels = { A: "A · H transfer", B: "B · single bond", C: "C · substitution", D: "D · 2-bond", E: "E · 3-bond", F: "F · H₂", G: "G · aromatic", H: "H · fallback" };
const displayStratum = (value) => stratumLabels[value[0]] ?? value;
const caseByReaction = new Map(manifest.records.map((record) => [record.reaction_id, record]));
const infoRows = candidates.map((c) => {
  const record = caseByReaction.get(c.reaction_id);
  if (!record || record.split !== c.split) throw new Error(`Case manifest mismatch: ${c.reaction_id}`);
  return [
    c.reaction_id, c.split, displayStratum(c.stratum), Number(c.n_atoms), Number(c.F), Number(c.B), Number(c.O),
    Number(c.n_h_transfer), Number(c.n_h_hh_events), Number(c.n_aromatic_edits),
    Number(c.R_components), Number(c.P_components), c.mapping_status,
    Number(c.radical_electrons_R), Number(c.radical_electrons_P),
    Number(c.charged_atoms_R), Number(c.charged_atoms_P), Number(c.min_endpoint_distance_A),
    c.family_labels, c.reaction_smiles, record.case_id, record.status, record.relative_path,
  ];
});
const reviewHeaders = [
  "反应ID", "split", "机制层", "反应结构与映射", "复核者", "结论",
  "反应中心/机制", "原子映射", "电荷与自旋", "几何/装配",
  "扫描可行性", "R multiplicity", "P multiplicity", "备注",
];
const blankReviewRow = (c) => [c.reaction_id, c.split, displayStratum(c.stratum), "见结构图", "", "pending",
  "not_reviewed", "not_reviewed", "not_reviewed", "not_reviewed", "pending", null, null, ""];
const reviewRows1 = candidates.map(blankReviewRow);
const reviewRows2 = candidates.map(blankReviewRow);
const adjudicationRows = candidates.map((c) => [c.reaction_id, c.split, displayStratum(c.stratum),
  "pending", "pending", "pending", "", null, null, "", ""]);

const workbook = Workbook.create();
const review1 = workbook.worksheets.add("复核者1");
const review2 = workbook.worksheets.add("复核者2");
const adjudication = workbook.worksheets.add("汇总裁定");
const details = workbook.worksheets.add("候选摘要");
const mapped = workbook.worksheets.add("映射SMILES");
const instructions = workbook.worksheets.add("复核说明");

async function setupReviewSheet(sheet, title, rows, tableName) {
  sheet.showGridLines = false;
  sheet.tabColor = "#D7A94B";
  sheet.getRange("A1:N29").format.font = { name: "Arial", size: 10, color: "#263746" };
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#173B57" };
  sheet.getRange("A3").values = [["先看 D 列反应图理解 R→P 的映射变化；黄色单元格为复核输入。两位复核者应独立完成，未填字段不代表通过。"]];
  sheet.getRange("A3:N3").format.font = { name: "Arial", size: 10, italic: true, color: "#5C6D78" };
  sheet.getRange("A5:N5").values = [reviewHeaders];
  sheet.getRange("A6:N29").values = rows;
  sheet.getRange("A5:N5").format = { fill: "#214B65", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
  sheet.getRange("A5:N29").format.verticalAlignment = "center";
  sheet.getRange("E6:E29").format.fill = "#FFF2CC";
  sheet.getRange("F6:M29").format.fill = "#FFF2CC";
  sheet.getRange("N6:N29").format.fill = "#FFF2CC";
  sheet.getRange("N6:N29").format.wrapText = true;
  sheet.getRange("A5:N29").format.borders = { insideHorizontal: { style: "thin", color: "#E4EAF0" }, bottom: { style: "thin", color: "#D5DEE5" } };
  sheet.tables.add("A5:N29", true, tableName);
  sheet.dataValidations.add({ range: "F6:F29", rule: { type: "list", values: ["pending", "accept", "reject", "replace", "needs_more_info"] } });
  for (const col of ["G", "H", "I", "J"]) {
    sheet.dataValidations.add({ range: `${col}6:${col}29`, rule: { type: "list", values: ["not_reviewed", "confirmed", "issue"] } });
  }
  sheet.dataValidations.add({ range: "K6:K29", rule: { type: "list", values: ["pending", "1D", "synchronized", "path_or_neb", "reject"] } });
  for (const col of ["L", "M"]) {
    sheet.dataValidations.add({ range: `${col}6:${col}29`, rule: { type: "whole", operator: "between", formula1: 1, formula2: 10 } });
  }
  const widths = { A: 20, B: 10, C: 18, D: 78, E: 16, F: 18, G: 23, H: 17, I: 19, J: 22, K: 19, L: 17, M: 17, N: 38 };
  for (const [col, width] of Object.entries(widths)) sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  sheet.getRange("A5:N5").format.rowHeight = 34;
  sheet.getRange("A6:N29").format.rowHeight = 270;
  for (let i = 0; i < candidates.length; i += 1) {
    const bytes = await fs.readFile(path.join(imageDir, `${candidates[i].reaction_id}.png`));
    sheet.images.add({ dataUrl: `data:image/png;base64,${bytes.toString("base64")}`,
      alt: `${candidates[i].reaction_id} mapped reactant and product structures`,
      anchor: { from: { row: 5 + i, col: 3 }, extent: { widthPx: 585, heightPx: 340 } } });
  }
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(3);
}

await setupReviewSheet(review1, "复核者 1｜端点化学独立复核", reviewRows1, "ReviewerOneTable");
await setupReviewSheet(review2, "复核者 2｜端点化学独立复核", reviewRows2, "ReviewerTwoTable");

adjudication.showGridLines = false;
adjudication.tabColor = "#426B83";
adjudication.getRange("A1:K29").format.font = { name: "Arial", size: 10, color: "#263746" };
adjudication.getRange("A2").values = [["双人结果对照与最终裁定"]];
adjudication.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#173B57" };
adjudication.getRange("A3").values = [["仅在两份独立复核完成后填写；存在分歧时先解决，不要把待议案例标为 accept。"]];
adjudication.getRange("A3:K3").format.font = { name: "Arial", size: 10, italic: true, color: "#5C6D78" };
adjudication.getRange("A5:K5").values = [["反应ID", "split", "机制层", "复核者1结论", "复核者2结论", "最终裁定", "裁定者", "R multiplicity", "P multiplicity", "分歧处理说明", "case_id"]];
adjudication.getRange("A6:K29").values = adjudicationRows.map((row, i) => [...row.slice(0, 10), caseByReaction.get(candidates[i].reaction_id).case_id]);
adjudication.getRange("A5:K5").format = { fill: "#214B65", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
adjudication.getRange("A5:K29").format.verticalAlignment = "center";
adjudication.getRange("F6:J29").format.fill = "#FFF2CC";
adjudication.getRange("A5:K29").format.borders = { insideHorizontal: { style: "thin", color: "#E4EAF0" }, bottom: { style: "thin", color: "#D5DEE5" } };
adjudication.tables.add("A5:K29", true, "AdjudicationTable");
for (const col of ["D", "E", "F"]) adjudication.dataValidations.add({ range: `${col}6:${col}29`, rule: { type: "list", values: ["pending", "accept", "reject", "replace", "needs_more_info"] } });
for (const col of ["H", "I"]) adjudication.dataValidations.add({ range: `${col}6:${col}29`, rule: { type: "whole", operator: "between", formula1: 1, formula2: 10 } });
for (const [col, width] of Object.entries({ A: 20, B: 10, C: 18, D: 20, E: 20, F: 18, G: 18, H: 20, I: 20, J: 38, K: 25 })) adjudication.getRange(`${col}:${col}`).format.columnWidth = width;
adjudication.getRange("A5:K5").format.rowHeight = 34;
adjudication.getRange("A6:K29").format.rowHeight = 28;
adjudication.freezePanes.freezeRows(5);
adjudication.freezePanes.freezeColumns(3);

details.showGridLines = false;
details.tabColor = "#8299A7";
const detailHeaders = ["反应ID", "split", "机制层", "原子数", "成键", "断键", "键级变化", "H转移", "H-H事件", "芳香编辑", "R组分", "P组分", "映射状态", "自由基e-R", "自由基e-P", "带电原子-R", "带电原子-P", "最小端点距离(Å)", "反应类型", "案例状态"];
const summaryRows = infoRows.map((row) => [...row.slice(0, 19), row[21]]);
details.getRange("A1:T29").format.font = { name: "Arial", size: 10, color: "#263746" };
details.getRange("A2").values = [["候选端点与来源信息"]];
details.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#173B57" };
details.getRange("A3").values = [["源自冻结候选 CSV 与 ReactionCase 清单；ReactionCase 状态均为 needs_review，多重度尚未确认。mapped SMILES 在单独工作表中查看。"]];
details.getRange("A3:T3").format.font = { name: "Arial", size: 10, italic: true, color: "#5C6D78" };
details.getRange("A5:T5").values = [detailHeaders];
details.getRange("A6:T29").values = summaryRows;
details.getRange("A5:T5").format = { fill: "#214B65", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
details.getRange("A5:T29").format.verticalAlignment = "center";
details.getRange("A5:T29").format.borders = { insideHorizontal: { style: "thin", color: "#E4EAF0" }, bottom: { style: "thin", color: "#D5DEE5" } };
details.tables.add("A5:T29", true, "CandidateSummaryTable");
const detailWidths = { A: 20, B: 10, C: 25, D: 10, E: 12, F: 12, G: 12, H: 12, I: 12, J: 13, K: 12, L: 12, M: 28, N: 12, O: 12, P: 16, Q: 16, R: 23, S: 32, T: 17 };
for (const [col, width] of Object.entries(detailWidths)) details.getRange(`${col}:${col}`).format.columnWidth = width;
details.getRange("A5:T5").format.rowHeight = 38;
details.getRange("A6:T29").format.rowHeight = 24;
details.getRange("D6:Q29").format.numberFormat = "#,##0";
details.getRange("R6:R29").format.numberFormat = "0.000";
details.getRange("A6:C29").format.numberFormat = "@";
details.freezePanes.freezeRows(5);
details.freezePanes.freezeColumns(3);

mapped.showGridLines = false;
mapped.tabColor = "#8299A7";
mapped.getRange("A1:D29").format.font = { name: "Arial", size: 10, color: "#263746" };
mapped.getRange("A2").values = [["R/P 映射反应式与标准案例引用"]];
mapped.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#173B57" };
mapped.getRange("A3").values = [["SMILES 字符串仅表示 R/P 端点图；未读取 TS/IRC。完整 XYZ 几何见 ReactionCase JSON 引用。"]];
mapped.getRange("A3:D3").format.font = { name: "Arial", size: 10, italic: true, color: "#5C6D78" };
mapped.getRange("A5:D5").values = [["反应ID", "R/P 映射 SMILES", "case_id", "ReactionCase JSON 相对路径"]];
mapped.getRange("A6:D29").values = infoRows.map((row) => [row[0], row[19], row[20], row[22]]);
mapped.getRange("A5:D5").format = { fill: "#214B65", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
mapped.getRange("A5:D29").format.verticalAlignment = "center";
mapped.getRange("A5:D29").format.borders = { insideHorizontal: { style: "thin", color: "#E4EAF0" }, bottom: { style: "thin", color: "#D5DEE5" } };
mapped.tables.add("A5:D29", true, "MappedEndpointsTable");
for (const [col, width] of Object.entries({ A: 20, B: 115, C: 25, D: 60 })) mapped.getRange(`${col}:${col}`).format.columnWidth = width;
mapped.getRange("A5:D5").format.rowHeight = 34;
mapped.getRange("A6:D29").format.rowHeight = 28;
mapped.getRange("A6:A29").format.wrapText = false;
mapped.getRange("B6:B29").format.wrapText = true;
mapped.getRange("C6:D29").format.wrapText = false;
mapped.getRange("A6:D29").format.rowHeight = 64;
mapped.freezePanes.freezeRows(5);
mapped.freezePanes.freezeColumns(1);

instructions.showGridLines = false;
instructions.tabColor = "#9AAAB4";
instructions.getRange("A1:C12").format.font = { name: "Arial", size: 10, color: "#263746" };
instructions.getRange("A2").values = [["Demo24｜双人端点化学复核说明"]];
instructions.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#173B57" };
instructions.getRange("A4:B11").values = [
  ["复核步骤", "要求"],
  ["独立核对", "两名复核者分别填写“复核者1”和“复核者2”页；完成前不要互看对方结论。"],
  ["先读反应图", "复核者页 D 列由 RDKit 按映射 SMILES 绘制 R→P。原子旁数字是 atom-map；绿=新增键，红=断键，橙=键级变化，蓝=形式电荷变化；图下逐项写出变化。"],
  ["反应中心与机制", "检查成断键、H 迁移、环/芳香变化是否和 R/P 图一致；判断是否像单个基元步骤。"],
  ["映射与几何", "在“映射SMILES”页核对 R/P 图；在 ReactionCase JSON 中查看 XYZ。检查多组分相对位置、碰撞与预装配风险；RXN_0000079731 为对称映射样本。"],
  ["电荷与自旋", "核对形式电荷、局域自由基和预期总自旋。端点 multiplicity 当前留空；不得从 multiplicity_max=1 推断。"],
  ["扫描可行性", "确认驱动键/扫描窗口和观察量；超出 1D 时选 synchronized、path_or_neb 或 reject。"],
  ["最终裁定", "两份复核完成后再填“汇总裁定”。有分歧先解决；accept 必须填写最终 R/P multiplicity。"],
];
instructions.getRange("A4:B4").format = { fill: "#214B65", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
instructions.getRange("A5:B11").format.wrapText = true;
instructions.getRange("A5:A11").format.font = { name: "Arial", size: 10, bold: true, color: "#173B57" };
instructions.getRange("A4:B11").format.borders = { insideHorizontal: { style: "thin", color: "#E4EAF0" }, bottom: { style: "thin", color: "#D5DEE5" } };
instructions.getRange("A4:A11").format.columnWidth = 22;
instructions.getRange("B4:B11").format.columnWidth = 105;
instructions.getRange("A5:B11").format.rowHeight = 48;
instructions.getRange("A13:B13").values = [["信息边界", "图像仅按映射 R/P SMILES 绘制端点键图差异，不读取 TS/IRC；立体化学、几何和反应上下文仍需人工核对。24 条当前均待人工确认，不是化学接受名单。"]];
instructions.getRange("A13").format.font = { name: "Arial", size: 10, bold: true, color: "#8A5B16" };
instructions.getRange("B13").format.wrapText = true;
instructions.getRange("A13:B13").format.rowHeight = 48;

workbook.recalculate();
const sheetSummary = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 1800 });
console.log(sheetSummary.ndjson);
const reviewSample = await workbook.inspect({ kind: "region", sheetId: "复核者1", range: "A5:N8", maxChars: 2200 });
console.log(reviewSample.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!", options: { useRegex: true, maxResults: 100 }, summary: "review workbook formula error scan" });
console.log(errors.ndjson);
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["复核者1", "复核者2", "汇总裁定", "候选摘要", "映射SMILES", "复核说明"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `demo24-review-${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(workbookPath);
const savedWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const savedSummary = await savedWorkbook.inspect({ kind: "sheet", include: "id,name", maxChars: 1800 });
console.log(savedSummary.ndjson);
const savedReview = await savedWorkbook.inspect({ kind: "region", sheetId: "复核者1", range: "A5:N8", maxChars: 2200 });
console.log(savedReview.ndjson);
console.log(`saved ${workbookPath}`);
