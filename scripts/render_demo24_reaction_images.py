"""Render mapped Demo24 reactant/product structure cards with RDKit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


COLORS = {
    "formed": (33, 145, 93),
    "broken": (205, 65, 62),
    "order": (222, 139, 38),
    "charge": (63, 113, 184),
    "center": (224, 161, 49),
}
RGB = {
    "background": (255, 255, 255),
    "ink": (38, 55, 70),
    "muted": (95, 111, 122),
    "line": (220, 228, 234),
    "green": COLORS["formed"],
    "red": COLORS["broken"],
    "orange": COLORS["order"],
    "blue": COLORS["charge"],
}


def _map_atoms(mol: Chem.Mol) -> dict[int, int]:
    result: dict[int, int] = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if not map_num:
            raise ValueError("a mapped endpoint contains an atom without an atom-map number")
        if map_num in result:
            raise ValueError(f"duplicate atom-map number: {map_num}")
        result[map_num] = atom.GetIdx()
    return result


def _map_bonds(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    result = {}
    for bond in mol.GetBonds():
        left = bond.GetBeginAtom().GetAtomMapNum()
        right = bond.GetEndAtom().GetAtomMapNum()
        result[tuple(sorted((left, right)))] = float(bond.GetBondTypeAsDouble())
    return result


def _bond_label(key: tuple[int, int]) -> str:
    return f"{key[0]}–{key[1]}"


def _charge(mol: Chem.Mol, atom_idx: int) -> int:
    return mol.GetAtomWithIdx(atom_idx).GetFormalCharge()


def _differences(reactant: Chem.Mol, product: Chem.Mol) -> dict[str, Any]:
    reactant_atoms = _map_atoms(reactant)
    product_atoms = _map_atoms(product)
    reactant_bonds = _map_bonds(reactant)
    product_bonds = _map_bonds(product)
    common_maps = set(reactant_atoms) & set(product_atoms)
    formed = sorted(set(product_bonds) - set(reactant_bonds))
    broken = sorted(set(reactant_bonds) - set(product_bonds))
    order_changes = sorted(key for key in set(reactant_bonds) & set(product_bonds)
                           if reactant_bonds[key] != product_bonds[key])
    charge_changes = sorted((map_num, _charge(reactant, reactant_atoms[map_num]),
                             _charge(product, product_atoms[map_num]))
                            for map_num in common_maps
                            if _charge(reactant, reactant_atoms[map_num])
                            != _charge(product, product_atoms[map_num]))
    hydrogen_moves = []
    for map_num in sorted(common_maps):
        r_atom = reactant.GetAtomWithIdx(reactant_atoms[map_num])
        p_atom = product.GetAtomWithIdx(product_atoms[map_num])
        if r_atom.GetAtomicNum() != 1:
            continue
        r_neighbors = sorted(n.GetAtomMapNum() for n in r_atom.GetNeighbors())
        p_neighbors = sorted(n.GetAtomMapNum() for n in p_atom.GetNeighbors())
        if r_neighbors != p_neighbors and (r_neighbors or p_neighbors):
            hydrogen_moves.append((map_num, r_neighbors, p_neighbors))
    return {
        "reactant_maps": sorted(reactant_atoms),
        "product_maps": sorted(product_atoms),
        "mapping_exact": set(reactant_atoms) == set(product_atoms),
        "formed": formed,
        "broken": broken,
        "order_changes": order_changes,
        "charge_changes": charge_changes,
        "hydrogen_moves": hydrogen_moves,
        "reactant_bonds": reactant_bonds,
        "product_bonds": product_bonds,
        "reactant_atoms": reactant_atoms,
        "product_atoms": product_atoms,
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in font_paths:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _draw_molecule(mol: Chem.Mol, maps: dict[int, int], *, role: str,
                   changes: dict[str, Any]) -> Image.Image:
    molecule = Chem.Mol(mol)
    atoms_to_highlight: set[int] = set()
    atom_colors: dict[int, tuple[float, float, float]] = {}
    bond_indices: set[int] = set()
    bond_colors: dict[int, tuple[float, float, float]] = {}

    for atom in molecule.GetAtoms():
        map_num = atom.GetAtomMapNum()
        atom.SetProp("atomNote", str(map_num))
        # Show map IDs as small annotations beside atoms, not inside element labels.
        atom.SetAtomMapNum(0)

    for key, color_name in [(key, "formed") for key in changes["formed"]] + [
            (key, "broken") for key in changes["broken"]] + [
            (key, "order") for key in changes["order_changes"]]:
        present = changes["reactant_bonds"] if role == "R" else changes["product_bonds"]
        if key not in present:
            continue
        left, right = (maps.get(key[0]), maps.get(key[1]))
        if left is None or right is None:
            continue
        bond = molecule.GetBondBetweenAtoms(left, right)
        if bond is None:
            continue
        bond_indices.add(bond.GetIdx())
        bond_colors[bond.GetIdx()] = tuple(component / 255 for component in COLORS[color_name])
        atoms_to_highlight.update((left, right))
        atom_colors[left] = tuple(component / 255 for component in COLORS[color_name])
        atom_colors[right] = tuple(component / 255 for component in COLORS[color_name])

    for map_num, _, _ in changes["charge_changes"]:
        idx = maps.get(map_num)
        if idx is not None:
            atoms_to_highlight.add(idx)
            atom_colors[idx] = tuple(component / 255 for component in COLORS["charge"])
    for map_num, _, _ in changes["hydrogen_moves"]:
        idx = maps.get(map_num)
        if idx is not None:
            atoms_to_highlight.add(idx)
            atom_colors.setdefault(idx, tuple(component / 255 for component in COLORS["center"]))

    drawer = rdMolDraw2D.MolDraw2DCairo(430, 254)
    options = drawer.drawOptions()
    options.padding = 0.12
    options.bondLineWidth = 2.2
    options.annotationFontScale = 0.72
    options.fixedFontSize = 17
    options.addAtomIndices = False
    drawer.DrawMolecule(molecule,
        highlightAtoms=sorted(atoms_to_highlight), highlightBonds=sorted(bond_indices),
        highlightAtomColors=atom_colors, highlightBondColors=bond_colors)
    drawer.FinishDrawing()
    return Image.open(__import__("io").BytesIO(drawer.GetDrawingText())).convert("RGBA")


def _summary_lines(changes: dict[str, Any]) -> list[tuple[str, tuple[int, int, int]]]:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if changes["formed"]:
        lines.append(("新增键（绿）: " + ", ".join(map(_bond_label, changes["formed"])), RGB["green"]))
    if changes["broken"]:
        lines.append(("断键（红）: " + ", ".join(map(_bond_label, changes["broken"])), RGB["red"]))
    if changes["order_changes"]:
        labels = [f"{_bond_label(key)} {changes['reactant_bonds'][key]:g}→{changes['product_bonds'][key]:g}"
                  for key in changes["order_changes"]]
        lines.append(("键级变化（橙）: " + ", ".join(labels), RGB["orange"]))
    if changes["charge_changes"]:
        labels = [f"map {map_num} {before:+d}→{after:+d}"
                  for map_num, before, after in changes["charge_changes"]]
        lines.append(("形式电荷变化（蓝）: " + ", ".join(labels), RGB["blue"]))
    if changes["hydrogen_moves"]:
        labels = []
        for map_num, before, after in changes["hydrogen_moves"]:
            left = ",".join(map(str, before)) or "无"
            right = ",".join(map(str, after)) or "无"
            labels.append(f"H map {map_num}: {left}→{right}")
        lines.append(("氢原子连接变化: " + "; ".join(labels), RGB["ink"]))
    if not lines:
        lines.append(("映射键图无成断键或形式电荷变化。请人工核对立体化学和反应上下文。", RGB["muted"]))
    if not changes["mapping_exact"]:
        lines.append(("警示: R/P atom-map 集合不一致，请核对映射。", RGB["red"]))
    return lines


def _render_card(reaction_id: str, stratum: str, family: str, reaction_smiles: str,
                 reactant: Chem.Mol, product: Chem.Mol, changes: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (930, 540), RGB["background"])
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22, bold=True)
    label_font = _font(18, bold=True)
    legend_font = _font(15)
    draw.text((18, 12), f"{reaction_id}   {stratum}", font=title_font, fill=RGB["ink"])
    family_label = family.replace("|", " · ") if family else "未标注反应类型"
    draw.text((18, 40), family_label, font=_font(15), fill=RGB["muted"])
    draw.text((145, 66), "R 反应物", font=label_font, fill=RGB["ink"])
    draw.text((645, 66), "P 产物", font=label_font, fill=RGB["ink"])

    reactant_image = _draw_molecule(reactant, changes["reactant_atoms"], role="R", changes=changes)
    product_image = _draw_molecule(product, changes["product_atoms"], role="P", changes=changes)
    canvas.paste(reactant_image, (12, 88), reactant_image)
    canvas.paste(product_image, (488, 88), product_image)
    draw.line((450, 208, 475, 208), fill=RGB["muted"], width=4)
    draw.polygon([(476, 198), (490, 208), (476, 218)], fill=RGB["muted"])
    draw.line((18, 350, 912, 350), fill=RGB["line"], width=2)
    draw.text((18, 357), "图例：原子旁数字为 atom-map；绿=新增键，红=断键，橙=键级变化，蓝=形式电荷变化。", font=legend_font, fill=RGB["muted"])

    # Wrap by rendered pixel width so every graph difference remains visible.
    lines = _summary_lines(changes)
    font = _font(14)
    y = 386
    for line, color in lines:
        words = line.split(" ")
        wrapped: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textlength(candidate, font=font) > 890:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
        for piece in wrapped:
            draw.text((18, y), piece, font=font, fill=color)
            y += 18
    if y > 532:
        raise ValueError(f"{reaction_id}: reaction-difference annotations exceed card space")
    return canvas


def render(csv_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 24:
        raise ValueError(f"expected 24 reactions, received {len(records)}")
    for row in records:
        reaction_id = row["reaction_id"]
        parts = row["reaction_smiles"].split(">>")
        if len(parts) != 2:
            raise ValueError(f"{reaction_id}: expected one R>>P mapped reaction")
        parser = Chem.SmilesParserParams()
        parser.removeHs = False  # Preserve mapped explicit H atoms for identity and transfer labels.
        reactant = Chem.MolFromSmiles(parts[0], parser)
        product = Chem.MolFromSmiles(parts[1], parser)
        if reactant is None or product is None:
            raise ValueError(f"{reaction_id}: RDKit could not parse an endpoint SMILES")
        changes = _differences(reactant, product)
        if not changes["mapping_exact"]:
            raise ValueError(f"{reaction_id}: R/P atom-map sets differ")
        card = _render_card(reaction_id, row["stratum"], row.get("family_labels", ""),
                            row["reaction_smiles"], reactant, product, changes)
        target = output_dir / f"{reaction_id}.png"
        card.save(target, format="PNG", optimize=True)
        manifest.append({"reaction_id": reaction_id, "image": target.name,
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "mapping_exact": changes["mapping_exact"],
                         "mapped_atom_count_R": len(changes["reactant_maps"]),
                         "mapped_atom_count_P": len(changes["product_maps"]),
                         "formed_bonds": [_bond_label(key) for key in changes["formed"]],
                         "broken_bonds": [_bond_label(key) for key in changes["broken"]],
                         "bond_order_changes": [_bond_label(key) for key in changes["order_changes"]],
                         "charge_changes": changes["charge_changes"],
                         "hydrogen_moves": changes["hydrogen_moves"]})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = render(args.csv, args.output_dir)
    print(json.dumps({"rendered": len(records),
                      "mapping_exact": sum(item["mapping_exact"] for item in records),
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
