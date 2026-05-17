#!/usr/bin/env python3
"""Plot a 2D structure overview for representative high-confidence cases."""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from textwrap import wrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass
class PanelCase:
    case_rank: int
    pred_prob: float
    pair_category: str
    smiles_a: str
    smiles_b: str
    motifs_a: str
    motifs_b: str
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 6-case medicinal chemistry overview.")
    parser.add_argument(
        "--cases-csv",
        type=Path,
        default=Path(
            "/home/xwl/药物禁忌/outputs/run_20260406_103701_autogluon_boosted/"
            "feature_sets/E+F+M/synthesis_planning_cases/selected_cases_top.csv"
        ),
    )
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_cases(path: Path, top_n: int) -> list[PanelCase]:
    df = pd.read_csv(path).head(top_n).copy()
    cases: list[PanelCase] = []
    for row in df.to_dict(orient="records"):
        cases.append(
            PanelCase(
                case_rank=int(row["case_rank"]),
                pred_prob=float(row["pred_prob"]),
                pair_category=str(row["pair_category"]),
                smiles_a=str(row["canonical_smiles_a"]),
                smiles_b=str(row["canonical_smiles_b"]),
                motifs_a=str(row["motifs_a"]),
                motifs_b=str(row["motifs_b"]),
                rationale=str(row["medchem_rationale"]),
            )
        )
    return cases


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arialbd.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(text: str, width: int) -> list[str]:
    return wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    rdMolDraw2D.PrepareMolForDrawing(mol)
    return mol


def draw_molecule_png(smiles: str, width: int, height: int) -> Image.Image:
    mol = mol_from_smiles(smiles)
    img = Draw.MolToImage(mol, size=(width, height), kekulize=True, fitImage=True)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def draw_molecule_svg(smiles: str, width: int, height: int) -> str:
    mol = mol_from_smiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.05
    opts.baseFontSize = 0.9
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    body = re.sub(r"^.*?<svg[^>]*>", "", svg, flags=re.S)
    body = re.sub(r"</svg>\s*$", "", body, flags=re.S)
    return body.strip()


def render_png(cases: list[PanelCase], output_path: Path) -> None:
    cols = 2
    rows = 3
    panel_w = 1200
    panel_h = 620
    margin = 50
    gutter_x = 40
    gutter_y = 40
    title_h = 90
    figure_w = margin * 2 + cols * panel_w + gutter_x
    figure_h = margin * 2 + rows * panel_h + (rows - 1) * gutter_y + title_h

    bg = "#ffffff"
    panel_bg = "#fcfcfc"
    border = "#d9dee7"
    title_color = "#1d2a39"
    sub_color = "#49566a"
    accent = "#5977e3"
    arrow_color = "#8a94a6"

    img = Image.new("RGBA", (figure_w, figure_h), bg)
    draw = ImageDraw.Draw(img)
    font_title = get_font(40, bold=True)
    font_panel = get_font(25, bold=True)
    font_text = get_font(24, bold=False)
    font_small = get_font(22, bold=False)
    font_small_bold = get_font(22, bold=True)

    draw.text(
        (margin, 18),
        "Representative high-confidence predictions for drug-discovery-oriented synthesis planning",
        fill=title_color,
        font=font_title,
    )

    mol_w = 450
    mol_h = 260

    for idx, case in enumerate(cases):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (panel_w + gutter_x)
        y0 = margin + title_h + row * (panel_h + gutter_y)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=26, fill=panel_bg, outline=border, width=3)

        header = f"Case {idx + 1}    P = {case.pred_prob:.3f}"
        draw.text((x0 + 30, y0 + 24), header, fill=title_color, font=font_panel)
        draw.text((x0 + 30, y0 + 66), case.pair_category, fill=accent, font=font_small_bold)

        mol_y = y0 + 120
        mol_a_x = x0 + 30
        mol_b_x = x0 + panel_w - 30 - mol_w

        draw.rounded_rectangle((mol_a_x, mol_y, mol_a_x + mol_w, mol_y + mol_h), radius=18, fill="#ffffff", outline=border, width=2)
        draw.rounded_rectangle((mol_b_x, mol_y, mol_b_x + mol_w, mol_y + mol_h), radius=18, fill="#ffffff", outline=border, width=2)

        mol_a_img = draw_molecule_png(case.smiles_a, mol_w - 24, mol_h - 24)
        mol_b_img = draw_molecule_png(case.smiles_b, mol_w - 24, mol_h - 24)
        img.alpha_composite(mol_a_img, (mol_a_x + 12, mol_y + 12))
        img.alpha_composite(mol_b_img, (mol_b_x + 12, mol_y + 12))

        arrow_y = mol_y + mol_h // 2
        arrow_x0 = mol_a_x + mol_w + 30
        arrow_x1 = mol_b_x - 30
        draw.line((arrow_x0, arrow_y, arrow_x1, arrow_y), fill=arrow_color, width=8)
        draw.polygon(
            [(arrow_x1, arrow_y), (arrow_x1 - 22, arrow_y - 12), (arrow_x1 - 22, arrow_y + 12)],
            fill=arrow_color,
        )

        draw.text((mol_a_x + 10, mol_y + mol_h + 14), "Reactant A", fill=title_color, font=font_small_bold)
        draw.text((mol_b_x + 10, mol_y + mol_h + 14), "Reactant B", fill=title_color, font=font_small_bold)

        motifs_a_lines = wrap_text("A: " + case.motifs_a, width=36)[:2]
        motifs_b_lines = wrap_text("B: " + case.motifs_b, width=36)[:2]
        motifs_a_y = mol_y + mol_h + 48
        motifs_b_y = mol_y + mol_h + 48
        for line in motifs_a_lines:
            draw.text((mol_a_x + 10, motifs_a_y), line, fill=sub_color, font=font_small)
            motifs_a_y += 26
        for line in motifs_b_lines:
            draw.text((mol_b_x + 10, motifs_b_y), line, fill=sub_color, font=font_small)
            motifs_b_y += 26

        rationale_lines = wrap_text(case.rationale, width=95)
        text_y = y0 + 500
        for line in rationale_lines[:3]:
            draw.text((x0 + 30, text_y), line, fill=sub_color, font=font_text)
            text_y += 30

    img.save(output_path)


def svg_text(x: int, y: int, text: str, size: int, color: str, weight: str = "normal") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Liberation Sans, DejaVu Sans, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(text)}</text>'
    )


def render_svg(cases: list[PanelCase], output_path: Path) -> None:
    cols = 2
    rows = 3
    panel_w = 1200
    panel_h = 620
    margin = 50
    gutter_x = 40
    gutter_y = 40
    title_h = 90
    figure_w = margin * 2 + cols * panel_w + gutter_x
    figure_h = margin * 2 + rows * panel_h + (rows - 1) * gutter_y + title_h

    bg = "#ffffff"
    panel_bg = "#fcfcfc"
    border = "#d9dee7"
    title_color = "#1d2a39"
    sub_color = "#49566a"
    accent = "#5977e3"
    arrow_color = "#8a94a6"

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{figure_w}" height="{figure_h}" viewBox="0 0 {figure_w} {figure_h}">',
        f'<rect width="{figure_w}" height="{figure_h}" fill="{bg}"/>',
        svg_text(
            margin,
            55,
            "Representative high-confidence predictions for drug-discovery-oriented synthesis planning",
            40,
            title_color,
            "700",
        ),
    ]

    mol_w = 450
    mol_h = 260

    for idx, case in enumerate(cases):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (panel_w + gutter_x)
        y0 = margin + title_h + row * (panel_h + gutter_y)

        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="26" ry="26" '
            f'fill="{panel_bg}" stroke="{border}" stroke-width="3"/>'
        )
        parts.append(svg_text(x0 + 30, y0 + 50, f"Case {idx + 1}    P = {case.pred_prob:.3f}", 25, title_color, "700"))
        parts.append(svg_text(x0 + 30, y0 + 92, case.pair_category, 22, accent, "700"))

        mol_y = y0 + 120
        mol_a_x = x0 + 30
        mol_b_x = x0 + panel_w - 30 - mol_w
        parts.append(
            f'<rect x="{mol_a_x}" y="{mol_y}" width="{mol_w}" height="{mol_h}" rx="18" ry="18" '
            f'fill="#ffffff" stroke="{border}" stroke-width="2"/>'
        )
        parts.append(
            f'<rect x="{mol_b_x}" y="{mol_y}" width="{mol_w}" height="{mol_h}" rx="18" ry="18" '
            f'fill="#ffffff" stroke="{border}" stroke-width="2"/>'
        )

        svg_a = draw_molecule_svg(case.smiles_a, mol_w - 24, mol_h - 24)
        svg_b = draw_molecule_svg(case.smiles_b, mol_w - 24, mol_h - 24)
        parts.append(f'<g transform="translate({mol_a_x + 12},{mol_y + 12})">{svg_a}</g>')
        parts.append(f'<g transform="translate({mol_b_x + 12},{mol_y + 12})">{svg_b}</g>')

        arrow_y = mol_y + mol_h // 2
        arrow_x0 = mol_a_x + mol_w + 30
        arrow_x1 = mol_b_x - 30
        parts.append(
            f'<line x1="{arrow_x0}" y1="{arrow_y}" x2="{arrow_x1}" y2="{arrow_y}" '
            f'stroke="{arrow_color}" stroke-width="8" stroke-linecap="round"/>'
        )
        parts.append(
            f'<polygon points="{arrow_x1},{arrow_y} {arrow_x1 - 22},{arrow_y - 12} {arrow_x1 - 22},{arrow_y + 12}" '
            f'fill="{arrow_color}"/>'
        )

        parts.append(svg_text(mol_a_x + 10, mol_y + mol_h + 36, "Reactant A", 22, title_color, "700"))
        parts.append(svg_text(mol_b_x + 10, mol_y + mol_h + 36, "Reactant B", 22, title_color, "700"))
        motifs_a_lines = wrap_text("A: " + case.motifs_a, width=36)[:2]
        motifs_b_lines = wrap_text("B: " + case.motifs_b, width=36)[:2]
        motifs_a_y = mol_y + mol_h + 70
        motifs_b_y = mol_y + mol_h + 70
        for line in motifs_a_lines:
            parts.append(svg_text(mol_a_x + 10, motifs_a_y, line, 20, sub_color))
            motifs_a_y += 24
        for line in motifs_b_lines:
            parts.append(svg_text(mol_b_x + 10, motifs_b_y, line, 20, sub_color))
            motifs_b_y += 24

        rationale_lines = wrap_text(case.rationale, width=95)[:3]
        text_y = y0 + 530
        for line in rationale_lines:
            parts.append(svg_text(x0 + 30, text_y, line, 22, sub_color))
            text_y += 30

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases_csv, args.top_n)
    output_dir = args.output_dir or args.cases_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "high_confidence_cases_overview.png"
    svg_path = output_dir / "high_confidence_cases_overview.svg"

    render_png(cases, png_path)
    render_svg(cases, svg_path)

    print(png_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
