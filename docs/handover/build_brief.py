"""Build the brief study guide (plain-language companion to the technical handover).

    C:\\Users\\shash\\.venvs\\msf_fusion\\Scripts\\python.exe docs/handover/build_brief.py

Output: docs/handover/Project_Explained_Brief_Guide.pdf
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph, Spacer  # noqa: E402

import diagrams  # noqa: E402
import render as R  # noqa: E402
from content.brief import BRIEF, HOW_TO_USE  # noqa: E402

OUT_PDF = HERE / "Project_Explained_Brief_Guide.pdf"
TITLE = "Multi-Sensor Fusion & Trajectory Planning — Brief Study Guide"


def cover() -> list:
    title = R.ParagraphStyle("ct", fontName="Body-Bold", fontSize=24, leading=30, textColor=R.colors.white)
    sub = R.ParagraphStyle("cs", fontName="Body", fontSize=11.5, leading=16, textColor=R.colors.HexColor("#dbe7f8"))
    flows = [Spacer(1, 12 * mm), Paragraph("The Project Explained", title),
             Spacer(1, 4 * mm), Paragraph("A brief study guide to the multi-sensor fusion &amp; trajectory planning system", sub),
             Spacer(1, 4 * mm), Paragraph("What it does · how each part works · the results · the questions you will be asked", sub),
             Spacer(1, 40 * mm)]
    rows = [["Item", "Value"],
            ["Reading time", "≈ 45 minutes (chapters 1–3: 10 minutes for the big picture)"],
            ["Companion document", "Multi_Sensor_Fusion_Technical_Handover.pdf (every file, class and function)"],
            ["Project location", r"D:\researcpaper\multi_sensor_fusion"],
            ["Numbers from", "configs/, outputs/ablation/results.md, outputs/evaluation/joint/report.json, model F checkpoint"],
            ["Document generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Generator", "docs/handover/build_brief.py"]]
    flows.append(R.table(rows, [1.4, 4]))
    flows.append(NextPageTemplate("normal"))
    flows.append(PageBreak())
    return flows


def main() -> None:
    print("rendering diagrams ...")
    diagrams.all_diagrams()
    story = cover()
    toc_heading = R.Heading("Contents", 0)
    toc_heading.in_toc = False
    story += [toc_heading, R.toc(), Spacer(1, 8 * mm)] + R.markdown(HOW_TO_USE) + [PageBreak()]
    for i, (title, text) in enumerate(BRIEF, 1):
        heading = R.heading(f"{i}. {title}", 0)
        story += heading[1:] if isinstance(story[-1], PageBreak) else heading
        story += R.markdown(text)
    print("building PDF ...")
    doc = R.HandoverDoc(OUT_PDF, TITLE, footer="multi_sensor_fusion · brief study guide", toc_levels=0)
    doc.multiBuild(story)
    from pypdf import PdfReader
    print(f"written {OUT_PDF} ({len(PdfReader(str(OUT_PDF)).pages)} pages)")


if __name__ == "__main__":
    main()
