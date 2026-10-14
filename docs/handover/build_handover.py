"""Build the technical handover PDF from the code base.

    C:\\Users\\shash\\.venvs\\msf_fusion\\Scripts\\python.exe docs/handover/build_handover.py

Output: docs/handover/Multi_Sensor_Fusion_Technical_Handover.pdf
"""
from __future__ import annotations

import ast
import datetime as dt
import html
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (NextPageTemplate, PageBreak, Paragraph, Spacer, KeepTogether,  # noqa: E402
                                CondPageBreak)

import diagrams  # noqa: E402
import render as R  # noqa: E402
from analysis import analyze_project  # noqa: E402
from content.chapters_a import CHAPTERS_A  # noqa: E402
from content.chapters_b import CHAPTERS_B  # noqa: E402
from content.file_notes import FILES  # noqa: E402
from content.function_notes import NOTES  # noqa: E402
from content.walk_data import WALKTHROUGHS as W_DATA  # noqa: E402
from content.walk_models import WALKTHROUGHS as W_MODELS  # noqa: E402
from content.walk_system import WALKTHROUGHS as W_SYSTEM  # noqa: E402

OUT_PDF = HERE / "Multi_Sensor_Fusion_Technical_Handover.pdf"
TITLE = "Multi-Sensor Fusion & Trajectory Planning on nuScenes — Technical Handover"
SKIP_DIRS = {"__pycache__", ".pytest_cache", "build", ".git"}
SUMMARISED_DIRS = {"outputs", "checkpoints"}
WARNINGS: list[str] = []


def doc_inline(text: str) -> str:
    """Escape docstrings: only `code` spans are formatted (no * emphasis, which docstrings use for math)."""
    text = R.safe_glyphs(text.replace("``", "`"))
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f'<font face="Mono" size="-0.8" color="#1c5cab">{html.escape(p[1:-1], quote=False)}</font>')
        else:
            out.append(html.escape(p, quote=False))
    return "".join(out)


def first_paragraph(doc: str) -> str:
    return doc.strip().split("\n\n")[0].replace("\n", " ").strip() if doc else ""


# ============================================================================ generated sections
def project_files() -> list[Path]:
    files = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.parts[0] in SUMMARISED_DIRS:
            continue
        if rel.suffix in (".pdf", ".pyc"):
            continue
        files.append(rel)
    return files


def gen_tree() -> list:
    files = project_files()
    tree: dict = {}
    for rel in files:
        node = tree
        for part in rel.parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[rel.parts[-1]] = None
    for d in sorted(SUMMARISED_DIRS):
        n = sum(1 for _ in (ROOT / d).rglob("*") if _.is_file()) if (ROOT / d).exists() else 0
        tree[f"{d}/"] = {f"... {n} generated files (see Artifacts chapter)": None}
    lines = ["multi_sensor_fusion/"]

    def walk(node: dict, prefix: str) -> None:
        keys = sorted(node, key=lambda k: (not k.endswith("/"), k.lower()))
        for i, k in enumerate(keys):
            last = i == len(keys) - 1
            lines.append(prefix + ("└── " if last else "├── ") + k)
            if isinstance(node[k], dict):
                walk(node[k], prefix + ("    " if last else "│   "))

    walk(tree, "")
    return R.code_block("\n".join(lines), "text")


def py_deps(mod_info) -> str:
    internal = [m for m in mod_info.internal_imports]
    external = [m for m in mod_info.external_imports if m not in ("typing", "__future__")]
    return ", ".join(internal + external) or "–"


def gen_inventory(mods) -> list:
    by_path = {m.path: m for m in mods.values()}
    rows = [["File", "Location", "Purpose", "Inputs", "Outputs", "Dependencies"]]
    covered = set()
    for rel in project_files():
        key = rel.as_posix()
        note = FILES.get(key)
        if note is None and key.startswith("docs/handover/content/"):
            note = ("Hand-written documentation content used by build_handover.py.", "Documentation tooling.", "–", "–")
        if note is None:
            WARNINGS.append(f"inventory: no file note for {key}")
            note = ("Not identifiable from the current codebase.", "", "–", "–")
        covered.add(key)
        deps = py_deps(by_path[key]) if key in by_path else "–"
        purpose = note[0] + (f" *Why:* {note[1]}" if note[1] and note[1] not in ("–",) else "")
        rows.append([rel.name, str(rel.parent.as_posix()) if str(rel.parent) != "." else "(root)", purpose, note[2], note[3], deps])
    for key in FILES:
        if key not in covered and not (ROOT / key).exists():
            WARNINGS.append(f"inventory: note for non-existent file {key}")
    return [R.table(rows, [1.25, 1.1, 3.0, 1.35, 1.35, 1.6], font_scale=0.9)]


def gen_package_graph(mods) -> list:
    graph: dict[str, set] = defaultdict(set)
    for m in mods.values():
        src = m.module.split(".")[0]
        for imp in m.internal_imports:
            dst = imp.split(".")[0]
            if dst != src:
                graph[src].add(dst)
    rows = [["Package", "Imports project packages"]]
    for pkg in sorted({m.module.split(".")[0] for m in mods.values()}):
        rows.append([pkg, ", ".join(sorted(graph.get(pkg, []))) or "–"])
    return [R.table(rows, [1, 4])]


def load_cfg():
    from utils.config import load_config
    return load_config(), load_config(use_local=False)


def gen_stages(cfg) -> list:
    rows = [["Stage", "Description", "Losses", "Init from", "Frozen / trainable", "Epochs", "LR scale", "Overrides"]]
    for name, s in cfg.stages.items():
        init = "; ".join(f"{i['stage']}" + (f" ({', '.join(i['prefixes'])})" if i.get("prefixes") else " (all)")
                         for i in (s.get("init_from") or [])) or "–"
        frozen = ", ".join(s.get("frozen") or []) or "–"
        if s.get("trainable_only"):
            frozen = "trainable only: " + ", ".join(s["trainable_only"])
        ov = s.get("overrides") or {}
        rows.append([name + (" (shared)" if s.get("shared") else ""), s.get("description", ""), ", ".join(s.losses), init,
                     frozen, str(s.get("epochs", cfg.train.epochs)), str(s.get("lr_scale", 1.0)),
                     json.dumps(ov.to_dict() if hasattr(ov, "to_dict") else ov)[:160] if ov else "–"])
    return [R.table(rows, [0.9, 2.2, 1.4, 1.6, 1.2, 0.5, 0.5, 1.8])]


def gen_stage_losses(cfg) -> list:
    groups = ["detection", "trajectory", "feature_selection", "fusion", "depth"]
    rows = [["Stage"] + groups]
    for name, s in cfg.stages.items():
        rows.append([name] + ["✓" if g in s.losses else "" for g in groups])
    return [R.table(rows, [1.4, 1, 1, 1.3, 0.8, 0.8])]


def yaml_comments() -> dict[str, str]:
    """Map dotted keys → inline comment from the base YAML files."""
    comments: dict[str, str] = {}
    for name in ("dataset.yaml", "model.yaml", "fusion.yaml", "planning.yaml", "training.yaml"):
        stack: list[tuple[int, str]] = []
        for line in (ROOT / "configs" / name).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            m = re.match(r"^(\s*)([A-Za-z0-9_]+):(.*)$", line)
            if not m:
                continue
            indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, key))
            dotted = ".".join(k for _, k in stack)
            if "#" in rest and not rest.strip().startswith('"${'):
                comments[dotted] = rest.split("#", 1)[1].strip()
            elif "#" in rest:
                comments[dotted] = rest.rsplit("#", 1)[1].strip() if rest.count("#") else ""
    return comments


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v and not (prefix == "" and k == "stages"):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def gen_config(cfg_eff, cfg_base) -> list:
    comments = yaml_comments()
    base = flatten(cfg_base.to_dict())
    eff = flatten(cfg_eff.to_dict())
    flows = []
    sections = defaultdict(list)
    for k in base:
        if k.startswith("stages") or k.startswith("runtime.experiment"):
            continue
        sections[k.split(".")[0]].append(k)
    for sec in ("dataset", "bev", "classes", "model", "fusion", "planning", "runtime", "train", "eval", "loss"):
        if sec not in sections:
            continue
        flows += R.heading(f"Section `{sec}`", 2)
        rows = [["Key", "Default (base YAML)", "This machine", "Description"]]
        for k in sections[sec]:
            b = json.dumps(base[k]) if not isinstance(base[k], str) else base[k]
            e = eff.get(k, base[k])
            e = json.dumps(e) if not isinstance(e, str) else e
            rows.append([k, b, e if e != b else "=", comments.get(k, "")])
        flows.append(R.table(rows, [2.0, 1.6, 1.3, 2.6], mono_cols=(0, 1, 2)))
    return flows


def gen_hardware() -> list:
    rows = [["Profile", "Overrides"]]
    for p in sorted((ROOT / "configs" / "hardware").glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rows.append([p.stem, "\n".join(f"{k} = {json.dumps(v)}" for k, v in flatten(data).items())])
    return [R.table(rows, [1, 4], mono_cols=(1,))]


def gen_experiments() -> list:
    data = yaml.safe_load((ROOT / "configs" / "ablation.yaml").read_text(encoding="utf-8"))
    rows = [["Experiment", "Label", "Description", "Fusion overrides"]]
    for name, e in data["experiments"].items():
        rows.append([name, e.get("label", ""), e.get("description", ""),
                     ", ".join(f"{k}={v}" for k, v in e.get("overrides", {}).get("fusion", {}).items())])
    flows = [R.table(rows, [1.6, 1.6, 2.2, 2.6], mono_cols=(0, 3))]
    flows.append(Paragraph(R.inline("Default run: " + ", ".join(f"`{x}`" for x in data.get("default_run", []))),
                           R.S["body"]))
    rows = [["Dashboard label", "Experiment"]] + [[k, v] for k, v in data.get("dashboard_methods", {}).items()]
    flows.append(R.table(rows, [2, 2]))
    return flows


def gen_cli() -> list:
    flows = []
    main_txt = ast.get_docstring(ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))) or ""
    flows += R.heading("main.py", 2)
    flows += R.code_block(main_txt, "text")
    scripts = sorted([*ROOT.glob("scripts/*.py"), ROOT / "training" / "cli.py", ROOT / "evaluation" / "ablation.py"])
    for path in scripts:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flags = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                args.append([", ".join(flags), kw.get("default", ""), kw.get("type", kw.get("action", "")),
                             kw.get("choices", ""), (kw.get("help", "") or "").strip("'\"")])
        if not args:
            continue
        rel = path.relative_to(ROOT).as_posix()
        flows += R.heading(rel, 2)
        doc = first_paragraph(ast.get_docstring(tree) or "")
        if doc:
            flows.append(Paragraph(doc_inline(doc), R.S["body"]))
        flows.append(R.table([["Argument", "Default", "Type / action", "Choices", "Help"]] + args, [1.4, 1.2, 1.0, 1.4, 2.6],
                             mono_cols=(0, 1, 2, 3)))
    return flows


def split_blocks(path: Path, blocks: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    starts, pos = [], 0
    for anchor, _ in blocks:
        for i in range(pos, len(lines)):
            if anchor in lines[i]:
                starts.append(i)
                pos = i + 1
                break
        else:
            raise SystemExit(f"Walkthrough anchor not found in {path.relative_to(ROOT)}: {anchor!r}")
    out = []
    for j, (start, (_, text)) in enumerate(zip(starts, blocks)):
        s = 0 if j == 0 else start
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        out.append((s + 1, "\n".join(lines[s:e]), text))
    return out


def gen_walkthroughs() -> list:
    flows = []
    groups = [("Utilities and data layer", W_DATA), ("Neural-network modules", W_MODELS),
              ("Planning, training, evaluation, inference and application", W_SYSTEM)]
    for gname, items in groups:
        flows += R.heading(gname, 1)
        for item in items:
            path = ROOT / item["file"]
            flows += R.heading(f"`{item['file']}`", 2)
            n_lines = len(path.read_text(encoding="utf-8").splitlines())
            flows.append(Paragraph(R.inline(item["intro"]) + f"  <font color='#7a8191'>({n_lines} lines)</font>", R.S["body"]))
            for start, code, text in split_blocks(path, item["blocks"]):
                flows += R.code_block(code, "python", start_line=start)
                flows += R.markdown(text)
    return flows


def gen_api(mods) -> list:
    flows = []
    cell = R.S["small"]

    def fn_block(f, qual_short: str) -> list:
        sig = html.escape(R.safe_glyphs(f.signature, mono=True), quote=False)
        parts = [f'<font face="Mono" size="7.2" color="#1c5cab"><b>{sig}</b></font>']
        doc = f.docstring
        purpose = first_paragraph(doc) if doc else NOTES.get(f.qualname, "")
        if not purpose:
            WARNINGS.append(f"api: no purpose for {f.qualname}")
            purpose = "Not identifiable from the current codebase."
        parts.append(f"<b>Purpose:</b> {doc_inline(purpose)}")
        if doc and f.qualname in NOTES:
            parts.append(f"<b>Behaviour:</b> {doc_inline(NOTES[f.qualname])}")
        if doc and len(doc.strip().split("\n\n")) > 1:
            rest = " ".join(doc.strip().split("\n\n")[1:]).replace("\n", " ")
            parts.append(f"<b>Details:</b> {doc_inline(rest[:900])}")
        params = [p for p in f.params if p[0] not in ("self", "cls")]
        if params:
            ptxt = "; ".join(f"<font face='Mono' size='6.9'>{html.escape(n)}</font>" +
                             (f": {html.escape(a)}" if a else "") + (f" = {html.escape(d)}" if d else "") for n, a, d in params)
            parts.append(f"<b>Parameters:</b> {ptxt}")
        parts.append(f"<b>Returns:</b> {html.escape(f.returns) if f.returns else 'None / not annotated'}")
        short = lambda q: q.replace(".", "·", 0)
        if f.calls:
            parts.append("<b>Calls:</b> " + ", ".join(html.escape(c) for c in f.calls[:14]) + (" …" if len(f.calls) > 14 else ""))
        if f.called_by:
            parts.append("<b>Called by:</b> " + ", ".join(html.escape(c) for c in f.called_by[:14]) +
                         (" …" if len(f.called_by) > 14 else ""))
        if f.raises:
            parts.append("<b>Raises:</b> " + "; ".join(html.escape(r) for r in f.raises))
        if f.io:
            parts.append("<b>Side effects (file / disk I/O calls):</b> " + ", ".join(f.io))
        if f.decorators:
            parts.append("<b>Decorators:</b> " + ", ".join(html.escape(d) for d in f.decorators))
        parts.append(f"<font color='#9aa1ad'>lines {f.lineno}–{f.end_lineno}</font>")
        return [Paragraph("<br/>".join(parts), cell), Spacer(1, 5)]

    by_pkg = defaultdict(list)
    for m in sorted(mods.values(), key=lambda m: m.path):
        by_pkg[m.path.split("/")[0] if "/" in m.path else "(root)"].append(m)
    for pkg, items in by_pkg.items():
        flows += R.heading(f"Package `{pkg}`", 1)
        for m in items:
            flows += R.heading(f"`{m.path}`", 2)
            meta = [f"<b>Module:</b> <font face='Mono'>{m.module}</font> · {m.lines} lines" +
                    (" · <b>entry point</b> (runnable script)" if m.is_entry_point else "")]
            doc = first_paragraph(m.docstring)
            if doc:
                meta.append(f"<b>Purpose:</b> {doc_inline(doc)}")
            elif m.path in FILES:
                meta.append(f"<b>Purpose:</b> {R.inline(FILES[m.path][0])}")
            meta.append("<b>Imports (project):</b> " + (", ".join(m.internal_imports) or "–"))
            meta.append("<b>Imports (third-party / stdlib):</b> " + (", ".join(x for x in m.external_imports if x != "typing") or "–"))
            meta.append("<b>Imported by:</b> " + (", ".join(m.imported_by) or "–"))
            flows.append(Paragraph("<br/>".join(meta), R.S["small"]))
            flows.append(Spacer(1, 4))
            if m.constants:
                flows.append(R.table([["Constant", "Value"]] + [[k, v] for k, v in m.constants], [1.3, 4], mono_cols=(0, 1)))
                flows.append(Spacer(1, 4))
            for c in m.classes:
                flows += R.heading(f"class `{c.name}`" + (f"({', '.join(c.bases)})" if c.bases else ""), 3)
                cdoc = c.docstring or ""
                if cdoc:
                    flows.append(Paragraph(doc_inline(cdoc.replace("\n\n", "  ").replace("\n", " ")[:1500]), R.S["small"]))
                if c.attributes:
                    flows.append(Paragraph("<b>Instance attributes (set in __init__):</b> " + "; ".join(
                        f"<font face='Mono' size='6.9'>{html.escape(a)}</font> = {html.escape(v)}" for a, v in c.attributes),
                        R.S["small"]))
                if c.class_vars:
                    flows.append(Paragraph("<b>Class attributes / fields:</b> " + "; ".join(
                        f"<font face='Mono' size='6.9'>{html.escape(a)}</font> = {html.escape(v)}" for a, v in c.class_vars),
                        R.S["small"]))
                flows.append(Spacer(1, 3))
                for f in c.methods:
                    flows += fn_block(f, f.name)
            if m.functions:
                flows += R.heading("Functions", 3)
                for f in m.functions:
                    flows += fn_block(f, f.name)
    return flows


def gen_tests(mods) -> list:
    rows = [["Test", "File", "What it verifies", "Needs nuScenes"]]
    for m in sorted(mods.values(), key=lambda m: m.path):
        if not m.path.startswith("tests/test_"):
            continue
        for f in m.functions:
            if f.name.startswith("test_"):
                purpose = first_paragraph(f.docstring) or NOTES.get(f.qualname, "")
                needs = "yes" if "requires_dataset" in " ".join(f.decorators) else "no"
                rows.append([f.name, m.path, purpose, needs])
    return [R.table(rows, [2.4, 1.4, 3.0, 0.7], mono_cols=(0,))]


def gen_checkpoints() -> list:
    import torch

    rows = [["Checkpoint", "Size (MB)", "Experiment / stage", "Epoch", "Stored val. NDS", "Stored val. mAP", "Stored val. ADE"]]
    for p in sorted((ROOT / "checkpoints").rglob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover
            rows.append([p.relative_to(ROOT).as_posix(), "", f"unreadable: {exc}", "", "", "", ""])
            continue
        met = ck.get("metrics", {}) or {}
        det = met.get("detection", {}) or {}
        plan = met.get("planning_learned", {}) or {}
        ex = ck.get("extra", {}) or {}
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "–"
        rows.append([p.relative_to(ROOT).as_posix(), f"{p.stat().st_size / 2**20:.1f}",
                     f"{ex.get('experiment', '?')} / {ex.get('stage', '?')}", str(ck.get("epoch")), fmt(det.get("NDS")),
                     fmt(det.get("mAP")), fmt(plan.get("ade"))])
        del ck
    flows = [R.table(rows, [3.0, 0.7, 2.1, 0.5, 0.8, 0.8, 0.8], mono_cols=(0,))]
    flows.append(Paragraph(R.inline(
        "`best.pt` = epoch with the best validation NDS during training (selected on the validation split); `last.pt` = "
        "final epoch (used for the reported ablation). The stored metrics were computed during training with the "
        "in-memory evaluator."), R.S["small"]))
    return flows


OUTPUT_DESC = {
    "ablation": "ablation results (results.md/csv/json, per-experiment JSON, plots; *_best_epoch = best-validation protocol)",
    "bev": "BEV galleries rendered by scripts/render_bev.py",
    "cache": "info caches (pickles) and reduced map rasters",
    "curriculum_test": "mechanical test of the 7-stage curriculum (2 samples per stage, CPU)",
    "eval_check": "submission written by scripts/check_eval_vs_devkit.py",
    "evaluation": "reports of scripts/evaluate.py (incl. official devkit run of model F)",
    "inference": "replay frames and videos written by scripts/inference.py",
    "logs": "training logs (run.log, history.jsonl) and run logs",
    "validation": "sensor validation report and images",
}


def gen_outputs() -> list:
    rows = [["Folder", "Files", "Size (MB)", "Content"]]
    base = ROOT / "outputs"
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        files = [f for f in d.rglob("*") if f.is_file()]
        rows.append([f"outputs/{d.name}/", str(len(files)), f"{sum(f.stat().st_size for f in files) / 2**20:.1f}",
                     OUTPUT_DESC.get(d.name, "Not identifiable from the current codebase.")])
    return [R.table(rows, [1.6, 0.5, 0.7, 4.0], mono_cols=(0,))]


def gen_ablation() -> list:
    path = ROOT / "outputs" / "ablation" / "results.json"
    if not path.exists():
        return [Paragraph("No ablation results found.", R.S["body"])]
    rows_in = json.loads(path.read_text())
    cols = [("label", "Configuration"), ("epoch", "Ep."), ("NDS", "NDS"), ("mAP", "mAP"), ("mAP_present", "mAP pres."),
            ("mATE", "mATE"), ("mASE", "mASE"), ("mAOE", "mAOE"), ("mAVE", "mAVE"), ("mAAE", "mAAE"), ("ADE", "ADE"),
            ("FDE", "FDE"), ("collision_rate", "Coll."), ("latency_ms", "ms"), ("params_M", "Params M")]
    rows = [[c[1] for c in cols]]
    for r in rows_in:
        rows.append([str(r.get(k)) if k in ("label", "epoch") else
                     ("–" if r.get(k) is None else f"{r[k]:.3f}" if k not in ("latency_ms", "params_M") else f"{r[k]:.1f}")
                     for k, _ in cols])
    flows = [R.table(rows, [2.6] + [0.55] * (len(cols) - 1))]
    md = (ROOT / "outputs" / "ablation" / "results.md").read_text(encoding="utf-8")
    hyp = md.split("## Hypothesis tests", 1)[1] if "## Hypothesis tests" in md else ""
    if hyp:
        lines = [l for l in hyp.splitlines() if l.startswith("|")]
        flows.append(Spacer(1, 4))
        flows.append(Paragraph("<b>Hypothesis deltas</b> (first minus second, from results.md):", R.S["body"]))
        flows += R.markdown("\n".join(lines))
    return flows


def gen_official() -> list:
    path = ROOT / "outputs" / "evaluation" / "joint" / "report.json"
    if not path.exists():
        return []
    rep = json.loads(path.read_text())
    off = rep.get("official_devkit")
    if not off:
        return []
    det = rep.get("detection", {})
    tp = off.get("tp_errors", {})
    text = (f"**Official devkit on the trained model F** (final weights, `scripts/evaluate.py --official`, mini-val, "
            f"detections exported with score ≥ 0.05): NDS {off['nd_score']:.4f}, mAP {off['mean_ap']:.4f}, "
            f"mATE {tp.get('trans_err', float('nan')):.3f}, mASE {tp.get('scale_err', float('nan')):.3f}, "
            f"mAOE {tp.get('orient_err', float('nan')):.3f}, mAVE {tp.get('vel_err', float('nan')):.3f}, "
            f"mAAE {tp.get('attr_err', float('nan')):.3f}. The in-memory evaluation of the same checkpoint at the training "
            f"threshold (score ≥ 0.1) gives NDS {det.get('NDS', float('nan')):.4f}, mAP {det.get('mAP', float('nan')):.4f}; "
            f"the difference comes from the lower export threshold (more low-confidence detections raise recall).")
    return R.markdown(text)


def gen_env() -> list:
    import cv2
    import matplotlib
    import numpy
    import reportlab
    import streamlit
    import torch
    import torchvision

    rows = [["Component", "Version"], ["Python", platform.python_version()], ["OS", platform.platform()],
            ["PyTorch", torch.__version__], ["CUDA (PyTorch build)", str(torch.version.cuda)],
            ["cuDNN", str(torch.backends.cudnn.version())], ["torchvision", torchvision.__version__],
            ["NumPy", numpy.__version__], ["OpenCV", cv2.__version__], ["matplotlib", matplotlib.__version__],
            ["Streamlit", streamlit.__version__], ["ReportLab (docs only)", reportlab.Version]]
    try:
        import nuscenes  # noqa: F401
        from importlib.metadata import version
        rows.append(["nuscenes-devkit", version("nuscenes-devkit")])
    except Exception:
        rows.append(["nuscenes-devkit", "not installed"])
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        rows.append(["GPU (nvidia-smi)", smi or "–"])
    except Exception:
        rows.append(["GPU (nvidia-smi)", "Not identifiable"])
    rows.append(["Virtual environment", sys.executable])
    return [R.table(rows, [1.5, 4])]


# ============================================================================ assembly
def cover(mods, n_files: int) -> list:
    n_cls = sum(len(m.classes) for m in mods.values())
    n_fn = sum(len(m.functions) + sum(len(c.methods) for c in m.classes) for m in mods.values())
    loc = sum(m.lines for m in mods.values())
    title = R.ParagraphStyle("ct", fontName="Body-Bold", fontSize=24, leading=30, textColor=R.colors.white)
    sub = R.ParagraphStyle("cs", fontName="Body", fontSize=11.5, leading=16, textColor=R.colors.HexColor("#dbe7f8"))
    flows = [Spacer(1, 12 * mm), Paragraph("Multi-Sensor Fusion &amp; Trajectory Planning on nuScenes", title),
             Spacer(1, 4 * mm), Paragraph("Technical Handover &amp; Developer Documentation", sub),
             Spacer(1, 4 * mm), Paragraph("Camera BEV · LiDAR · Radar · Learned Feature Selection · Fusion · 3-D Detection · "
                                          "Tracking · Planning · Streamlit Demo", sub),
             Spacer(1, 40 * mm)]
    rows = [["Item", "Value"],
            ["Project location", r"D:\researcpaper\multi_sensor_fusion"],
            ["Dataset location", r"D:\researcpaper  (nuScenes v1.0-mini)"],
            ["Python environment", r"C:\Users\shash\.venvs\msf_fusion"],
            ["Document generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Code base analysed", f"{len(mods)} Python modules · {loc:,} lines · {n_cls} classes · {n_fn} functions/methods · "
                                   f"{n_files} project files"],
            ["Generator", "docs/handover/build_handover.py (regenerate after code changes)"]]
    flows.append(R.table(rows, [1.4, 4]))
    flows.append(NextPageTemplate("normal"))
    flows.append(PageBreak())
    return flows


def main() -> None:
    print("analysing code ...")
    mods = analyze_project()
    print("rendering diagrams ...")
    diagrams.all_diagrams()
    cfg_eff, cfg_base = load_cfg()
    n_files = len(project_files())
    auto = {
        "tree": lambda: gen_tree(),
        "inventory": lambda: gen_inventory(mods),
        "package_graph": lambda: gen_package_graph(mods),
        "stages": lambda: gen_stages(cfg_base),
        "stage_losses": lambda: gen_stage_losses(cfg_base),
        "config": lambda: gen_config(cfg_eff, cfg_base),
        "hardware": lambda: gen_hardware(),
        "experiments": lambda: gen_experiments(),
        "cli": lambda: gen_cli(),
        "walkthroughs": lambda: gen_walkthroughs(),
        "api": lambda: gen_api(mods),
        "tests": lambda: gen_tests(mods),
        "checkpoints": lambda: gen_checkpoints(),
        "outputs": lambda: gen_outputs(),
        "ablation": lambda: gen_ablation(),
        "official": lambda: gen_official(),
        "env": lambda: gen_env(),
    }
    story = cover(mods, n_files)
    toc_heading = R.Heading("Table of Contents", 0)
    toc_heading.in_toc = False
    story += [toc_heading, R.toc(), PageBreak()]
    for title, text in CHAPTERS_A + CHAPTERS_B:
        story += R.heading(title, 0)[1:] if story and isinstance(story[-1], PageBreak) else R.heading(title, 0)
        chunks = re.split(r"^\{\{AUTO:(\w+)\}\}\s*$", text.strip("\n"), flags=re.M)
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                if chunk.strip():
                    story += R.markdown(chunk)
            else:
                print(f"  generating {chunk} ...")
                story += auto[chunk]()
    print("building PDF (two passes for the table of contents) ...")
    doc = R.HandoverDoc(OUT_PDF, TITLE)
    doc.multiBuild(story)
    from pypdf import PdfReader
    print(f"written {OUT_PDF} ({len(PdfReader(str(OUT_PDF)).pages)} pages)")
    if WARNINGS:
        print(f"{len(WARNINGS)} warnings:")
        for w in WARNINGS:
            print("  -", w)


if __name__ == "__main__":
    main()
