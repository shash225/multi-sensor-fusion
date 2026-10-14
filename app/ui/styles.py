"""CSS and small HTML components for the dashboard (dark "mission control" theme)."""
from __future__ import annotations

import html
from typing import Iterable, Optional

CSS = """
<style>
:root {
  --bg: #0d0f14; --panel: #151922; --panel-2: #1b2030; --line: #2a3142;
  --ink: #f2f4f8; --ink-2: #b8bfcc; --muted: #7d8596;
  --accent: #3987e5; --ok: #0ca30c; --warn: #fab219; --bad: #d03b3b; --pred: #00e5ff;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1600px; }
h1, h2, h3 { letter-spacing: 0.01em; }
.msf-title { display:flex; align-items:center; justify-content:space-between; gap:16px;
  background: linear-gradient(90deg, #121826, #0d0f14); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 18px; margin-bottom: 10px; }
.msf-title h1 { font-size: 1.35rem; margin: 0; color: var(--ink); font-weight: 700; }
.msf-title .sub { color: var(--ink-2); font-size: 0.85rem; }
.msf-banner { border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.06em; font-size: 0.9rem;
  display:inline-block; border: 1px solid; }
.msf-banner.gt { color: #ffd166; border-color: #8a6d1f; background: rgba(250,178,25,0.10); }
.msf-banner.model { color: #7ee787; border-color: #2f6f3a; background: rgba(12,163,12,0.12); }
.msf-banner.untrained { color: #ff8b8b; border-color: #8a2f2f; background: rgba(208,59,59,0.14); }
.msf-panel-title { color: var(--ink-2); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; margin: 2px 0 6px 0; }
.msf-flow { display:flex; flex-wrap: wrap; gap: 6px; align-items: stretch; margin: 6px 0 10px 0; }
.msf-step { flex: 1 1 120px; min-width: 112px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 8px 10px; color: var(--muted); font-size: 0.74rem; font-weight: 700;
  letter-spacing: 0.04em; transition: all 0.2s ease; }
.msf-step .t { display:block; font-weight: 500; font-size: 0.72rem; margin-top: 2px; color: var(--muted); letter-spacing: 0; }
.msf-step.done { color: var(--ink); border-color: #2f6f3a; background: rgba(12,163,12,0.10); }
.msf-step.done::before { content: "✓ "; color: #7ee787; }
.msf-step.active { color: var(--ink); border-color: var(--accent); background: rgba(57,135,229,0.18);
  box-shadow: 0 0 0 2px rgba(57,135,229,0.35); }
.msf-step.skip { opacity: 0.45; }
.msf-arrow { align-self:center; color: var(--muted); font-size: 0.9rem; }
.msf-status { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-top: 6px; }
.msf-kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; }
.msf-kpi .k { color: var(--muted); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.msf-kpi .v { color: var(--ink); font-size: 1.15rem; font-weight: 700; margin-top: 2px; }
.msf-kpi .n { color: var(--ink-2); font-size: 0.72rem; }
.msf-explain { background: var(--panel); border-left: 3px solid var(--accent); border-radius: 6px; padding: 10px 14px;
  color: var(--ink-2); font-size: 0.92rem; }
.msf-note { color: var(--muted); font-size: 0.78rem; }
.msf-chip { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
  border: 1px solid var(--line); color: var(--ink-2); margin-right: 4px; }
.msf-chip.ok { color: #7ee787; border-color: #2f6f3a; }
.msf-chip.bad { color: #ff8b8b; border-color: #8a2f2f; }
.msf-chip.warn { color: #ffd166; border-color: #8a6d1f; }
div[data-testid="stImage"] img { border-radius: 6px; }
</style>
"""

FLOW_STAGES = [
    ("camera", "CAMERA → BEV", ("camera_bev",)),
    ("sensors", "LiDAR + RADAR", ("lidar_encoder", "radar_encoder")),
    ("selection", "FEATURE SELECTION", ("feature_selection",)),
    ("fusion", "SENSOR FUSION", ("fusion",)),
    ("detection", "OBJECT DETECTION", ("bev_backbone", "detection_head", "decode", "tracking")),
    ("planning", "TRAJECTORY PLANNING", ("planning_head", "planning")),
    ("output", "DRIVING OUTPUT", ("render",)),
]


def flow_html(done: dict[str, float], active: Optional[str] = None, skipped: Iterable[str] = ()) -> str:
    """Pipeline strip; ``done`` maps stage key -> ms."""
    skipped = set(skipped)
    parts = ['<div class="msf-flow">']
    for i, (key, label, _) in enumerate(FLOW_STAGES):
        cls = "done" if key in done else "active" if key == active else "skip" if key in skipped else ""
        t = f"{done[key]:.0f} ms" if key in done else ("not used" if key in skipped else "waiting")
        parts.append(f'<div class="msf-step {cls}">{html.escape(label)}<span class="t">{t}</span></div>')
        if i < len(FLOW_STAGES) - 1:
            parts.append('<div class="msf-arrow">→</div>')
    parts.append("</div>")
    return "".join(parts)


def stage_key_for(timer_stage: str) -> Optional[str]:
    for key, _, names in FLOW_STAGES:
        if timer_stage in names:
            return key
    return None


def banner_html(kind: str, text: str) -> str:
    return f'<span class="msf-banner {kind}">{html.escape(text)}</span>'


def kpis_html(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="msf-kpi"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
                    f'<div class="n">{html.escape(n)}</div></div>' for k, v, n in items)
    return f'<div class="msf-status">{cells}</div>'


def panel_title(text: str) -> str:
    return f'<div class="msf-panel-title">{html.escape(text)}</div>'


def chip(text: str, kind: str = "") -> str:
    return f'<span class="msf-chip {kind}">{html.escape(text)}</span>'
