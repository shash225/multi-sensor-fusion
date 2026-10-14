"""Build the coordinator submission (ReadMe + Short Description PDFs, live-demo link file).

    C:\\Users\\shash\\.venvs\\msf_fusion\\Scripts\\python.exe docs/handover/build_submission.py --url https://<link>

Output folder: D:\\researcpaper\\Project_Submission (no source code is included).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image, NextPageTemplate, PageBreak, Paragraph, Spacer  # noqa: E402

import render as R  # noqa: E402

OUT = Path(r"D:\researcpaper\Project_Submission")
SHOTS = OUT / "04_Screenshots"
TITLE = "Multi-Sensor Fusion & Trajectory Planning for Autonomous Driving"
REPO = "https://github.com/shash225/multi-sensor-fusion (private)"


def cover(subtitle: str, rows: list[list[str]]) -> list:
    t = R.ParagraphStyle("ct", fontName="Body-Bold", fontSize=22, leading=28, textColor=R.colors.white)
    s = R.ParagraphStyle("cs", fontName="Body", fontSize=12, leading=17, textColor=R.colors.HexColor("#dbe7f8"))
    return [Spacer(1, 14 * mm), Paragraph(TITLE.replace("&", "&amp;"), t), Spacer(1, 4 * mm), Paragraph(subtitle, s),
            Spacer(1, 42 * mm), R.table([["Item", "Value"]] + rows, [1.3, 4]), NextPageTemplate("normal"), PageBreak()]


def shot(name: str, caption: str, width: float = R.FRAME_W) -> list:
    return R.figure(SHOTS / name, caption, max_height=150 * mm, width=width)


def readme_md(url: str) -> str:
    return f"""
## 1. What this is

A research prototype of the perception and planning software of a self-driving car. It reads the real
sensors of a test vehicle (6 cameras, 1 LiDAR, 5 radars) from the public **nuScenes** dataset, combines them
into one top-down map (Bird's-Eye View), lets a neural network decide which LiDAR and radar features to keep
(**learned feature selection**), fuses the sensors, detects 3-D objects (cars, pedestrians, trucks ...), tracks
them and plans the vehicle's path for the next 3 seconds. A web dashboard shows every step live.

## 2. How to test it — nothing to install

> **Open the live demo link in any web browser** (PC, laptop, tablet or phone). The link is in the file
> **03_Live_Demo_Link.txt** in this folder: `{url}`

The dashboard runs on the developer's computer (NVIDIA GTX 1650 GPU) and is shared through a secure
Cloudflare link. If the page does not open, the host computer is switched off — please contact the
developer for a new link. The first page load takes about 20–30 seconds (the model and data are loaded).

## 3. Suggested test (about 5 minutes)

1. **Demo mode → 1 · Dataset Viewer.** Real synchronised camera, LiDAR and radar data of one moment.
2. **Scene → scene-0103 or scene-0916** (marked *val*): scenes the AI never saw during training.
3. **Demo mode → 3 · AI Autonomous Driving**, **Fusion method → 6. Camera + LiDAR + Radar + Feature Selection**
   (the proposed system). Press **Play**. The banner reads **MODEL PREDICTION**; coloured boxes are the
   AI's detections, green boxes are the human-labelled ground truth, the cyan line is the planned path.
4. Watch the **pipeline strip** tick: Camera → BEV, LiDAR + Radar, Feature Selection, Fusion, Detection, Planning.
5. Press **Pause** and open **Explain Feature Selection**: which LiDAR / radar features the network kept.
6. **Weather → Dense fog** or **Heavy rain**: the same frame with simulated fog / rain on all sensors (marked
   SYNTHETIC). The number of detected objects drops — the robustness study in numbers is in the
   **Adverse-weather study** section.
7. **Compare fusion methods**: the six trained designs (LiDAR only ... full system) on the same scene.

## 4. Screen layout

| Panel | Shows |
|---|---|
| Top strip | Processing stages with timings; key numbers (objects, FPS, NDS score, mAP, ADE, collision) |
| Panel 1 · Camera | Front camera with predicted 3-D boxes (colour = class, number = confidence) and ground truth |
| Panel 2 · BEV | Top-down view: LiDAR (grey), radar with velocity (pink), detections with track IDs, planned path |
| Panel 3 · Sensor fusion | Camera, LiDAR and radar feature maps, the learned feature-selection gates, the fused map |
| Panel 5 · Planning | Ego speed, heading, navigation command; planned vs human-driven trajectory |

## 5. Results (nuScenes-mini, 81 validation frames)

| Result | Value |
|---|---|
| Proposed system, official nuScenes score | NDS 0.103 · mAP 0.046 |
| Adding radar to LiDAR | +0.056 NDS, velocity error −0.33 m/s |
| Adding the camera | +0.039 NDS, best mAP of all designs |
| Feature selection vs simple concatenation | −0.061 NDS (not better at this small data scale) |
| Learned planner vs constant velocity | path error 1.26–1.53 m vs 1.77 m |
| Dense fog (simulated) | −56 % NDS for the full system; radar unaffected |
| Speed | ≈ 100 ms per frame on a 4 GB laptop GPU |

Scores are low in absolute terms because nuScenes-mini has only 323 training frames (full nuScenes: 28 130);
the value is the fair comparison of six sensor-fusion designs trained identically.

## 6. Requirements and notes

* **For testing:** only a web browser and internet access.
* All data shown is real nuScenes data except the *Weather* options, which are simulated and clearly labelled.
* The dataset (nuScenes, © Motional, CC BY-NC-SA 4.0) and the trained model files are not distributed.
* Source code: {REPO}, available on request.
"""


def description_md() -> str:
    return """
## Multi-Sensor Fusion & Trajectory Planning for Autonomous Driving

**Type:** AI research software with an interactive web dashboard · **Domain:** autonomous vehicles, computer
vision, sensor fusion · **Data:** nuScenes (real driving data, Boston and Singapore)

**Problem.** A self-driving car must understand its surroundings reliably. Cameras see *what* objects are but not
how far away; LiDAR measures precise 3-D shape; radar measures speed and works in bad weather. Combining them
well — and knowing which sensor to trust where — is a key open problem.

**Solution.** The software converts all three sensors into one common top-down map (Bird's-Eye View) and adds a
small neural network that **learns which LiDAR and radar features to keep** before fusing them, guided by the
camera. The fused map is used to **detect 3-D objects** (10 classes, with speed), **track** them over time and
**plan** a safe 3-second path for the vehicle using either a classical planner (A* search + model-predictive
control) or a learned planner.

**Key features**

* End-to-end pipeline: 6 cameras + LiDAR + 5 radars → BEV fusion → 3-D detection → tracking → path planning.
* Explainable AI: the feature-selection gates show which sensor information the network relies on, and where.
* Controlled comparison of six sensor-fusion designs (ablation study) with the official nuScenes metrics.
* Adverse-weather testing: simulated fog and rain on all sensors to measure robustness.
* Interactive dashboard for live demonstration — usable from any browser through a shared link.
* Runs on a 4 GB laptop GPU (≈ 100 ms per frame).

**Outcome.** Radar clearly improves LiDAR (+0.056 NDS, better speed estimates) and the camera adds the best
detection accuracy; learned feature selection makes the fusion interpretable but did not beat simple fusion on the
small training set — a measured, honestly reported result that defines the next experiment on the full dataset.

**Technology:** Python, PyTorch, Lift-Splat-Shoot camera BEV, PointPillars, CenterPoint, iLQR planning,
Streamlit, nuScenes devkit metrics.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="current public dashboard link")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().strftime("%d %B %Y")

    story = cover("ReadMe — how to access and test the software",
                  [["Live demo", f"See 03_Live_Demo_Link.txt ({args.url})"], ["What you need", "A web browser — no installation"],
                   ["Date", today], ["Contents of this folder", "01 ReadMe · 02 Short Description · 03 Live Demo Link · 04 Screenshots"]])
    story += [R.Heading("ReadMe", 0)] + R.markdown(readme_md(args.url))
    story += R.heading("Screenshots", 0)
    story += shot("dashboard_ai_driving.png", "Dashboard, mode 3 · AI Autonomous Driving (live model predictions)", 150 * mm)
    story += shot("weather_fog_rain_sensor_inputs.png", "Weather option: real clear frame and SYNTHETIC fog / rain on camera, LiDAR and radar", 150 * mm)
    doc = R.HandoverDoc(OUT / "01_ReadMe.pdf", TITLE + " — ReadMe", footer="Project submission · ReadMe", toc_levels=0)
    doc.multiBuild(story)

    story = cover("Short description of the software", [["Software", "AI perception and planning system + web dashboard"],
                                                        ["Access", "Live demo link (03_Live_Demo_Link.txt)"], ["Date", today]])
    story += [R.Heading("Short Description", 0)] + R.markdown(description_md())
    story += shot("bev_all_sensors_and_feature_selection.jpg",
                  "Inside the system: camera, LiDAR and radar maps, the learned feature-selection gates, fused map and final detections")
    doc = R.HandoverDoc(OUT / "02_Short_Description.pdf", TITLE + " — Short Description",
                        footer="Project submission · Short description", toc_levels=0)
    doc.multiBuild(story)

    (OUT / "03_Live_Demo_Link.txt").write_text(
        f"{TITLE}\nLive demo dashboard (open in any web browser, no installation):\n\n    {args.url}\n\n"
        "Notes\n- Hosted on the developer's computer and shared through a Cloudflare link; if it does not open, the host\n"
        "  computer is offline - please contact the developer for the current link.\n"
        "- First load takes about 20-30 seconds. Suggested test steps: see 01_ReadMe.pdf, section 3.\n"
        f"- Link generated {today}.\n", encoding="utf-8")
    (OUT / "Live_Demo.url").write_text(f"[InternetShortcut]\nURL={args.url}\n", encoding="utf-8")
    from pypdf import PdfReader
    for f in ("01_ReadMe.pdf", "02_Short_Description.pdf"):
        print(f, len(PdfReader(str(OUT / f)).pages), "pages")


if __name__ == "__main__":
    main()
