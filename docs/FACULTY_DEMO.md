# Faculty demonstration script (≈10 minutes)

## Before the presentation (once)

1. Train the proposed model and the comparison methods:
   `python main.py ablation --train --epochs 20` (GTX 1650: ≈3 h for the six experiments A–F).
2. Optional backup in case the laptop misbehaves: render a replay video
   `python main.py infer --scene scene-0103 --video` → `outputs/inference/scene-0103_model/scene-0103_model.mp4`.
3. Cache the road map once (first map decode takes ~5 s otherwise): `python main.py preprocess --maps`.
4. Close the training process — the dashboard needs the GPU.

## Launch

```
python app/streamlit_app.py
```
(the same as `streamlit run app/streamlit_app.py`; the browser opens at http://localhost:8501).

## Script

| Step | Do | Say |
|---|---|---|
| 1 | Sidebar: **🎓 Faculty Demo** on, mode **1 · Dataset Viewer**, scene **scene-0103 (val)**. | "This is a real nuScenes key-frame: six cameras, a 32-beam LiDAR and five radars, all synchronised and calibrated into one vehicle frame." Point at the 3D boxes projected into the camera and the radar velocity arrows in the BEV. |
| 2 | Mode **2 · Ground Truth Simulation**, press **▶ Play**. | "This replays the human annotations and the human driver's path (white dashed). The classical planner runs on the ground-truth objects — this is the upper bound for perception." |
| 3 | Mode **3 · AI Autonomous Driving**, method **6 · Camera + LiDAR + Radar + Feature Selection**, press **▶ Play**. | Walk along the pipeline strip as the stages tick: camera → BEV, LiDAR + radar, feature selection, fusion, detection, planning. The banner reads **MODEL PREDICTION**; green boxes in the camera are ground truth, coloured ones are predictions. |
| 4 | **⏸ Pause** on a busy frame, click **🔍 Explain Feature Selection**. | Read the explanation line. Show *raw → importance → selected* for LiDAR and radar, the sensor-dominance map (blue LiDAR / orange radar), the channel-gate chart (e.g. how strongly radar velocity is kept) and the per-class trust table. "Every number here comes from the network's own gates for this frame." |
| 5 | Panel 5 (trajectory). | Planned path (cyan) vs logged human path (white dashed), ADE / FDE, collision check against the real future positions of other agents. |
| 6 | Expander **⚖️ Compare fusion methods** → *Evaluate all trained methods on the whole scene*. | Same scene, six fusion designs: NDS / mAP / velocity error / ADE / collisions / latency side by side. |
| 7 | Expander **📊 Ablation study**. | The full validation-split comparison (Experiments A–F) and the hypothesis test table (feature selection vs concatenation and vs attention-only). |

## Honest-demo guarantees built into the app

* Without a trained checkpoint, mode 3 shows **GROUND TRUTH / DATASET VISUALIZATION** and says why.
* "Allow untrained network" shows a red **UNTRAINED** banner on every output.
* Training-split scenes are flagged; use **scene-0103** and **scene-0916** (nuScenes-mini val) to show
  generalisation.
* nuScenes-mini has 323 training samples; absolute numbers are far below full-nuScenes results. The
  point of the demo is the pipeline, the explainability and the *relative* comparison of fusion designs.
* Be ready for the question "does feature selection help?": on mini it did **not** beat concatenation
  (E vs C: −0.061 NDS), while radar (+0.056 NDS, −0.33 m/s velocity error) and the camera BEV (+0.039 NDS)
  clearly helped. The interpretability results (which channels / where each sensor is kept) stand on their
  own; testing the hypothesis properly needs full nuScenes and several seeds (docs/EXPERIMENTS.md).
