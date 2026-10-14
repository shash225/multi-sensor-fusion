"""nuScenes detection metrics (mAP, mATE, mASE, mAOE, mAVE, mAAE, NDS) — in-memory port.

This reproduces the official ``nuscenes.eval.detection`` algorithm (center-distance
matching at 0.5/1/2/4 m, 101-point interpolated precision, min recall/precision 0.1, TP
errors at the 2 m threshold with confidence-aligned cumulative means, class-specific
ranges and NaN rules), so numbers are comparable without writing result files. Use
:mod:`evaluation.official_eval` to cross-check with the devkit itself.

Note (official behaviour): a class with **no GT** in the evaluated split gets AP = 0 and
TP errors = 1. On ``mini_val`` (2 scenes) several classes are absent, which caps the
achievable mAP/NDS; ``mAP_present_classes`` is reported additionally for context.

Boxes are ``[x, y, z, l, w, h, yaw, vx, vy]`` in the ego/reference frame (all nuScenes
metrics are invariant to the rigid ego->global transform).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

from utils.geometry import points_in_boxes

CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50, "construction_vehicle": 50, "pedestrian": 40,
               "motorcycle": 40, "bicycle": 40, "traffic_cone": 30, "barrier": 30}
DIST_THS = (0.5, 1.0, 2.0, 4.0)
DIST_TH_TP = 2.0
MIN_RECALL, MIN_PRECISION = 0.1, 0.1
MAX_BOXES_PER_SAMPLE = 500
TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")
NELEM = 101


def default_attribute(name: str, velocity: np.ndarray) -> str:
    """Attribute heuristic used by CenterPoint-style detectors (no attribute head)."""
    moving = float(np.hypot(*velocity[:2])) > 0.2
    if name in ("car", "truck", "bus", "trailer", "construction_vehicle"):
        return "vehicle.moving" if moving else "vehicle.parked"
    if name == "pedestrian":
        return "pedestrian.moving" if moving else "pedestrian.standing"
    if name in ("motorcycle", "bicycle"):
        return "cycle.with_rider" if moving else "cycle.without_rider"
    return ""


def _angle_diff(x: float, y: float, period: float) -> float:
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff -= 2 * np.pi
    return abs(diff)


def _scale_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.prod(np.minimum(a, b))
    return float(inter / (np.prod(a) + np.prod(b) - inter))


def _cummean(x: np.ndarray) -> np.ndarray:
    if np.isnan(x).all():
        return np.ones(len(x))
    s = np.nancumsum(x.astype(float))
    n = np.cumsum(~np.isnan(x))
    return np.divide(s, n, out=np.zeros_like(s), where=n != 0)


class DetectionEvaluator:
    """Accumulate predictions/GT sample by sample, then call :meth:`evaluate`."""

    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.gt: dict[str, list[dict]] = defaultdict(list)     # class -> list of gt dicts
        self.pred: dict[str, list[dict]] = defaultdict(list)
        self.samples: list[str] = []

    def add_sample(self, token: str, pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_labels: np.ndarray,
                   gt_boxes: np.ndarray, gt_labels: np.ndarray, gt_attributes: Optional[Sequence[str]] = None,
                   gt_num_pts: Optional[np.ndarray] = None, gt_velocity_valid: Optional[np.ndarray] = None,
                   pred_attributes: Optional[Sequence[str]] = None, bike_racks: Optional[np.ndarray] = None) -> None:
        self.samples.append(token)
        racks = bike_racks if bike_racks is not None and len(bike_racks) else None

        def in_rack(name: str, box: np.ndarray) -> bool:
            return racks is not None and name in ("bicycle", "motorcycle") and bool(
                points_in_boxes(box[None, :3].astype(np.float64), racks.astype(np.float64)).any())

        order = np.argsort(-pred_scores)[:MAX_BOXES_PER_SAMPLE]
        for i in order:
            name = self.classes[int(pred_labels[i])]
            box = pred_boxes[i]
            if np.hypot(box[0], box[1]) > CLASS_RANGE[name] or in_rack(name, box):
                continue
            attr = pred_attributes[i] if pred_attributes is not None else default_attribute(name, box[7:9])
            self.pred[name].append({"token": token, "box": box.astype(np.float64), "score": float(pred_scores[i]), "attr": attr})
        for j in range(len(gt_labels)):
            name = self.classes[int(gt_labels[j])]
            box = gt_boxes[j].astype(np.float64).copy()
            if np.hypot(box[0], box[1]) > CLASS_RANGE[name]:
                continue
            if (gt_num_pts is not None and gt_num_pts[j] == 0) or in_rack(name, box):
                continue
            if gt_velocity_valid is not None and not gt_velocity_valid[j]:
                box[7:9] = np.nan
            self.gt[name].append({"token": token, "box": box, "attr": gt_attributes[j] if gt_attributes is not None else ""})

    # ------------------------------------------------------------ accumulate
    def _accumulate(self, name: str, dist_th: float) -> dict[str, np.ndarray]:
        gts = self.gt[name]
        npos = len(gts)
        empty = {"precision": np.zeros(NELEM), "confidence": np.zeros(NELEM),
                 **{m: np.ones(NELEM) for m in TP_METRICS}}
        if npos == 0:
            return empty
        by_token: dict[str, list[int]] = defaultdict(list)
        for k, g in enumerate(gts):
            by_token[g["token"]].append(k)
        preds = sorted(self.pred[name], key=lambda p: -p["score"])
        taken = np.zeros(npos, dtype=bool)
        tp, fp, conf = [], [], []
        md = {m: [] for m in TP_METRICS}
        md_conf = []
        period = np.pi if name == "barrier" else 2 * np.pi
        for p in preds:
            best, best_d = None, np.inf
            for k in by_token.get(p["token"], ()):
                if taken[k]:
                    continue
                d = float(np.hypot(*(gts[k]["box"][:2] - p["box"][:2])))
                if d < best_d:
                    best, best_d = k, d
            if best is not None and best_d < dist_th:
                taken[best] = True
                tp.append(1)
                fp.append(0)
                g, b = gts[best]["box"], p["box"]
                md["trans_err"].append(best_d)
                md["vel_err"].append(float(np.hypot(*(g[7:9] - b[7:9]))))
                # nuScenes sizes are [w, l, h]; the IoU is symmetric in the order, only alignment matters.
                md["scale_err"].append(1.0 - _scale_iou(g[3:6], b[3:6]))
                md["orient_err"].append(_angle_diff(g[6], b[6], period))
                ga = gts[best]["attr"]
                md["attr_err"].append(np.nan if ga == "" else float(ga != p["attr"]))
                md_conf.append(p["score"])
            else:
                tp.append(0)
                fp.append(1)
            conf.append(p["score"])
        if sum(tp) == 0:
            return empty
        tp_c, fp_c = np.cumsum(tp).astype(float), np.cumsum(fp).astype(float)
        prec = tp_c / (fp_c + tp_c)
        rec = tp_c / float(npos)
        rec_interp = np.linspace(0, 1, NELEM)
        prec_i = np.interp(rec_interp, rec, prec, right=0)
        conf_i = np.interp(rec_interp, rec, np.asarray(conf), right=0)
        out = {"precision": prec_i, "confidence": conf_i}
        md_conf_a = np.asarray(md_conf)
        for m in TP_METRICS:
            tmp = _cummean(np.asarray(md[m], dtype=float))
            out[m] = np.interp(conf_i[::-1], md_conf_a[::-1], tmp[::-1])[::-1]
        return out

    @staticmethod
    def _calc_ap(md: dict[str, np.ndarray]) -> float:
        prec = md["precision"][round(100 * MIN_RECALL) + 1:].copy()
        prec -= MIN_PRECISION
        prec[prec < 0] = 0
        return float(np.mean(prec)) / (1.0 - MIN_PRECISION)

    @staticmethod
    def _calc_tp(md: dict[str, np.ndarray], metric: str) -> float:
        first = round(100 * MIN_RECALL) + 1
        nz = np.nonzero(md["confidence"])[0]
        last = int(nz[-1]) if len(nz) else 0
        if last < first:
            return 1.0
        return float(np.mean(md[metric][first:last + 1]))

    # ---------------------------------------------------- precision / recall
    def precision_recall(self, dist_th: float = DIST_TH_TP, score_th: float = 0.3) -> dict[str, float]:
        """Operating-point counts: detections with score ≥ ``score_th`` matched greedily (highest score
        first, same class, same sample) to GT within ``dist_th`` metres centre distance.

        Not part of NDS/mAP; reported for the adverse-weather study (missed detections = FN,
        incorrect detections = FP).
        """
        tp = fp = fn = 0
        loc_err: list[float] = []
        for name in self.classes:
            gts = self.gt[name]
            by_token: dict[str, list[int]] = defaultdict(list)
            for k, g in enumerate(gts):
                by_token[g["token"]].append(k)
            taken = np.zeros(len(gts), dtype=bool)
            for p in sorted((p for p in self.pred[name] if p["score"] >= score_th), key=lambda p: -p["score"]):
                best, best_d = None, dist_th
                for k in by_token.get(p["token"], ()):
                    if not taken[k]:
                        d = float(np.hypot(*(gts[k]["box"][:2] - p["box"][:2])))
                        if d < best_d:
                            best, best_d = k, d
                if best is None:
                    fp += 1
                else:
                    taken[best] = True
                    tp += 1
                    loc_err.append(best_d)
            fn += int((~taken).sum())
        return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                "tp": tp, "fp": fp, "fn": fn, "mean_center_error_m": float(np.mean(loc_err)) if loc_err else float("nan"),
                "dist_th": dist_th, "score_th": score_th}

    # -------------------------------------------------------------- evaluate
    def evaluate(self) -> dict[str, object]:
        per_class: dict[str, dict[str, float]] = {}
        for name in self.classes:
            aps = {th: self._calc_ap(self._accumulate(name, th)) for th in DIST_THS}
            md_tp = self._accumulate(name, DIST_TH_TP)
            res = {"AP": float(np.mean(list(aps.values()))), **{f"AP@{th}": v for th, v in aps.items()},
                   "num_gt": len(self.gt[name]), "num_pred": len(self.pred[name])}
            for m in TP_METRICS:
                if (name == "traffic_cone" and m in ("attr_err", "vel_err", "orient_err")) or \
                        (name == "barrier" and m in ("attr_err", "vel_err")):
                    res[m] = float("nan")
                else:
                    res[m] = self._calc_tp(md_tp, m)
            per_class[name] = res
        m_ap = float(np.mean([per_class[c]["AP"] for c in self.classes]))
        tp_errors = {m: float(np.nanmean([per_class[c][m] for c in self.classes])) for m in TP_METRICS}
        tp_scores = {m: max(0.0, 1.0 - tp_errors[m]) for m in TP_METRICS}
        nds = (5 * m_ap + sum(tp_scores.values())) / (5 + len(TP_METRICS))
        present = [c for c in self.classes if per_class[c]["num_gt"] > 0]
        return {
            "NDS": nds, "mAP": m_ap,
            "mATE": tp_errors["trans_err"], "mASE": tp_errors["scale_err"], "mAOE": tp_errors["orient_err"],
            "mAVE": tp_errors["vel_err"], "mAAE": tp_errors["attr_err"],
            "mAP_present_classes": float(np.mean([per_class[c]["AP"] for c in present])) if present else 0.0,
            "num_samples": len(self.samples), "per_class": per_class,
            "PR@2m": self.precision_recall(DIST_TH_TP, 0.3),
        }
