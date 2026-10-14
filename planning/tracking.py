"""Online multi-object tracking for scene replay (CenterPoint-style greedy association).

Detections of each key-frame are moved to the global frame; existing tracks are
predicted with their velocity, and detections are greedily matched (highest score
first) to the nearest compatible track within a class-specific distance. Tracks give
persistent IDs, smoothed velocities and motion histories for the visualisation and the
planner's motion forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.geometry import invert_transform

# Max centre distance (m) between a predicted track and a detection (CenterPoint nuScenes values).
MATCH_DISTANCE = {"car": 4.0, "truck": 4.0, "bus": 5.5, "trailer": 3.0, "construction_vehicle": 3.0,
                  "pedestrian": 1.0, "motorcycle": 13.0, "bicycle": 3.0, "barrier": 1.0, "traffic_cone": 1.0}


@dataclass
class Track:
    track_id: int
    label: int
    position: np.ndarray            # global xy
    velocity: np.ndarray            # global vxy
    score: float
    age: int = 1
    misses: int = 0
    history: list = field(default_factory=list)


class MultiObjectTracker:
    def __init__(self, classes: list[str], max_misses: int = 2, history: int = 8) -> None:
        self.classes = classes
        self.max_misses = max_misses
        self.history_len = history
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_time: Optional[float] = None

    def update(self, boxes_ref: np.ndarray, labels: np.ndarray, scores: np.ndarray, ref_ego2global: np.ndarray,
               timestamp_us: int) -> np.ndarray:
        """Associate detections; returns track ids ``(N,)`` aligned with the detections."""
        t = timestamp_us * 1e-6
        dt = 0.5 if self.last_time is None else max(t - self.last_time, 1e-3)
        if self.last_time is not None and (t - self.last_time > 2.0 or t < self.last_time):
            self.reset()  # jumped in the replay: start fresh
            dt = 0.5
        self.last_time = t
        rot = ref_ego2global[:2, :2]
        pos = boxes_ref[:, :2] @ rot.T + ref_ego2global[:2, 3] if len(boxes_ref) else np.zeros((0, 2))
        vel = boxes_ref[:, 7:9] @ rot.T if len(boxes_ref) else np.zeros((0, 2))
        predicted = [tr.position + tr.velocity * dt for tr in self.tracks]
        ids = np.zeros(len(boxes_ref), dtype=np.int64)
        used = set()
        for i in np.argsort(-scores):
            name = self.classes[int(labels[i])]
            best, best_d = None, MATCH_DISTANCE.get(name, 2.0)
            for j, (tr, p) in enumerate(zip(self.tracks, predicted)):
                if j in used or tr.label != int(labels[i]):
                    continue
                d = float(np.hypot(*(pos[i] - p)))
                if d < best_d:
                    best, best_d = j, d
            if best is None:
                tr = Track(self.next_id, int(labels[i]), pos[i].copy(), vel[i].copy(), float(scores[i]))
                self.next_id += 1
                self.tracks.append(tr)
                used.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[best]
                measured = (pos[i] - tr.position) / dt
                tr.velocity = 0.5 * vel[i] + 0.5 * measured
                tr.position, tr.score, tr.age, tr.misses = pos[i].copy(), float(scores[i]), tr.age + 1, 0
                used.add(best)
            tr.history = (tr.history + [tr.position.copy()])[-self.history_len:]
            ids[i] = tr.track_id
        for j, tr in enumerate(self.tracks):
            if j not in used:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return ids

    def histories_ref(self, ids: np.ndarray, ref_ego2global: np.ndarray) -> list[np.ndarray]:
        """Motion history of the given track ids in the current reference frame."""
        g2r = invert_transform(ref_ego2global)
        by_id = {tr.track_id: tr for tr in self.tracks}
        out = []
        for tid in ids:
            tr = by_id.get(int(tid))
            if tr is None or not tr.history:
                out.append(np.zeros((0, 2)))
                continue
            h = np.asarray(tr.history)
            out.append(h @ g2r[:2, :2].T + g2r[:2, 3])
        return out
