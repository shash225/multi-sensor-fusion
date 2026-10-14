"""Model Predictive Control: iterative LQR (iLQR) on a kinematic bicycle model.

State ``s = [x, y, yaw, v, δ]`` (rear-axle position, heading, speed, steering angle),
control ``u = [a, ω]`` (acceleration, steering rate)::

    x'   = x + v cos(yaw) dt
    y'   = y + v sin(yaw) dt
    yaw' = yaw + v / L · tan(δ) dt
    v'   = v + a dt
    δ'   = δ + ω dt

Stage cost ``w_pos‖p − p_ref‖² + w_yaw Δyaw² + w_speed (v − v_ref)² + w_steer δ² + w_accel a² +
w_steer_rate ω²``. Controls and states are clamped to the vehicle limits after every
forward pass (projected iLQR), which is sufficient for this open-loop tracking problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.geometry import wrap_angle


@dataclass
class MPCResult:
    states: np.ndarray    # (N+1, 5)
    controls: np.ndarray  # (N, 2)
    cost: float
    iterations: int


class BicycleILQR:
    def __init__(self, cfg_mpc: Any, ego_cfg: Any) -> None:
        self.dt = float(cfg_mpc.dt)
        self.iterations = int(cfg_mpc.iterations)
        self.L = float(ego_cfg.wheelbase)
        self.max_steer = float(ego_cfg.max_steer)
        self.max_accel, self.max_decel = float(ego_cfg.max_accel), float(ego_cfg.max_decel)
        self.max_speed = float(ego_cfg.max_speed)
        self.Q = np.diag([cfg_mpc.w_pos, cfg_mpc.w_pos, cfg_mpc.w_yaw, cfg_mpc.w_speed, cfg_mpc.w_steer]).astype(float)
        self.R = np.diag([cfg_mpc.w_accel, cfg_mpc.w_steer_rate]).astype(float)

    # --------------------------------------------------------------- model
    def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        x, y, yaw, v, d = s
        a, w = u
        dt = self.dt
        nxt = np.array([x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt, yaw + v / self.L * np.tan(d) * dt,
                        v + a * dt, d + w * dt])
        nxt[3] = np.clip(nxt[3], 0.0, self.max_speed)
        nxt[4] = np.clip(nxt[4], -self.max_steer, self.max_steer)
        return nxt

    def jacobians(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, yaw, v, d = s
        dt = self.dt
        A = np.eye(5)
        A[0, 2], A[0, 3] = -v * np.sin(yaw) * dt, np.cos(yaw) * dt
        A[1, 2], A[1, 3] = v * np.cos(yaw) * dt, np.sin(yaw) * dt
        A[2, 3], A[2, 4] = np.tan(d) / self.L * dt, v / (self.L * np.cos(d) ** 2) * dt
        B = np.zeros((5, 2))
        B[3, 0], B[4, 1] = dt, dt
        return A, B

    def rollout(self, s0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = [s0]
        for u in controls:
            states.append(self.step(states[-1], u))
        return np.asarray(states)

    def _clip(self, u: np.ndarray) -> np.ndarray:
        u = u.copy()
        u[:, 0] = np.clip(u[:, 0], -self.max_decel, self.max_accel)
        u[:, 1] = np.clip(u[:, 1], -1.0, 1.0)
        return u

    def _err(self, s: np.ndarray, ref: np.ndarray) -> np.ndarray:
        e = s - ref
        e[..., 2] = wrap_angle(e[..., 2])
        return e

    def cost(self, states: np.ndarray, controls: np.ndarray, ref: np.ndarray) -> float:
        e = self._err(states[1:], ref)
        return float(np.einsum("ni,ij,nj->", e, self.Q, e) + np.einsum("ni,ij,nj->", controls, self.R, controls))

    # --------------------------------------------------------------- solve
    def solve(self, s0: np.ndarray, ref: np.ndarray, u_init: np.ndarray | None = None) -> MPCResult:
        """Track ``ref (N, 5)`` (``[x, y, yaw, v, 0]`` per step) from initial state ``s0``."""
        n = ref.shape[0]
        controls = np.zeros((n, 2)) if u_init is None else self._clip(u_init)
        states = self.rollout(s0, controls)
        cost = self.cost(states, controls, ref)
        it = 0
        for it in range(1, self.iterations + 1):
            # Backward pass (Riccati recursion on the quadratic cost-to-go).
            Vx, Vxx = np.zeros(5), np.zeros((5, 5))
            ks, Ks = np.zeros((n, 2)), np.zeros((n, 2, 5))
            for t in reversed(range(n)):
                A, B = self.jacobians(states[t])
                e = self._err(states[t + 1], ref[t])
                lx = 2 * self.Q @ e + Vx
                lxx = 2 * self.Q + Vxx
                Qx = A.T @ lx
                Qu = B.T @ lx + 2 * self.R @ controls[t]
                Qxx = A.T @ lxx @ A
                Quu = B.T @ lxx @ B + 2 * self.R + 1e-6 * np.eye(2)
                Qux = B.T @ lxx @ A
                Quu_inv = np.linalg.inv(Quu)
                ks[t] = -Quu_inv @ Qu
                Ks[t] = -Quu_inv @ Qux
                Vx = Qx + Ks[t].T @ Quu @ ks[t] + Ks[t].T @ Qu + Qux.T @ ks[t]
                Vxx = Qxx + Ks[t].T @ Quu @ Ks[t] + Ks[t].T @ Qux + Qux.T @ Ks[t]
                Vxx = 0.5 * (Vxx + Vxx.T)
            # Forward pass with backtracking line search.
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1):
                new_u = np.zeros_like(controls)
                new_s = [s0]
                for t in range(n):
                    dx = self._err(new_s[-1], states[t])
                    new_u[t] = controls[t] + alpha * ks[t] + Ks[t] @ dx
                    new_u[t] = self._clip(new_u[t:t + 1])[0]
                    new_s.append(self.step(new_s[-1], new_u[t]))
                new_s = np.asarray(new_s)
                new_cost = self.cost(new_s, new_u, ref)
                if new_cost < cost:
                    states, controls, improved = new_s, new_u, True
                    converged = cost - new_cost < 1e-4 * max(cost, 1.0)
                    cost = new_cost
                    break
            if not improved or converged:
                break
        return MPCResult(states, controls, cost, it)
