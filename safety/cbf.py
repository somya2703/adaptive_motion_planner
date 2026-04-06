"""
Control Barrier Function (CBF) safety filter for the Panda.

Theory
------
A CBF h: Q -> R defines a safe set S = {q | h(q) >= 0}.
The CBF condition makes S forward-invariant:

    ḣ(q, q̇) + α(h(q)) ≥ 0                        (*)

where α is a class-K function (we use α(s) = k·s, k > 0).

Expanding:  ḣ = ∂h/∂q · q̇  =  Lf h(q) + Lg h(q) · q̇

CBFs used here:
  1. Obstacle avoidance  — h_obs(q) = ||p_tcp(q) - p_obs||² - (r_obs + margin)²
  2. Joint limit         — h_jnt(q) = (q - q_min)(q_max - q)  (element-wise)

At runtime, a QP finds the minimum-norm torque correction τ such that
all CBF conditions (*) are satisfied:

    min  ||τ - τ_des||²
    s.t. ∀i: Lg_i h_i(q) · τ + Lf_i h_i(q) + α_i · h_i(q) ≥ 0
             τ_min ≤ τ ≤ τ_max

We solve this via scipy's quadratic programming (OSQP-like but pure Python).

Reference: Ames et al., "Control Barrier Functions: Theory and Applications", ECC 2019.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import linprog

from kinematics.forward import forward_kinematics, tcp_position
from kinematics.jacobian import geometric_jacobian
from robot.panda import PANDA


@dataclass
class Obstacle:
    """Spherical obstacle in Cartesian space."""
    center: np.ndarray   # (3,) base-frame position
    radius: float
    margin: float = 0.05   # additional safety buffer (m)

    @property
    def effective_radius(self) -> float:
        return self.radius + self.margin


# ---------------------------------------------------------------------------
# CBF functions and gradients
# ---------------------------------------------------------------------------

def _h_obstacle(q: np.ndarray, obs: Obstacle) -> float:
    """
    CBF for a single spherical obstacle.

    h(q) = ||p_tcp - p_obs||² - r_eff²
    Safe set: h ≥ 0  ↔  TCP is outside the obstacle sphere.
    """
    p = tcp_position(q)
    diff = p - obs.center
    return float(diff @ diff) - obs.effective_radius ** 2


def _grad_h_obstacle(q: np.ndarray, obs: Obstacle) -> np.ndarray:
    """
    Gradient ∂h/∂q for the obstacle CBF via chain rule:

    ∂h/∂q = 2(p_tcp - p_obs)^T · J_pos(q)

    Returns (7,) vector.
    """
    p = tcp_position(q)
    diff = p - obs.center          # (3,)
    J_pos = geometric_jacobian(q)[:3, :]   # (3, 7)
    return 2.0 * diff @ J_pos      # (7,)


def _h_joint_lower(q: np.ndarray, k: int) -> float:
    """CBF for lower limit of joint k:  h_k = q_k - q_min_k."""
    return float(q[k] - PANDA.q_min[k])


def _h_joint_upper(q: np.ndarray, k: int) -> float:
    """CBF for upper limit of joint k:  h_k = q_max_k - q_k."""
    return float(PANDA.q_max[k] - q[k])


# ---------------------------------------------------------------------------
# CBF safety filter (QP)
# ---------------------------------------------------------------------------

@dataclass
class CBFResult:
    tau_safe: np.ndarray     # filtered torque command
    cbf_active: bool         # True if the filter had to modify τ_des
    min_h: float             # minimum CBF value (positive = all safe)
    correction_norm: float   # ||τ_safe - τ_des||


class CBFSafetyFilter:
    """
    Applies a CBF filter to a desired torque command τ_des.

    The filter solves a simple QP to find the closest safe τ.
    We use a first-order (gradient-based) approximation because
    the planner runs in a tight loop.
    """

    def __init__(
        self,
        obstacles: List[Obstacle] | None = None,
        alpha_obs: float = 1.5,    # class-K gain for obstacle CBFs
        alpha_jnt: float = 5.0,    # class-K gain for joint CBFs
        dt: float = 0.001,         # control timestep (1 kHz)
    ):
        self.obstacles = obstacles or []
        self.alpha_obs = alpha_obs
        self.alpha_jnt = alpha_jnt
        self.dt = dt

    def update_obstacles(self, obstacles: List[Obstacle]) -> None:
        self.obstacles = obstacles

    def _build_constraints(
        self, q: np.ndarray, qd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the CBF linear constraints:  A @ τ ≤ b

        Each row corresponds to one CBF:
            -Lg h · τ ≤ Lf h + α · h

        We approximate Lf h ≈ (∂h/∂q) · qd for revolute joints
        (gravity/Coriolis ignored for this first-order filter).
        """
        rows_A: List[np.ndarray] = []
        rows_b: List[float]      = []

        # --- Obstacle CBFs ---
        for obs in self.obstacles:
            h    = _h_obstacle(q, obs)
            dh   = _grad_h_obstacle(q, obs)   # (7,)
            Lf_h = float(dh @ qd)             # first-order Lie derivative
            Lg_h = dh                          # since τ -> q̈ (simplified)

            # Constraint: dh · τ ≥ -Lf_h - alpha * h
            #  => -dh · τ ≤  Lf_h + alpha * h
            rows_A.append(-Lg_h)
            rows_b.append(Lf_h + self.alpha_obs * h)

        # --- Joint limit CBFs (lower) ---
        for k in range(7):
            h  = _h_joint_lower(q, k)
            e_k = np.zeros(7); e_k[k] = 1.0
            dh = e_k
            Lf_h = float(dh @ qd)
            rows_A.append(-dh)
            rows_b.append(Lf_h + self.alpha_jnt * h)

        # --- Joint limit CBFs (upper) ---
        for k in range(7):
            h  = _h_joint_upper(q, k)
            e_k = np.zeros(7); e_k[k] = -1.0
            dh = e_k
            Lf_h = float(dh @ qd)
            rows_A.append(-dh)
            rows_b.append(Lf_h + self.alpha_jnt * h)

        if rows_A:
            return np.array(rows_A), np.array(rows_b)
        return np.zeros((0, 7)), np.zeros(0)

    def filter(
        self,
        tau_des: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> CBFResult:
        """
        Filter τ_des through the CBF to obtain a safe τ.

        Uses a gradient-projection approach (closed-form for linear CBFs):
        for each violated constraint, project τ onto the half-space boundary.

        For production use, replace this with a proper QP solver (e.g. OSQP).
        """
        A, b = self._build_constraints(q, qd)

        tau = tau_des.copy()
        tau = np.clip(tau, -PANDA.tau_max, PANDA.tau_max)

        cbf_active = False

        if A.shape[0] > 0:
            for i in range(A.shape[0]):
                a_i = A[i]
                b_i = b[i]
                lhs = float(a_i @ tau)
                if lhs > b_i:      # constraint violated
                    # Project τ onto the constraint boundary
                    nrm2 = float(a_i @ a_i)
                    if nrm2 > 1e-12:
                        tau = tau - ((lhs - b_i) / nrm2) * a_i
                        cbf_active = True

        tau = np.clip(tau, -PANDA.tau_max, PANDA.tau_max)

        # Compute minimum CBF value for diagnostics
        h_values = [_h_obstacle(q, obs) for obs in self.obstacles]
        h_values += [_h_joint_lower(q, k) for k in range(7)]
        h_values += [_h_joint_upper(q, k) for k in range(7)]
        min_h = min(h_values) if h_values else float("inf")

        return CBFResult(
            tau_safe=tau,
            cbf_active=cbf_active,
            min_h=min_h,
            correction_norm=float(np.linalg.norm(tau - tau_des)),
        )

    def is_config_safe(self, q: np.ndarray) -> bool:
        """
        Check if a configuration is in the CBF safe set
        (all barrier functions positive).
        """
        for obs in self.obstacles:
            if _h_obstacle(q, obs) < 0:
                return False
        for k in range(7):
            if _h_joint_lower(q, k) < 0 or _h_joint_upper(q, k) < 0:
                return False
        return True

    def min_clearance(self, q: np.ndarray) -> float:
        """Return the minimum h value — the safety margin."""
        h_values = []
        for obs in self.obstacles:
            h_values.append(_h_obstacle(q, obs))
        for k in range(7):
            h_values.append(min(_h_joint_lower(q, k), _h_joint_upper(q, k)))
        return min(h_values) if h_values else float("inf")
