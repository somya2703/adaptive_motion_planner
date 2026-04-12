"""
Used damped least-squares with nullspace.

Given a desired TCP pose T_des (position + rotation), find joint angles q
such that FK(q) ≈ T_des, while:

  1. Respecting joint position limits via nullspace gradient descent
  2. Avoiding singularities via variable damping (based on manipulability)
  3. Exploiting redundancy (7-DoF) to pull joints toward their midpoints

Algorithm (iterative, first-order):
    Δx_e  = error in TCP pose (6D: position + axis-angle rotation error)
    J     = geometric_jacobian(q)
    λ     = adaptive_damping(manipulability(q))
    Δq_task  = J^†(λ) · Δx_e               task-space step
    Δq_null  = (I - J^†J) · k · ∇q_mid    nullspace gradient
    q    += Δq_task + Δq_null
    q     = clamp(q, q_min, q_max)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from kinematics.forward import forward_kinematics, tcp_position
from kinematics.jacobian import (
    geometric_jacobian,
    damped_pseudoinverse,
    nullspace_projector,
    manipulability,
)
from robot.panda import PANDA


@dataclass
class IKResult:
    q: np.ndarray           # (7,) solution joint angles
    success: bool
    error_pos: float        # final TCP position error (m)
    error_rot: float        # final TCP rotation error (rad)
    iterations: int


def _rotation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """
    Compute the 3D rotation error between desired and current rotation matrices.

    Uses the axis-angle representation of R_err = R_des @ R_cur^T.
    Returns a 3-vector proportional to the rotation axis and angle.
    """
    R_err = R_des @ R_cur.T
    # Rodrigues formula: sin(θ)·n = skew_part(R_err)
    skew = (R_err - R_err.T) / 2.0
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def _adaptive_damping(manip: float, manip_threshold: float = 0.04) -> float:
    """
    Increase damping near singularities (low manipulability).

    Below the threshold, damping grows quadratically to smoothly
    handle the loss of rank without numerical blow-up.
    """
    lambda_max = 0.2
    if manip >= manip_threshold:
        return 0.001  # low damping far from singularity
    ratio = 1.0 - (manip / manip_threshold)
    return lambda_max * ratio ** 2


def _nullspace_gradient(q: np.ndarray, k: float = 0.5) -> np.ndarray:
    """
    Gradient of the joint midpoint cost function:
        H(q) = 0.5 * sum((q_i - q_mid_i) / (q_max_i - q_min_i))^2

    The nullspace step will pull joints toward their midpoints,
    keeping the arm away from joint limits during redundancy resolution.
    """
    q_range = PANDA.q_max - PANDA.q_min
    return -k * (q - PANDA.q_mid) / (q_range ** 2)


def solve_ik(
    T_desired: np.ndarray,
    q_init: np.ndarray | None = None,
    max_iter: int = 200,
    tol_pos: float = 1e-4,   # 0.1 mm
    tol_rot: float = 1e-3,   # ~0.057 deg
    step_size: float = 0.5,
    nullspace_gain: float = 0.3,
) -> IKResult:
    """
    Iterative damped least-squares IK solver.

    Parameters
    ----------
    T_desired      : (4, 4) SE(3) desired TCP pose
    q_init         : (7,) initial guess; random if None
    max_iter       : iteration limit
    tol_pos        : convergence tolerance, position (metres)
    tol_rot        : convergence tolerance, rotation (radians)
    step_size      : task-space step scale α
    nullspace_gain : weight of nullspace gradient step

    Returns
    -------
    IKResult with solution and diagnostics
    """
    p_des = T_desired[:3, 3]
    R_des = T_desired[:3, :3]

    if q_init is None:
        q_init = PANDA.random_config()

    q = q_init.copy()

    for iteration in range(max_iter):
        T_cur, _ = forward_kinematics(q)
        p_cur = T_cur[:3, 3]
        R_cur = T_cur[:3, :3]

        # ---- Compute 6D error -------------------------------------------
        e_pos = p_des - p_cur                          # (3,)
        e_rot = _rotation_error(R_des, R_cur)          # (3,)
        e = np.concatenate([e_pos, e_rot])             # (6,)

        err_pos = float(np.linalg.norm(e_pos))
        err_rot = float(np.linalg.norm(e_rot))

        if err_pos < tol_pos and err_rot < tol_rot:
            return IKResult(q, True, err_pos, err_rot, iteration)

        # ---- Compute Jacobian and pseudoinverse --------------------------
        J = geometric_jacobian(q)
        manip = manipulability(q)
        lam = _adaptive_damping(manip)
        J_pinv = damped_pseudoinverse(J, lam)

        # ---- Task-space step --------------------------------------------
        dq_task = step_size * J_pinv @ e

        # ---- Nullspace step (redundancy for joint-limit avoidance) ------
        P = nullspace_projector(J, lam)
        grad = _nullspace_gradient(q, nullspace_gain)
        dq_null = P @ grad

        # ---- Update and clamp -------------------------------------------
        q = q + dq_task + dq_null
        q = PANDA.clamp(q)

    # Did not converge — return best attempt
    T_cur, _ = forward_kinematics(q)
    err_pos = float(np.linalg.norm(T_desired[:3, 3] - T_cur[:3, 3]))
    err_rot = float(np.linalg.norm(_rotation_error(R_des, T_cur[:3, :3])))
    return IKResult(q, False, err_pos, err_rot, max_iter)


def solve_ik_position_only(
    p_desired: np.ndarray,
    q_init: np.ndarray | None = None,
    max_iter: int = 150,
    tol: float = 1e-4,
) -> IKResult:
    """
    Simpler IK solving only for TCP position (orientation free).
    Uses only the 3×7 position Jacobian.
    Useful for the planner when orientation doesn't matter.
    """
    if q_init is None:
        q_init = PANDA.random_config()

    q = q_init.copy()

    for iteration in range(max_iter):
        p_cur = tcp_position(q)
        e_pos = p_desired - p_cur
        err = float(np.linalg.norm(e_pos))

        if err < tol:
            # Build a dummy T for the result
            T, _ = forward_kinematics(q)
            return IKResult(q, True, err, 0.0, iteration)

        J = geometric_jacobian(q)[:3, :]      # 3×7
        manip = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
        lam = _adaptive_damping(manip / 10.0)  # scale: 3x3 det is smaller
        J_pinv = damped_pseudoinverse(J, lam)

        P = nullspace_projector(J, lam)
        dq = 0.5 * J_pinv @ e_pos + P @ _nullspace_gradient(q)
        q = PANDA.clamp(q + dq)

    p_cur = tcp_position(q)
    return IKResult(q, False, float(np.linalg.norm(p_desired - p_cur)), 0.0, max_iter)
