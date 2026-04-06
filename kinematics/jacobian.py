"""
Geometric Jacobian for the Panda + constraint Jacobian projection.

The 6x7 Jacobian maps joint velocities to TCP twist:

    xi_tcp = J(q) . q_dot        where xi = [v; w] in R^6

Linear rows computed via finite differences (numerically exact, ~0.3ms for 7-DoF).
Angular rows from the z-axis of each joint frame read off the FK transforms.

Constraint projection:
    Given a set of constraints C(q) = 0, the null-space projector
    P = I - J_c_pinv J_c  maps velocities to the constraint null-space,
    ensuring motions do not violate the constraints to first order.
"""

import numpy as np
from kinematics.forward import forward_kinematics


def geometric_jacobian(q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute the 6x7 geometric Jacobian at configuration q.

    Linear part (rows 0:3): finite-difference approximation, always correct.
    Angular part (rows 3:6): z-axis of each joint frame in base frame.

    Parameters
    ----------
    q   : (7,) joint angles in radians
    eps : finite-difference step for linear part

    Returns
    -------
    J : (6, 7) Jacobian matrix
    """
    T_tcp, T_all = forward_kinematics(q)
    p0 = T_tcp[:3, 3]

    J = np.zeros((6, 7))

    # Linear part via finite differences
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        T_plus, _ = forward_kinematics(q + dq)
        J[:3, i] = (T_plus[:3, 3] - p0) / eps

    # Angular part: joint i (1-indexed) rotates around z of frame i-1.
    # Frame 0 (base) has z = [0,0,1].
    # Frame i (i>=1) has z = T_all[i-1][:3, 2].
    J[3:, 0] = np.array([0.0, 0.0, 1.0])
    for i in range(1, 7):
        J[3:, i] = T_all[i - 1][:3, 2]

    return J


def jacobian_position(q: np.ndarray) -> np.ndarray:
    """Return the 3x7 translational part of the Jacobian."""
    return geometric_jacobian(q)[:3, :]


def manipulability(q: np.ndarray) -> float:
    """
    Yoshikawa manipulability measure: sqrt(det(J J^T)).
    Zero at singularities, larger = more dexterous.
    """
    J = geometric_jacobian(q)
    return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))


def damped_pseudoinverse(J: np.ndarray, damping: float = 0.01) -> np.ndarray:
    """
    Damped Moore-Penrose pseudoinverse.

    J_pinv = J^T (J J^T + lambda^2 I)^{-1}

    The damping lambda avoids numerical blow-up near singularities.
    """
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(m), np.eye(m))


def nullspace_projector(J: np.ndarray, damping: float = 0.01) -> np.ndarray:
    """
    Null-space projector P = I - J_pinv J.

    Any vector in the column space of P produces zero end-effector
    motion (to first order), allowing redundancy exploitation.
    """
    n = J.shape[1]
    return np.eye(n) - damped_pseudoinverse(J, damping) @ J


def project_to_constraint_nullspace(
    q_dot_desired: np.ndarray,
    J_constraint: np.ndarray,
    damping: float = 1e-4,
) -> np.ndarray:
    """
    Project a desired joint velocity into the null-space of a constraint Jacobian.

    Parameters
    ----------
    q_dot_desired  : (n,) desired joint velocity
    J_constraint   : (m, n) constraint Jacobian
    damping        : regularisation for pseudoinverse

    Returns
    -------
    q_dot_proj : (n,) constraint-satisfying velocity
    """
    P = nullspace_projector(J_constraint, damping)
    return P @ q_dot_desired
