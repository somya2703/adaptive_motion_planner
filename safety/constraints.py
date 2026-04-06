"""
Hard constraints for safe Panda operation.

Three constraint types are enforced:

1. JointLimitConstraint   — position and velocity limits
2. TorqueLimitConstraint  — actuator torque bounds
3. SelfCollisionConstraint — sphere-sphere checks on adjacent links

All constraints expose a common interface:
    .is_satisfied(q, qd, tau) -> bool
    .violation(q, qd, tau)    -> float  (0 = satisfied, >0 = violated by this amount)
    .margin(q, qd, tau)       -> float  (distance to constraint boundary)
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from robot.panda import PANDA, LinkSphere
from kinematics.forward import forward_kinematics


# ---------------------------------------------------------------------------
# Joint limit constraint
# ---------------------------------------------------------------------------

class JointLimitConstraint:
    """
    Enforces:   q_min ≤ q ≤ q_max
                |q̇_i| ≤ qd_max_i

    A configurable margin pulls the effective limit inward to leave
    a safe buffer for the controller.
    """

    def __init__(self, position_margin: float = 0.05, velocity_margin: float = 0.1):
        """
        Parameters
        ----------
        position_margin : inward margin on joint angles (rad)
        velocity_margin : fraction of velocity limit reserved as buffer
        """
        self.q_min  = PANDA.q_min  + position_margin
        self.q_max  = PANDA.q_max  - position_margin
        self.qd_max = PANDA.qd_max * (1.0 - velocity_margin)

    def is_satisfied(
        self,
        q:   np.ndarray,
        qd:  np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ) -> bool:
        pos_ok = np.all(q >= self.q_min) and np.all(q <= self.q_max)
        if qd is None:
            return bool(pos_ok)
        vel_ok = np.all(np.abs(qd) <= self.qd_max)
        return bool(pos_ok and vel_ok)

    def violation(self, q: np.ndarray, qd: np.ndarray | None = None, **_) -> float:
        """Return the maximum constraint violation (0 if satisfied)."""
        v = max(
            float(np.max(self.q_min - q)),
            float(np.max(q - self.q_max)),
        )
        if qd is not None:
            v = max(v, float(np.max(np.abs(qd) - self.qd_max)))
        return max(0.0, v)

    def margin(self, q: np.ndarray, **_) -> float:
        """Minimum distance to any joint limit (positive = inside limits)."""
        return float(min(
            np.min(q - self.q_min),
            np.min(self.q_max - q),
        ))

    def clamp_velocity(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Clip a velocity command to the effective velocity limits."""
        return np.clip(qd, -self.qd_max, self.qd_max)


# ---------------------------------------------------------------------------
# Torque limit constraint
# ---------------------------------------------------------------------------

class TorqueLimitConstraint:
    """
    Enforces: |τ_i| ≤ τ_max_i  (per-joint actuator torque)
    """

    def __init__(self, margin_fraction: float = 0.10):
        self.tau_max = PANDA.tau_max * (1.0 - margin_fraction)

    def is_satisfied(self, tau: np.ndarray, **_) -> bool:
        return bool(np.all(np.abs(tau) <= self.tau_max))

    def violation(self, tau: np.ndarray, **_) -> float:
        return float(max(0.0, np.max(np.abs(tau) - self.tau_max)))

    def margin(self, tau: np.ndarray, **_) -> float:
        return float(np.min(self.tau_max - np.abs(tau)))

    def clamp(self, tau: np.ndarray) -> np.ndarray:
        return np.clip(tau, -self.tau_max, self.tau_max)


# ---------------------------------------------------------------------------
# Self-collision constraint
# ---------------------------------------------------------------------------

@dataclass
class SpherePair:
    i: int     # index into SELF_COLLISION_SPHERES
    j: int
    min_dist: float   # minimum allowed centre-to-centre distance


class SelfCollisionConstraint:
    """
    Approximate self-collision checking using per-link bounding spheres.

    For each non-adjacent sphere pair (i, j), we check:
        ||c_i - c_j||  ≥  r_i + r_j + safety_margin

    We skip adjacent links (|i - j| ≤ 1) as they are always close.
    """

    def __init__(self, safety_margin: float = 0.005):
        """
        Parameters
        ----------
        safety_margin : extra clearance beyond sum of radii (metres)
        """
        self.spheres = PANDA.SELF_COLLISION_SPHERES
        self.safety_margin = safety_margin

        # Build list of non-adjacent pairs to check
        self.pairs: List[SpherePair] = []
        n = len(self.spheres)
        for i in range(n):
            for j in range(i + 2, n):   # skip adjacent links
                min_d = (self.spheres[i].radius
                         + self.spheres[j].radius
                         + safety_margin)
                self.pairs.append(SpherePair(i, j, min_d))

    def _sphere_centres(self, q: np.ndarray) -> List[np.ndarray]:
        """Compute sphere centres in the base frame for a given q."""
        _, T_all = forward_kinematics(q)
        centres = []
        for sphere in self.spheres:
            T = T_all[sphere.joint_idx]
            c = T[:3, :3] @ sphere.center_local + T[:3, 3]
            centres.append(c)
        return centres

    def is_satisfied(self, q: np.ndarray, **_) -> bool:
        centres = self._sphere_centres(q)
        for pair in self.pairs:
            dist = float(np.linalg.norm(centres[pair.i] - centres[pair.j]))
            if dist < pair.min_dist:
                return False
        return True

    def min_clearance(self, q: np.ndarray) -> float:
        """
        Return the minimum clearance across all sphere pairs.
        Negative means self-collision.
        """
        centres = self._sphere_centres(q)
        min_cl = float("inf")
        for pair in self.pairs:
            dist = float(np.linalg.norm(centres[pair.i] - centres[pair.j]))
            cl = dist - pair.min_dist
            if cl < min_cl:
                min_cl = cl
        return min_cl

    def violation(self, q: np.ndarray, **_) -> float:
        return max(0.0, -self.min_clearance(q))


# ---------------------------------------------------------------------------
# Composite constraint checker
# ---------------------------------------------------------------------------

class ConstraintSet:
    """
    Combines all constraints into a single checker used by the planner
    and safety monitor.
    """

    def __init__(
        self,
        pos_margin: float = 0.05,
        vel_margin: float = 0.10,
        torque_margin: float = 0.10,
        self_col_margin: float = 0.005,
    ):
        self.joint    = JointLimitConstraint(pos_margin, vel_margin)
        self.torque   = TorqueLimitConstraint(torque_margin)
        self.self_col = SelfCollisionConstraint(self_col_margin)

    def config_is_valid(
        self,
        q:   np.ndarray,
        qd:  np.ndarray | None = None,
        tau: np.ndarray | None = None,
        check_self_col: bool = True,
    ) -> bool:
        """Return True if q satisfies all active constraints."""
        if not self.joint.is_satisfied(q, qd):
            return False
        if tau is not None and not self.torque.is_satisfied(tau):
            return False
        if check_self_col and not self.self_col.is_satisfied(q):
            return False
        return True

    def diagnostics(self, q: np.ndarray) -> dict:
        """Return per-constraint margins for logging / visualisation."""
        return {
            "joint_margin":      self.joint.margin(q),
            "self_col_clearance": self.self_col.min_clearance(q),
        }
