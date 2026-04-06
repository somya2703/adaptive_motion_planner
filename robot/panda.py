"""
Franka Panda kinematic model.

DH parameters, joint limits, torque limits, and link geometry
(sphere approximations for self-collision checking).

All angles in radians, distances in metres.
This is essentially the spec sheet for Franka Panda
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class JointSpec:
    """Physical specification for a single revolute joint."""
    name: str
    q_min: float          # lower position limit (rad)
    q_max: float          # upper position limit (rad)
    qd_max: float         # velocity limit (rad/s)
    tau_max: float        # torque limit (Nm)
    # DH parameters (Modified DH convention)
    a: float              # link length
    d: float              # link offset
    alpha: float          # link twist


@dataclass
class LinkSphere:
    """Bounding sphere on a link for self-collision."""
    joint_idx: int        # attached to this joint's frame
    center_local: np.ndarray   # centre in the joint frame (3,)
    radius: float


class PandaModel:



    # fmt: off
    JOINTS: List[JointSpec] = [
        JointSpec("panda_joint1", -2.8973,  2.8973, 2.175, 87,   a=0,       d=0.333,  alpha=0),
        JointSpec("panda_joint2", -1.7628,  1.7628, 2.175, 87,   a=0,       d=0,      alpha=-np.pi/2),
        JointSpec("panda_joint3", -2.8973,  2.8973, 2.175, 87,   a=0,       d=0.316,  alpha=np.pi/2),
        JointSpec("panda_joint4", -3.0718, -0.0698, 2.175, 87,   a=0.0825,  d=0,      alpha=np.pi/2),
        JointSpec("panda_joint5", -2.8973,  2.8973, 2.610, 12,   a=-0.0825, d=0.384,  alpha=-np.pi/2),
        JointSpec("panda_joint6", -0.0175,  3.7525, 2.610, 12,   a=0,       d=0,      alpha=np.pi/2),
        JointSpec("panda_joint7", -2.8973,  2.8973, 2.610, 12,   a=0.088,   d=0.107,  alpha=np.pi/2),
    ]
    # fmt: on
    """fmt is FAST Marching Tree, used for sampling based motion planning.
        We Expands like a wave, moves to next node based on distance and collision free feasibility
        Precomputed trajectories. Faster convergence than RRT*"""

    # Self-collision sphere approximation.
    # Radii tuned so canonical home/mid configs do not self-flag.
    SELF_COLLISION_SPHERES: List[LinkSphere] = [
        LinkSphere(0, np.array([0, 0, 0.15]),   0.055),  # link 1  
        LinkSphere(1, np.array([0, 0, 0]),       0.07),  # link 2
        LinkSphere(2, np.array([0, 0, 0.15]),   0.07),  # link 3
        LinkSphere(3, np.array([0.08, 0, 0]),   0.05),  # link 4
        LinkSphere(4, np.array([0, 0, 0.15]),   0.06),  # link 5
        LinkSphere(5, np.array([0, 0, 0]),       0.06),  # link 6
        LinkSphere(6, np.array([0, 0, 0.05]),   0.05),  # link 7
    ]

    DOF = 7
    # Tool-centre-point (TCP) offset from joint-7 frame
    TCP_OFFSET = np.array([0, 0, 0.10])  # 10 cm along z

    @property
    def q_min(self) -> np.ndarray:
        return np.array([j.q_min for j in self.JOINTS])

    @property
    def q_max(self) -> np.ndarray:
        return np.array([j.q_max for j in self.JOINTS])

    @property
    def qd_max(self) -> np.ndarray:
        return np.array([j.qd_max for j in self.JOINTS])

    @property
    def tau_max(self) -> np.ndarray:
        return np.array([j.tau_max for j in self.JOINTS])

    @property
    def q_mid(self) -> np.ndarray:
        """Joint-space midpoint — used as nullspace attractor in IK."""
        return (self.q_min + self.q_max) / 2.0

    def random_config(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a uniformly random valid joint configuration."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.q_min, self.q_max)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Hard-clamp joint angles to their limits."""
        return np.clip(q, self.q_min, self.q_max)

    def joint_in_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        """Return True if all joints are within limits (with optional margin)."""
        return bool(
            np.all(q >= self.q_min + margin) and np.all(q <= self.q_max - margin)
        )


#
PANDA = PandaModel()
