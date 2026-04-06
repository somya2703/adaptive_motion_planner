"""
Real-time dynamic replanning for moving obstacles.

Architecture
------------
  DynamicReplanner maintains the current planned path and monitors it
  against the live obstacle positions.  When a collision is detected on
  the upcoming path, it triggers a background replan from the current
  robot state.

  The design is single-threaded (for clarity); in production this would
  run the replan in a separate thread / ROS action server.

Replanning triggers:
  1. Obstacle enters the "danger corridor" around the next N waypoints.
  2. Deviation of the executed path beyond a threshold.
  3. Goal change requested by the operator.

Key parameters (all tunable in configs/planner.yaml):
  danger_horizon  : how many waypoints ahead to check (int)
  danger_margin   : additional clearance to trigger replan (m)
  replan_timeout  : maximum time budget for a replan (s)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from safety.cbf import Obstacle
from planner.rrt_star import InformedRRTStar, PlannerResult
from safety.constraints import ConstraintSet
from kinematics.forward import tcp_position


@dataclass
class ReplanStats:
    total_replans: int = 0
    successful_replans: int = 0
    avg_replan_ms: float = 0.0
    _times: List[float] = field(default_factory=list, repr=False)

    def record(self, elapsed_s: float, success: bool) -> None:
        self.total_replans += 1
        if success:
            self.successful_replans += 1
        self._times.append(elapsed_s * 1000)
        self.avg_replan_ms = float(np.mean(self._times))


class DynamicReplanner:
    """
    Wraps InformedRRTStar with continuous path monitoring and replanning.

    Usage
    -----
    >>> replanner = DynamicReplanner(constraints)
    >>> replanner.set_goal(q_goal)
    >>> replanner.replan(q_current)          # initial plan
    >>>
    >>> # In the control loop:
    >>> q_next = replanner.next_waypoint(q_current, obstacles)
    """

    def __init__(
        self,
        constraints: ConstraintSet,
        danger_horizon: int = 5,
        danger_margin: float = 0.08,
        replan_timeout: float = 0.5,
        planner_iter: int = 2000,
        step_size: float = 0.3,
        seed: int | None = 42,
    ):
        self.constraints     = constraints
        self.danger_horizon  = danger_horizon
        self.danger_margin   = danger_margin
        self.replan_timeout  = replan_timeout
        self.planner_iter    = planner_iter
        self.step_size       = step_size
        self.seed            = seed

        self._path: List[np.ndarray] = []
        self._path_idx: int = 0
        self._q_goal: Optional[np.ndarray] = None
        self._obstacles: List[Obstacle] = []
        self.stats = ReplanStats()

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def set_goal(self, q_goal: np.ndarray) -> None:
        self._q_goal = q_goal.copy()
        self._path = []
        self._path_idx = 0

    def update_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Call this each control cycle with fresh obstacle positions."""
        self._obstacles = obstacles

    def replan(self, q_current: np.ndarray) -> bool:
        """
        Trigger a full replan from q_current to the goal.

        Returns True if a new plan was found.
        """
        if self._q_goal is None:
            return False

        t0 = time.perf_counter()
        planner = self._make_planner()
        result = planner.plan(q_current, self._q_goal)
        elapsed = time.perf_counter() - t0

        self.stats.record(elapsed, result.success)

        if result.success:
            self._path     = result.path
            self._path_idx = 0
            return True
        return False

    def next_waypoint(
        self,
        q_current: np.ndarray,
        obstacles: List[Obstacle] | None = None,
    ) -> Optional[np.ndarray]:
        """
        Return the next waypoint on the current path, replanning if needed.

        Parameters
        ----------
        q_current : current robot joint configuration
        obstacles : updated obstacle list (optional, updates internal state)

        Returns
        -------
        q_next : next target configuration, or None if path is finished
        """
        if obstacles is not None:
            self.update_obstacles(obstacles)

        if not self._path or self._q_goal is None:
            return None

        # Advance the path pointer if we've reached the current waypoint
        if self._path_idx < len(self._path):
            dist = np.linalg.norm(q_current - self._path[self._path_idx])
            if dist < 0.05:   # within 0.05 rad of waypoint
                self._path_idx += 1

        if self._path_idx >= len(self._path):
            return None   # goal reached

        # Check if upcoming path is still collision-free
        if self._danger_detected():
            triggered = self.replan(q_current)
            if not triggered:
                # Could not replan — stay put or halt (safety policy)
                return q_current

        return self._path[self._path_idx]

    def path_length_remaining(self, q_current: np.ndarray) -> float:
        """Approximate remaining path length (joint-space L2 sum)."""
        if self._path_idx >= len(self._path):
            return 0.0
        remaining = self._path[self._path_idx:]
        length = float(np.linalg.norm(q_current - remaining[0]))
        for i in range(len(remaining) - 1):
            length += float(np.linalg.norm(remaining[i+1] - remaining[i]))
        return length

    @property
    def has_plan(self) -> bool:
        return len(self._path) > 0 and self._path_idx < len(self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_planner(self) -> InformedRRTStar:
        return InformedRRTStar(
            constraints=self.constraints,
            obstacles=self._obstacles,
            max_iter=self.planner_iter,
            step_size=self.step_size,
            seed=self.seed,
        )

    def _danger_detected(self) -> bool:
        """
        Scan the next `danger_horizon` waypoints for collisions with
        any of the current obstacles.
        """
        end = min(self._path_idx + self.danger_horizon, len(self._path))
        for i in range(self._path_idx, end):
            q = self._path[i]
            p = tcp_position(q)
            for obs in self._obstacles:
                eff_r = obs.effective_radius + self.danger_margin
                if float(np.linalg.norm(p - obs.center)) < eff_r:
                    return True
        return False
