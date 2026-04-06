"""
tests/test_planner.py
---------------------
Unit tests for InformedRRTStar, safety constraints, and CBF.
Run with: pytest tests/test_planner.py -v
"""

import numpy as np
import pytest

from robot.panda import PANDA
from safety.constraints import ConstraintSet, JointLimitConstraint, SelfCollisionConstraint
from safety.cbf import CBFSafetyFilter, Obstacle, _h_obstacle
from planner.rrt_star import InformedRRTStar
from kinematics.forward import tcp_position, forward_kinematics


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------

class TestConstraints:

    def test_joint_limits_accept_mid(self):
        c = JointLimitConstraint()
        assert c.is_satisfied(PANDA.q_mid)

    def test_joint_limits_reject_out_of_range(self):
        c = JointLimitConstraint()
        q_bad = PANDA.q_max + 0.5
        assert not c.is_satisfied(q_bad)

    def test_joint_margin_positive(self):
        c = JointLimitConstraint(position_margin=0.1)
        assert c.margin(PANDA.q_mid) > 0

    def test_self_collision_midpoint_clear(self):
        c = SelfCollisionConstraint()
        assert c.is_satisfied(PANDA.q_mid), \
            f"q_mid self-collision check failed (clearance={c.min_clearance(PANDA.q_mid):.4f})"

    def test_constraint_set_qmid_valid(self):
        cs = ConstraintSet()
        assert cs.config_is_valid(PANDA.q_mid)

    def test_constraint_set_home_valid(self):
        cs = ConstraintSet()
        q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
        assert cs.config_is_valid(q_home)

    def test_constraint_set_diagnostics(self):
        cs = ConstraintSet()
        d = cs.diagnostics(PANDA.q_mid)
        assert "joint_margin" in d
        assert d["joint_margin"] > 0

    def test_torque_clamp(self):
        from safety.constraints import TorqueLimitConstraint
        tc = TorqueLimitConstraint()
        tau_big = PANDA.tau_max * 2.0
        tau_clamped = tc.clamp(tau_big)
        assert np.all(np.abs(tau_clamped) <= PANDA.tau_max)


# ---------------------------------------------------------------------------
# CBF tests
# ---------------------------------------------------------------------------

class TestCBF:

    def setup_method(self):
        self.obs = Obstacle(center=np.array([2.0, 0.0, 0.5]), radius=0.1)
        self.cbf = CBFSafetyFilter(obstacles=[self.obs])

    def test_h_positive_far_from_obstacle(self):
        """CBF h must be positive when arm is far from obstacle."""
        q = PANDA.q_mid
        h = _h_obstacle(q, self.obs)
        assert h > 0, f"Expected h > 0, got {h:.4f}"

    def test_h_negative_at_obstacle_centre(self):
        """CBF h must be negative when TCP is inside obstacle."""
        q = PANDA.q_mid
        p = tcp_position(q)
        nearby_obs = Obstacle(center=p, radius=0.01, margin=0.0)
        h = _h_obstacle(q, nearby_obs)
        assert h <= 0.0, f"Expected h <= 0 for obstacle at TCP, got h={h:.4f}"

    def test_filter_returns_safe_tau(self):
        q = PANDA.q_mid
        qd = np.zeros(7)
        tau = PANDA.tau_max * 0.5
        result = self.cbf.filter(tau, q, qd)
        assert result.tau_safe.shape == (7,)
        assert np.all(np.abs(result.tau_safe) <= PANDA.tau_max + 1e-6)

    def test_cbf_result_has_fields(self):
        result = self.cbf.filter(np.zeros(7), PANDA.q_mid, np.zeros(7))
        assert hasattr(result, "cbf_active")
        assert hasattr(result, "min_h")
        assert hasattr(result, "correction_norm")

    def test_config_safe_far_from_obstacle(self):
        assert self.cbf.is_config_safe(PANDA.q_mid)

    def test_min_clearance_positive(self):
        cl = self.cbf.min_clearance(PANDA.q_mid)
        assert cl > 0


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlanner:

    def setup_method(self):
        self.constraints = ConstraintSet()
        # Use configurations that are known to be valid
        self.q_start = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
        self.q_goal  = np.array([0.5, -0.3, 0.2, -1.0, 0.1, 0.9, 0.4])

    def _make_planner(self, obstacles=None, seed=0):
        return InformedRRTStar(
            constraints=self.constraints,
            obstacles=obstacles or [],
            max_iter=1500,
            seed=seed,
        )

    def test_start_and_goal_are_valid(self):
        assert self.constraints.config_is_valid(self.q_start), "Start invalid"
        assert self.constraints.config_is_valid(self.q_goal), "Goal invalid"

    def test_plans_in_free_space(self):
        result = self._make_planner().plan(self.q_start, self.q_goal)
        assert result.success, f"Should find path in free space (tree={result.tree_size})"

    def test_path_has_at_least_two_waypoints(self):
        result = self._make_planner().plan(self.q_start, self.q_goal)
        if result.success:
            assert len(result.path) >= 2

    def test_all_waypoints_satisfy_joint_limits(self):
        """Planner guarantees joint limits (self-col skipped in hot path)."""
        result = self._make_planner().plan(self.q_start, self.q_goal)
        if result.success:
            for q in result.path:
                ok = self.constraints.joint.is_satisfied(q)
                assert ok, f"Waypoint violates joint limits"
        result = self._make_planner(seed=99).plan(self.q_start, self.q_goal)
        assert hasattr(result, "success")
        assert hasattr(result, "cost")
        assert hasattr(result, "iterations")
        assert hasattr(result, "time_s")
        assert hasattr(result, "tree_size")

    def test_cost_is_positive_on_success(self):
        result = self._make_planner(seed=1).plan(self.q_start, self.q_goal)
        if result.success:
            assert result.cost > 0

    def test_avoids_obstacle_blocking_straight_line(self):
        """Planner should route around an obstacle between start and goal TCPs."""
        p_s = tcp_position(self.q_start)
        p_g = tcp_position(self.q_goal)
        p_mid = (p_s + p_g) / 2
        obs = Obstacle(center=p_mid, radius=0.04, margin=0.02)
        result = self._make_planner(obstacles=[obs], seed=2).plan(
            self.q_start, self.q_goal
        )
        if result.success:
            for q in result.path:
                p = tcp_position(q)
                dist = np.linalg.norm(p - obs.center)
                assert dist >= obs.effective_radius - 0.02, \
                    f"Path enters obstacle: dist={dist:.3f} < r_eff={obs.effective_radius:.3f}"


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------

class TestTrajectory:

    def test_smooth_path_output_shape(self):
        from planner.trajectory import smooth_path
        waypoints = [PANDA.random_config(np.random.default_rng(i)) for i in range(10)]
        smooth = smooth_path(waypoints, n_points=50)
        assert smooth.shape == (50, 7)

    def test_build_trajectory_is_nonempty(self):
        from planner.trajectory import build_trajectory
        waypoints = [
            np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
            np.array([0.3, -0.4, 0.1, -1.1, 0.1, 0.9, 0.3]),
        ]
        traj = build_trajectory(waypoints, dt=0.01)
        assert len(traj) > 0
        assert traj[0].t == pytest.approx(0.0, abs=1e-9)

    def test_trajectory_respects_velocity_limits(self):
        from planner.trajectory import build_trajectory
        waypoints = [
            np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
            np.array([0.5, -0.3, 0.2, -1.0, 0.2, 1.0, 0.4]),
        ]
        traj = build_trajectory(waypoints, dt=0.01, v_max_fraction=0.6)
        for pt in traj:
            assert np.all(np.abs(pt.qd) <= PANDA.qd_max + 0.01), \
                f"Velocity limit exceeded: {pt.qd}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
