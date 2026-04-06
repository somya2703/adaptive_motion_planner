"""
tests/test_kinematics.py
------------------------
Unit tests for forward kinematics, Jacobian, and IK solver.
Run with: pytest tests/test_kinematics.py -v
"""

import numpy as np
import pytest
from kinematics.forward import forward_kinematics, tcp_position, link_positions
from kinematics.jacobian import (
    geometric_jacobian, manipulability, damped_pseudoinverse, nullspace_projector
)
from kinematics.ik import solve_ik, solve_ik_position_only
from robot.panda import PANDA


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

class TestForwardKinematics:

    def test_home_pose_is_se3(self):
        """FK output must be a valid SE(3) matrix."""
        q = np.zeros(7)
        T, _ = forward_kinematics(q)
        assert T.shape == (4, 4)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8), "R not orthogonal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-8, "det(R) != 1"
        assert T[3, :].tolist() == [0, 0, 0, 1], "Last row wrong"

    def test_returns_7_link_frames(self):
        q = np.zeros(7)
        _, T_all = forward_kinematics(q)
        assert len(T_all) == 7

    def test_link_positions_shape(self):
        q = np.zeros(7)
        pts = link_positions(q)
        assert pts.shape == (7, 3)

    def test_tcp_changes_with_q(self):
        q1 = np.zeros(7)
        q2 = np.array([0.5, -0.3, 0.2, -1.0, 0.1, 0.8, 0.3])
        p1 = tcp_position(q1)
        p2 = tcp_position(q2)
        assert not np.allclose(p1, p2), "TCP should change when joints change"

    def test_valid_for_random_configs(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = PANDA.random_config(rng)
            T, _ = forward_kinematics(q)
            R = T[:3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

class TestJacobian:

    def test_shape(self):
        q = np.zeros(7)
        J = geometric_jacobian(q)
        assert J.shape == (6, 7)

    def test_pseudoinverse_shape(self):
        q = np.zeros(7)
        J = geometric_jacobian(q)
        J_pinv = damped_pseudoinverse(J)
        assert J_pinv.shape == (7, 6)

    def test_nullspace_projector_is_approximate_projection(self):
        """P^2 ≈ P must hold (damping introduces small error < 1e-2)."""
        q = np.array([0.3, -0.5, 0.1, -1.2, 0.2, 0.9, 0.4])
        J = geometric_jacobian(q)
        P = nullspace_projector(J)
        assert np.allclose(P @ P, P, atol=1e-2), \
            f"P not approximately idempotent, max err={np.max(np.abs(P@P - P)):.4f}"

    def test_manipulability_positive(self):
        q = np.zeros(7)
        m = manipulability(q)
        assert m >= 0.0

    def test_numerical_jacobian(self):
        """Jacobian (finite-diff implementation) matches finite differences."""
        q = np.array([0.1, -0.3, 0.0, -1.5, 0.1, 1.4, 0.5])
        eps = 1e-7
        p0 = tcp_position(q)
        J_num = np.zeros((3, 7))
        for i in range(7):
            dq = np.zeros(7); dq[i] = eps
            J_num[:, i] = (tcp_position(q + dq) - p0) / eps

        J_geo = geometric_jacobian(q)[:3, :]
        assert np.allclose(J_geo, J_num, atol=1e-5), \
            f"Max error: {np.max(np.abs(J_geo - J_num)):.2e}"

    def test_nullspace_produces_zero_tcp_motion(self):
        """Velocity in nullspace of J should produce (near-)zero TCP velocity."""
        q = np.array([0.2, -0.4, 0.3, -1.1, 0.2, 1.0, 0.5])
        J = geometric_jacobian(q)
        P = nullspace_projector(J)
        v = np.random.default_rng(7).standard_normal(7)
        q_null = P @ v
        tcp_vel = J[:3] @ q_null
        assert np.linalg.norm(tcp_vel) < 0.02, \
            f"Nullspace velocity causes TCP motion: {np.linalg.norm(tcp_vel):.4f}"


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

class TestIK:

    def test_position_ik_converges_from_nearby(self):
        """IK should recover a reachable position from a nearby init."""
        q_true = np.array([0.4, -0.6, 0.2, -1.3, 0.1, 1.0, 0.5])
        p_target = tcp_position(q_true)
        result = solve_ik_position_only(p_target, q_init=q_true + 0.05)
        assert result.success or result.error_pos < 0.005, \
            f"Position IK failed: error={result.error_pos:.4f} m"

    def test_full_ik_converges_from_nearby(self):
        """Full pose IK should converge from a close initial guess."""
        q_true = np.array([0.3, -0.4, 0.1, -1.2, 0.2, 1.1, 0.4])
        T_target, _ = forward_kinematics(q_true)
        result = solve_ik(T_target, q_init=q_true + 0.1)
        assert result.error_pos < 0.005, \
            f"Position error too large: {result.error_pos:.4f}"

    def test_ik_respects_joint_limits(self):
        """IK solution must be within joint limits."""
        q_true = np.array([0.5, -0.5, 0.3, -1.0, 0.3, 1.2, 0.3])
        T_target, _ = forward_kinematics(q_true)
        result = solve_ik(T_target, q_init=q_true + 0.1)
        assert PANDA.joint_in_limits(result.q), \
            f"IK solution violates joint limits: {result.q}"

    def test_ik_returns_ikresult(self):
        """IK should return an IKResult with expected fields."""
        q_true = np.array([0.2, -0.3, 0.1, -1.0, 0.1, 0.9, 0.3])
        T_target, _ = forward_kinematics(q_true)
        result = solve_ik(T_target, q_init=q_true)
        assert hasattr(result, 'q')
        assert hasattr(result, 'success')
        assert hasattr(result, 'error_pos')
        assert hasattr(result, 'iterations')
        assert result.q.shape == (7,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
