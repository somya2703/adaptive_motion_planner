"""
visualize.py
------------
Matplotlib 3D visualisation of a planned joint-space path.

Shows:
  - Panda link chain (simplified stick model)
  - Planned path as a Cartesian TCP trace
  - Obstacle spheres
  - Start / goal end-effector positions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List

from kinematics.forward import forward_kinematics, link_positions, tcp_position
from safety.cbf import Obstacle


def _draw_sphere(ax: Axes3D, centre: np.ndarray, radius: float,
                 color: str = "red", alpha: float = 0.25) -> None:
    """Draw a semi-transparent wireframe sphere."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def _draw_arm(ax: Axes3D, q: np.ndarray, color: str = "steelblue",
              lw: float = 3, alpha: float = 0.8) -> None:
    """Draw the Panda arm as a stick figure."""
    link_pts = link_positions(q)          # (7, 3)
    origin   = np.zeros((1, 3))
    pts      = np.vstack([origin, link_pts])

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            "o-", color=color, linewidth=lw, alpha=alpha,
            markersize=4, markerfacecolor="white")


def _tcp_trace(path: List[np.ndarray]) -> np.ndarray:
    """Compute TCP positions for all configs in the path."""
    return np.array([tcp_position(q) for q in path])


def visualize_plan(
    path: List[np.ndarray],
    obstacles: List[Obstacle],
    q_start: np.ndarray,
    q_goal:  np.ndarray,
    title: str = "Adaptive Motion Planner",
) -> None:
    """
    Interactive 3D plot of the planned path.

    Parameters
    ----------
    path       : list of (7,) joint configs (raw or smoothed)
    obstacles  : list of Obstacle
    q_start    : start configuration
    q_goal     : goal configuration
    """
    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Draw obstacles
    for obs in obstacles:
        _draw_sphere(ax, obs.center, obs.effective_radius,
                     color="#e05252", alpha=0.22)
        _draw_sphere(ax, obs.center, obs.radius,
                     color="#c03030", alpha=0.35)

    # TCP path trace
    tcp_pts = _tcp_trace(path)
    ax.plot(tcp_pts[:, 0], tcp_pts[:, 1], tcp_pts[:, 2],
            "-", color="#2196F3", linewidth=2, label="TCP path", zorder=5)

    # Start/goal arm poses
    _draw_arm(ax, q_start, color="#4CAF50", lw=4, alpha=0.9)   # green = start
    _draw_arm(ax, q_goal,  color="#FF9800", lw=4, alpha=0.9)   # orange = goal

    # Mark start and goal TCP
    p_start = tcp_position(q_start)
    p_goal  = tcp_position(q_goal)
    ax.scatter(*p_start, color="#4CAF50", s=80, zorder=10, label="Start TCP")
    ax.scatter(*p_goal,  color="#FF9800", s=80, zorder=10, label="Goal TCP")

    # Draw a few intermediate arm poses
    n_poses = min(5, len(path))
    indices = np.linspace(0, len(path) - 1, n_poses, dtype=int)
    for idx in indices[1:-1]:
        _draw_arm(ax, path[idx], color="#90CAF9", lw=2, alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

    # Equal aspect ratio
    all_pts = np.vstack([tcp_pts, p_start.reshape(1, 3), p_goal.reshape(1, 3)])
    ranges  = all_pts.max(axis=0) - all_pts.min(axis=0)
    mid     = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    max_r   = max(ranges.max() / 2.0, 0.5)
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(max(0, mid[2] - max_r), mid[2] + max_r)

    plt.tight_layout()
    plt.show()


def visualize_trajectory(
    traj_points,   # List[TrajectoryPoint]
    title: str = "Joint Trajectory",
) -> None:
    """
    Plot joint angles, velocities, and accelerations over time.
    """
    t   = np.array([pt.t   for pt in traj_points])
    q   = np.array([pt.q   for pt in traj_points])
    qd  = np.array([pt.qd  for pt in traj_points])
    qdd = np.array([pt.qdd for pt in traj_points])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    for j in range(7):
        axes[0].plot(t, np.degrees(q[:, j]),   color=colors[j], label=f"J{j+1}")
        axes[1].plot(t, np.degrees(qd[:, j]),  color=colors[j])
        axes[2].plot(t, np.degrees(qdd[:, j]), color=colors[j])

    axes[0].set_ylabel("Position (deg)")
    axes[1].set_ylabel("Velocity (deg/s)")
    axes[2].set_ylabel("Acceleration (deg/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", ncol=7, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
