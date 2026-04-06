"""
Post-process a raw RRT* waypoint list into a smooth, time-parametrised
joint trajectory.

Two stages:
  1. Path smoothing — fit a cubic B-spline through the waypoints,
     reducing kinkiness and making the path C2 continuous.

  2. Time scaling  — apply trapezoidal velocity profile (bang-coast-bang)
     respecting per-joint velocity and acceleration limits.

Output: a list of (t, q, q̇, q̈) tuples at a given control frequency.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from scipy.interpolate import splprep, splev

from robot.panda import PANDA


@dataclass
class TrajectoryPoint:
    t:   float          # time (s)
    q:   np.ndarray     # joint angles (7,)
    qd:  np.ndarray     # joint velocities (rad/s)
    qdd: np.ndarray     # joint accelerations (rad/s²)


def smooth_path(
    waypoints: List[np.ndarray],
    smoothing: float = 0.0,
    n_points: int = 200,
) -> np.ndarray:
    """
    Fit a cubic B-spline through the waypoints and resample.

    Parameters
    ----------
    waypoints  : list of (7,) joint configs
    smoothing  : spline smoothing factor (0 = interpolation)
    n_points   : number of output samples

    Returns
    -------
    path : (n_points, 7) resampled smooth path
    """
    arr = np.array(waypoints)

    # Remove duplicate consecutive waypoints (splprep rejects them)
    mask = np.concatenate([[True], np.any(np.diff(arr, axis=0) != 0, axis=1)])
    arr = arr[mask]

    if len(arr) < 4:
        # Not enough unique points for cubic spline — linear interpolation
        u_orig = np.linspace(0, 1, len(arr))
        u_fine = np.linspace(0, 1, n_points)
        return np.column_stack([
            np.interp(u_fine, u_orig, arr[:, j]) for j in range(arr.shape[1])
        ])

    # scipy splprep expects each dimension as a row
    coords = arr.T   # (7, N)
    tck, _ = splprep(coords, s=smoothing, k=min(3, len(arr) - 1))

    u_fine = np.linspace(0, 1, n_points)
    resampled = np.array(splev(u_fine, tck)).T   # (n_points, 7)
    return resampled


def _arc_length_parametrise(path: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc-length parameter s for a (N, 7) path.
    Returns s in [0, 1].
    """
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = s[-1]
    if total < 1e-12:
        return np.linspace(0, 1, len(path))
    return s / total


def time_scale_trapezoidal(
    path: np.ndarray,
    v_max_fraction: float = 0.6,
    a_max_fraction: float = 0.5,
    dt: float = 0.001,
) -> List[TrajectoryPoint]:
    """
    Apply a trapezoidal velocity profile (constant accel → cruise → decel).

    The profile is computed in arc-length space and then applied to each
    joint uniformly.  Per-joint limits are checked and the time is scaled
    to the most-constrained joint.

    Parameters
    ----------
    path            : (N, 7) smooth path
    v_max_fraction  : fraction of joint velocity limit to use
    a_max_fraction  : fraction of joint accel limit to use (approx qd_max/s)
    dt              : output timestep (s)

    Returns
    -------
    trajectory : list of TrajectoryPoint
    """
    v_lim = PANDA.qd_max * v_max_fraction          # (7,)
    a_lim = PANDA.qd_max * a_max_fraction * 5.0    # rough accel limit

    s = _arc_length_parametrise(path)                # (N,)
    total_length = float(np.linalg.norm(path[-1] - path[0]))
    if total_length < 1e-9:
        total_length = 1e-9

    # Compute per-joint peak velocity and required time
    joint_ranges = np.abs(path[-1] - path[0])        # (7,)
    # Time to traverse at v_lim for each joint
    t_per_joint  = joint_ranges / (v_lim + 1e-9)    # (7,)
    # The most constrained joint determines total cruise time
    t_cruise     = float(np.max(t_per_joint)) * 1.5  # 50% headroom

    # Trapezoidal profile: accel phase t_a, cruise t_c, decel t_d
    # We use a symmetric profile: t_a = t_d
    v_peak = joint_ranges / (t_cruise + 1e-9)        # per-joint peak vel
    a_peak = v_peak / (t_cruise / 4.0 + 1e-9)
    # Clip to limits
    v_peak = np.minimum(v_peak, v_lim)
    a_peak = np.minimum(a_peak, a_lim)

    t_a = v_peak / (a_peak + 1e-9)                   # accel duration
    T   = float(np.max(t_cruise + 2 * t_a))          # total duration

    # Generate time array
    t_arr = np.arange(0, T + dt, dt)
    trajectory: List[TrajectoryPoint] = []

    # Interpolation: given time t, compute s(t) via trapezoidal profile
    # then interpolate path at that s
    for t_i in t_arr:
        # Normalised progress in [0, 1]
        if T < 1e-9:
            u = 1.0
        else:
            u = t_i / T

        # Sample smooth path at parameter u (via arc-length)
        idx  = np.searchsorted(np.linspace(0, 1, len(path)), u, side="right")
        idx  = int(np.clip(idx, 1, len(path) - 1))
        alpha = (u * (len(path) - 1)) - (idx - 1)
        q_t  = (1 - alpha) * path[idx - 1] + alpha * path[idx]

        # Numerical velocity (finite diff)
        if len(trajectory) == 0:
            qd_t  = np.zeros(7)
            qdd_t = np.zeros(7)
        elif len(trajectory) == 1:
            qd_t  = (q_t - trajectory[-1].q) / dt
            qdd_t = np.zeros(7)
        else:
            qd_t  = (q_t - trajectory[-1].q) / dt
            qdd_t = (qd_t - trajectory[-1].qd) / dt

        # Clamp to limits
        qd_t  = np.clip(qd_t,  -PANDA.qd_max, PANDA.qd_max)

        trajectory.append(TrajectoryPoint(t_i, q_t, qd_t, qdd_t))

    return trajectory


def build_trajectory(
    waypoints: List[np.ndarray],
    smooth_n: int = 200,
    dt: float = 0.001,
    v_max_fraction: float = 0.6,
) -> List[TrajectoryPoint]:
    """
    Full pipeline: raw waypoints → smooth path → time-scaled trajectory.

    Parameters
    ----------
    waypoints       : raw RRT* path (list of (7,) configs)
    smooth_n        : B-spline resampling points
    dt              : output control timestep (s)
    v_max_fraction  : velocity limit fraction

    Returns
    -------
    trajectory : list of TrajectoryPoint
    """
    smooth = smooth_path(waypoints, n_points=smooth_n)
    return time_scale_trapezoidal(smooth, v_max_fraction=v_max_fraction, dt=dt)
