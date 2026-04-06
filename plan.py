"""
Main entry point for a single planning query.

Usage
-----
    python plan.py --scene cluttered --visualize
    python plan.py --start "0,0,0,-1.57,0,1.57,0" --goal "1.2,-0.5,0.3,1.0,0.1,-0.8,0.5"
    python plan.py --scene narrow_corridor --iter 8000 --visualize
"""

import argparse
import sys
import time
import numpy as np

from robot.panda import PANDA
from safety.constraints import ConstraintSet
from safety.cbf import Obstacle
from planner.rrt_star import InformedRRTStar
from planner.trajectory import build_trajectory


# ---------------------------------------------------------------------------
# Preset scenes
# ---------------------------------------------------------------------------

SCENES = {
    "free": [],
    "cluttered": [
        Obstacle(center=np.array([0.45,  0.20, 0.40]), radius=0.10, margin=0.04),
        Obstacle(center=np.array([0.50, -0.15, 0.35]), radius=0.08, margin=0.04),
        Obstacle(center=np.array([0.40,  0.00, 0.65]), radius=0.09, margin=0.04),
        Obstacle(center=np.array([0.60,  0.25, 0.25]), radius=0.07, margin=0.04),
    ],
    "narrow_corridor": [
        Obstacle(center=np.array([0.50,  0.30, 0.35]), radius=0.16, margin=0.04),
        Obstacle(center=np.array([0.50, -0.30, 0.35]), radius=0.16, margin=0.04),
    ],
}

# Preset configs
Q_HOME = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
Q_REACH = np.array([0.749, 0.05, -0.018, -2.329, -2.829, 0.708, 1.113])


def parse_q(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 7:
        raise ValueError(f"Expected 7 joint values, got {len(vals)}")
    return np.array(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single motion planning query")
    parser.add_argument("--scene",     default="cluttered",
                        choices=list(SCENES.keys()))
    parser.add_argument("--start",     default=None,
                        help="Comma-separated 7 joint angles (rad). Default: home.")
    parser.add_argument("--goal",      default=None,
                        help="Comma-separated 7 joint angles (rad). Default: reach.")
    parser.add_argument("--iter",      type=int, default=5000,
                        help="RRT* iteration budget")
    parser.add_argument("--visualize", action="store_true",
                        help="Show matplotlib 3D visualisation")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    q_start = parse_q(args.start) if args.start else Q_HOME.copy()
    q_goal  = parse_q(args.goal)  if args.goal  else Q_REACH.copy()
    obstacles = SCENES[args.scene]
    constraints = ConstraintSet()

    print(f"\n  Scene       : {args.scene}")
    print(f"  Obstacles   : {len(obstacles)}")
    print(f"  RRT* iters  : {args.iter}")
    print(f"  Start       : {np.round(q_start, 3)}")
    print(f"  Goal        : {np.round(q_goal, 3)}")

    # Validate start/goal
    if not constraints.config_is_valid(q_start):
        sys.exit("  ERROR: start configuration violates constraints.")
    if not constraints.config_is_valid(q_goal):
        sys.exit("  ERROR: goal configuration violates constraints.")

    # Plan
    planner = InformedRRTStar(
        constraints=constraints,
        obstacles=obstacles,
        max_iter=args.iter,
        seed=args.seed,
    )

    print("\n  Planning ...", flush=True)
    t0 = time.perf_counter()
    result = planner.plan(q_start, q_goal)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not result.success:
        print(f"\n  FAILED — no path found in {args.iter} iterations ({elapsed_ms:.0f} ms)")
        sys.exit(1)

    print(f"\n  SUCCESS")
    print(f"  Plan time   : {elapsed_ms:.1f} ms")
    print(f"  Tree size   : {result.tree_size} nodes")
    print(f"  Path length : {result.cost:.4f} rad (joint-space L2)")
    print(f"  Waypoints   : {len(result.path)}")

    # Build trajectory
    traj = build_trajectory(result.path, dt=0.001)
    print(f"  Trajectory  : {len(traj)} points @ 1 kHz  ({traj[-1].t:.2f} s total)")

    # Quick constraint check on trajectory
    violations = 0
    for pt in traj[::10]:   # check every 10th point
        if not constraints.config_is_valid(pt.q, pt.qd):
            violations += 1
    print(f"  Constraint violations (sampled): {violations}")

    if args.visualize:
        try:
            from visualize import visualize_plan
            visualize_plan(result.path, obstacles, q_start, q_goal)
        except ImportError:
            print("  matplotlib not available — skipping visualisation")


if __name__ == "__main__":
    main()
