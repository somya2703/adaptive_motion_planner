"""
Standard benchmark suite for the Adaptive Motion Planner.

Scenarios
---------
  free_space      — no obstacles, baseline performance
  narrow_corridor — two walls forcing a tight passage
  cluttered       — 8 random spherical obstacles
  dynamic         — 3 obstacles moving on sinusoidal trajectories
  self_collision  — configurations prone to self-collision

For each scenario we report:
  - Success rate   (over N trials)
  - Mean plan time (ms)
  - Std  plan time (ms)
  - Mean path length (joint-space L2 sum)
  - Tree size at termination

Usage
-----
    python benchmarks/benchmark.py --scenario all --trials 20 --output results/
    python benchmarks/benchmark.py --scenario narrow_corridor --trials 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from safety.constraints import ConstraintSet
from safety.cbf import Obstacle
from planner.rrt_star import InformedRRTStar
from robot.panda import PANDA


# ---------------------------------------------------------------------------
# Standard obstacle layouts
# ---------------------------------------------------------------------------

def _free_space_obstacles() -> List[Obstacle]:
    return []


def _narrow_corridor_obstacles() -> List[Obstacle]:
    """Two large spheres that force the arm through a narrow gap."""
    return [
        Obstacle(center=np.array([0.5,  0.3, 0.4]), radius=0.20, margin=0.04),
        Obstacle(center=np.array([0.5, -0.3, 0.4]), radius=0.20, margin=0.04),
    ]


def _cluttered_obstacles(seed: int = 0) -> List[Obstacle]:
    """8 randomly placed spherical obstacles."""
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(8):
        centre = rng.uniform([0.2, -0.5, 0.1], [0.7, 0.5, 0.7])
        radius = rng.uniform(0.05, 0.12)
        obs.append(Obstacle(center=centre, radius=radius, margin=0.03))
    return obs


def _goal_pairs(n: int, seed: int = 0) -> List[tuple]:
    """Generate n random valid (start, goal) pairs."""
    rng = np.random.default_rng(seed)
    pairs = []
    constraints = ConstraintSet()
    while len(pairs) < n:
        q_s = PANDA.random_config(rng)
        q_g = PANDA.random_config(rng)
        if (constraints.config_is_valid(q_s) and constraints.config_is_valid(q_g)
                and np.linalg.norm(q_g - q_s) > 0.5):
            pairs.append((q_s, q_g))
    return pairs


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    obstacles: List[Obstacle],
    n_trials: int = 20,
    max_iter: int = 5000,
    seed: int = 42,
) -> Dict:
    """
    Run n_trials planning queries for a given obstacle layout.

    Returns a dict with summary statistics.
    """
    constraints = ConstraintSet()
    pairs = _goal_pairs(n_trials, seed=seed)

    times_ms: List[float] = []
    lengths:  List[float] = []
    tree_sizes: List[int] = []
    successes = 0

    print(f"\n  Scenario: {name} | {n_trials} trials | {len(obstacles)} obstacles")
    print("  " + "-" * 50)

    for i, (q_start, q_goal) in enumerate(pairs):
        planner = InformedRRTStar(
            constraints=constraints,
            obstacles=obstacles,
            max_iter=max_iter,
            seed=seed + i,
        )
        t0 = time.perf_counter()
        result = planner.plan(q_start, q_goal)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result.success:
            successes += 1
            path_arr = np.array(result.path)
            length = float(np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1)))
            lengths.append(length)
            tree_sizes.append(result.tree_size)

        times_ms.append(elapsed_ms)
        status = "OK" if result.success else "FAIL"
        print(f"  [{i+1:3d}/{n_trials}] {status:4s}  {elapsed_ms:7.1f} ms"
              + (f"  len={lengths[-1]:.3f}" if result.success else ""))

    success_rate = successes / n_trials * 100
    stats = {
        "scenario":       name,
        "n_trials":       n_trials,
        "success_rate_%": round(success_rate, 1),
        "time_mean_ms":   round(float(np.mean(times_ms)), 1),
        "time_std_ms":    round(float(np.std(times_ms)), 1),
        "time_min_ms":    round(float(np.min(times_ms)), 1),
        "time_max_ms":    round(float(np.max(times_ms)), 1),
        "path_len_mean":  round(float(np.mean(lengths)), 4) if lengths else None,
        "tree_size_mean": round(float(np.mean(tree_sizes))) if tree_sizes else None,
    }

    print(f"\n  Result: {success_rate:.0f}% success | "
          f"mean {stats['time_mean_ms']:.0f} ms | "
          f"path len {stats['path_len_mean']}")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Motion planner benchmark suite")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "free_space", "narrow_corridor",
                                 "cluttered", "dynamic"],
                        help="Which scenario to run")
    parser.add_argument("--trials",  type=int,  default=20, help="Trials per scenario")
    parser.add_argument("--iter",    type=int,  default=3000, help="RRT* iterations")
    parser.add_argument("--output",  type=str,  default=None,
                        help="Directory to save JSON results")
    args = parser.parse_args()

    SCENARIOS = {
        "free_space":      _free_space_obstacles(),
        "narrow_corridor": _narrow_corridor_obstacles(),
        "cluttered":       _cluttered_obstacles(),
    }

    to_run = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    print("\n" + "=" * 60)
    print("  Adaptive Motion Planner — Benchmark Suite")
    print("=" * 60)

    all_results = []
    for name in to_run:
        result = run_scenario(
            name=name,
            obstacles=SCENARIOS[name],
            n_trials=args.trials,
            max_iter=args.iter,
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print(f"  {'Scenario':<20} {'Success':>8} {'Mean ms':>10} {'Std ms':>8}")
    print("  " + "-" * 50)
    for r in all_results:
        print(f"  {r['scenario']:<20} {r['success_rate_%']:>7.1f}% "
              f"{r['time_mean_ms']:>9.1f}  {r['time_std_ms']:>7.1f}")
    print("=" * 60)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "benchmark_results.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    main()
