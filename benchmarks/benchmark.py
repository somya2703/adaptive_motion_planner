"""
benchmarks/benchmark.py
-----------------------
Standard benchmark suite for the Adaptive Motion Planner.

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
from planner.rrt_star import InformedRRTStar, _cartesian_collision_check
from robot.panda import PANDA


# ---------------------------------------------------------------------------
# Standard obstacle layouts
# ---------------------------------------------------------------------------

def _free_space_obstacles() -> List[Obstacle]:
    return []


def _narrow_corridor_obstacles() -> List[Obstacle]:
    return [
        Obstacle(center=np.array([0.5,  0.3, 0.4]), radius=0.20, margin=0.04),
        Obstacle(center=np.array([0.5, -0.3, 0.4]), radius=0.20, margin=0.04),
    ]


def _cluttered_obstacles(seed: int = 0) -> List[Obstacle]:
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(8):
        centre = rng.uniform([0.2, -0.5, 0.1], [0.7, 0.5, 0.7])
        radius = rng.uniform(0.05, 0.12)
        obs.append(Obstacle(center=centre, radius=radius, margin=0.03))
    return obs


def _goal_pairs(
    n: int,
    obstacles: List[Obstacle],
    seed: int = 0,
    min_dist: float = 0.5,
) -> List[tuple]:
    """
    Generate n random (start, goal) pairs that are valid for the given
    obstacle layout — both joint constraints and Cartesian collision free.
    """
    rng = np.random.default_rng(seed)
    constraints = ConstraintSet()
    pairs = []
    max_attempts = n * 500
    attempts = 0
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        q_s = PANDA.random_config(rng)
        q_g = PANDA.random_config(rng)
        if not constraints.config_is_valid(q_s, check_self_col=False):
            continue
        if not constraints.config_is_valid(q_g, check_self_col=False):
            continue
        if np.linalg.norm(q_g - q_s) < min_dist:
            continue
        # Check that start and goal TCP are not inside any obstacle
        if obstacles and not _cartesian_collision_check(q_s, obstacles):
            continue
        if obstacles and not _cartesian_collision_check(q_g, obstacles):
            continue
        pairs.append((q_s, q_g))

    if len(pairs) < n:
        print("  WARNING: only found %d/%d valid pairs after %d attempts" % (
            len(pairs), n, max_attempts))
    return pairs


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    obstacles: List[Obstacle],
    n_trials: int = 20,
    max_iter: int = 3000,
    seed: int = 42,
) -> Dict:
    constraints = ConstraintSet()
    pairs = _goal_pairs(n_trials, obstacles=obstacles, seed=seed)

    times_ms: List[float] = []
    lengths:  List[float] = []
    tree_sizes: List[int] = []
    successes = 0

    print("\n  Scenario: %s | %d trials | %d obstacles" % (name, n_trials, len(obstacles)))
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
        line = "  [%3d/%d] %s  %7.1f ms" % (i+1, n_trials, status, elapsed_ms)
        if result.success:
            line += "  len=%.3f" % lengths[-1]
        print(line)

    success_rate = successes / max(len(pairs), 1) * 100
    stats = {
        "scenario":       name,
        "success_rate_%": round(success_rate, 1),
        "time_mean_ms":   round(float(np.mean(times_ms)), 1) if times_ms else 0,
        "time_std_ms":    round(float(np.std(times_ms)),  1) if times_ms else 0,
        "time_min_ms":    round(float(np.min(times_ms)),  1) if times_ms else 0,
        "time_max_ms":    round(float(np.max(times_ms)),  1) if times_ms else 0,
        "path_len_mean":  round(float(np.mean(lengths)),  4) if lengths else None,
        "tree_size_mean": round(float(np.mean(tree_sizes)))   if tree_sizes else None,
    }

    print("\n  Result: %.0f%% success | mean %.0f ms | path len %s" % (
        success_rate,
        stats["time_mean_ms"],
        ("%.4f" % stats["path_len_mean"]) if stats["path_len_mean"] else "n/a"))

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Motion planner benchmark suite")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "free_space", "narrow_corridor", "cluttered"],
                        help="Which scenario to run")
    parser.add_argument("--trials",  type=int, default=20)
    parser.add_argument("--iter",    type=int, default=3000)
    parser.add_argument("--output",  type=str, default=None)
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

    print("\n" + "=" * 60)
    print("  %-20s %8s %10s %8s" % ("Scenario", "Success", "Mean ms", "Std ms"))
    print("  " + "-" * 50)
    for r in all_results:
        print("  %-20s %7.1f%% %9.1f  %7.1f" % (
            r["scenario"], r["success_rate_%"],
            r["time_mean_ms"], r["time_std_ms"]))
    print("=" * 60)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "benchmarks.json"
        with open(str(out_file), "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n  Results saved to %s" % out_file)


if __name__ == "__main__":
    main()
