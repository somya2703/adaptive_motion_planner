"""
pipeline.py
-----------
End-to-end pipeline for the Adaptive Motion Planner.

Runs inside Docker. Produces artifacts in /app/results/:

  results/
  ├── summary.json
  ├── test_report.xml       JUnit XML (GitHub Actions reads this)
  ├── benchmarks.json
  ├── benchmarks.png
  ├── plans/                3D TCP trace per scene
  ├── trajectories/         Joint pos/vel/accel per scene
  └── cbf/                  CBF clearance h(q) per scene

Usage:
    python pipeline.py                 # full pipeline
    python pipeline.py --quick         # skip multi-trial benchmarks
    python pipeline.py --scene cluttered
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot.panda import PANDA
from safety.constraints import ConstraintSet
from safety.cbf import CBFSafetyFilter, Obstacle
from planner.rrt_star import InformedRRTStar
from planner.trajectory import build_trajectory
from kinematics.forward import tcp_position, link_positions

RESULTS = Path("/app/results")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dir(p: Path) -> Path:
    """Create directory, tolerating pre-existing host-owned dirs."""
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if not p.is_dir():
            print(f"ERROR: cannot create {p}")
            print("Run on host first:  mkdir -p results/plans results/trajectories results/cbf")
            sys.exit(1)
    return p


STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
}


# ─────────────────────────────────────────────────────────────────────────────
# Scene definitions
# ─────────────────────────────────────────────────────────────────────────────

Q_START = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
Q_GOAL  = np.array([0.749, 0.05, -0.018, -2.329, -2.829, 0.708, 1.113])

SCENES = {
    "free_space": {
        "label":     "Free space",
        "obstacles": [],
        "color":     "#2196F3",
    },
    "narrow_corridor": {
        "label": "Narrow corridor",
        "obstacles": [
            Obstacle(center=np.array([0.50,  0.30, 0.35]), radius=0.16, margin=0.04),
            Obstacle(center=np.array([0.50, -0.30, 0.35]), radius=0.16, margin=0.04),
        ],
        "color": "#FF9800",
    },
    "cluttered": {
        "label": "Cluttered (4 obs.)",
        "obstacles": [
            Obstacle(center=np.array([0.45,  0.20, 0.40]), radius=0.10, margin=0.04),
            Obstacle(center=np.array([0.50, -0.15, 0.35]), radius=0.08, margin=0.04),
            Obstacle(center=np.array([0.40,  0.00, 0.65]), radius=0.09, margin=0.04),
            Obstacle(center=np.array([0.60,  0.25, 0.25]), radius=0.07, margin=0.04),
        ],
        "color": "#9C27B0",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — pytest
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(results_dir: Path) -> dict:
    print("\n" + "="*60)
    print("  STEP 1 — Unit tests")
    print("="*60)
    xml_out = results_dir / "test_report.xml"
    cmd = [
        "python", "-m", "pytest", "tests/", "-v",
        "--junitxml=" + str(xml_out),
        "--tb=short",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    passed = (proc.returncode == 0)
    print("\n  Tests %s in %.1fs" % ("PASSED" if passed else "FAILED", elapsed))
    return {"tests_passed": passed, "test_time_s": round(elapsed, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Plan
# ─────────────────────────────────────────────────────────────────────────────

def plan_scene(name: str, scene: dict, max_iter: int = 2000, seed: int = 42):
    constraints = ConstraintSet()
    planner = InformedRRTStar(
        constraints=constraints,
        obstacles=scene["obstacles"],
        max_iter=max_iter,
        seed=seed,
    )
    t0 = time.perf_counter()
    result = planner.plan(Q_START, Q_GOAL)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    traj = build_trajectory(result.path, dt=0.005) if result.success else None
    return result, traj, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_path_3d(name: str, scene: dict, result, out_dir: Path):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection="3d")

        for obs in scene["obstacles"]:
            u = np.linspace(0, 2*np.pi, 24)
            v = np.linspace(0, np.pi, 12)
            r  = obs.effective_radius
            cx, cy, cz = obs.center
            xs = cx + r * np.outer(np.cos(u), np.sin(v))
            ys = cy + r * np.outer(np.sin(u), np.sin(v))
            zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, color="#ef5350", alpha=0.18, linewidth=0)

        if result.success:
            tcp_pts = np.array([tcp_position(q) for q in result.path])
            ax.plot(tcp_pts[:, 0], tcp_pts[:, 1], tcp_pts[:, 2],
                    "-o", color=scene["color"], linewidth=2.5,
                    markersize=4, label="TCP path")

            def draw_arm(q, color, alpha=0.7, lw=3):
                pts = np.vstack([np.zeros((1, 3)), link_positions(q)])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        "o-", color=color, linewidth=lw, alpha=alpha,
                        markersize=3, markerfacecolor="white")

            draw_arm(Q_START, "#4CAF50")
            draw_arm(result.path[len(result.path)//2], "#90CAF9", alpha=0.45, lw=2)
            draw_arm(Q_GOAL, "#FF9800")
            ax.scatter(*tcp_position(Q_START), s=80, color="#4CAF50", zorder=10, label="Start TCP")
            ax.scatter(*tcp_position(Q_GOAL),  s=80, color="#FF9800", zorder=10, label="Goal TCP")

            all_pts = np.vstack([tcp_pts,
                                 tcp_position(Q_START).reshape(1, 3),
                                 tcp_position(Q_GOAL).reshape(1, 3)])
            mid = (all_pts.max(0) + all_pts.min(0)) / 2
            r   = max((all_pts.max(0) - all_pts.min(0)).max() / 2, 0.4)
            ax.set_xlim(mid[0]-r, mid[0]+r)
            ax.set_ylim(mid[1]-r, mid[1]+r)
            ax.set_zlim(max(0, mid[2]-r), mid[2]+r)

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(scene["label"] + " — planned TCP path")
        ax.legend(fontsize=9, loc="upper left")
        path = out_dir / (name + "_path.png")
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        print("  Saved " + path.name)


def plot_trajectory(name: str, scene: dict, traj, out_dir: Path):
    if traj is None:
        return
    t   = np.array([pt.t   for pt in traj])
    q   = np.array([pt.q   for pt in traj])
    qd  = np.array([pt.qd  for pt in traj])
    qdd = np.array([pt.qdd for pt in traj])

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 0.85, 7))
        for j in range(7):
            axes[0].plot(t, np.degrees(q[:, j]),   color=colors[j], label="J%d"%(j+1), linewidth=1.2)
            axes[1].plot(t, np.degrees(qd[:, j]),  color=colors[j], linewidth=1.2)
            axes[2].plot(t, np.degrees(qdd[:, j]), color=colors[j], linewidth=1.0, alpha=0.85)
        for j in range(7):
            axes[0].axhline(np.degrees(PANDA.q_max[j]), color=colors[j], linewidth=0.4, linestyle="--", alpha=0.4)
            axes[0].axhline(np.degrees(PANDA.q_min[j]), color=colors[j], linewidth=0.4, linestyle="--", alpha=0.4)
        axes[0].set_ylabel("Position (deg)")
        axes[1].set_ylabel("Velocity (deg/s)")
        axes[2].set_ylabel("Accel. (deg/s^2)")
        axes[2].set_xlabel("Time (s)")
        axes[0].set_title(scene["label"] + " — joint trajectory")
        axes[0].legend(ncol=7, fontsize=8, loc="upper right")
        fig.tight_layout()
        path = out_dir / (name + "_traj.png")
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        print("  Saved " + path.name)


def plot_cbf_clearance(name: str, scene: dict, result, traj, cbf_dir: Path):
    if traj is None or not scene["obstacles"]:
        return
    cbf    = CBFSafetyFilter(obstacles=scene["obstacles"])
    step   = max(1, len(traj)//200)
    pts    = traj[::step]
    times  = np.array([pt.t for pt in pts])
    clears = [cbf.min_clearance(pt.q) for pt in pts]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, clears, color=scene["color"], linewidth=1.8)
        ax.axhline(0, color="#e53935", linewidth=1.2, linestyle="--", label="Safety boundary h=0")
        ax.fill_between(times, clears, 0,
                         where=np.array(clears) > 0,
                         alpha=0.15, color=scene["color"], label="Safe margin")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CBF clearance h(q)")
        ax.set_title(scene["label"] + " — Control Barrier Function clearance")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = cbf_dir / (name + "_cbf.png")
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        print("  Saved " + path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks(results_dir: Path, n_trials: int = 8) -> list:
    print("\n" + "="*60)
    print("  STEP 4 — Benchmarks (%d trials per scene)" % n_trials)
    print("="*60)
    constraints = ConstraintSet()
    all_stats = []

    for scene_name, scene in SCENES.items():
        times_ms, lengths, successes = [], [], []
        for trial in range(n_trials):
            planner = InformedRRTStar(
                constraints=constraints,
                obstacles=scene["obstacles"],
                max_iter=2000,
                seed=trial,
            )
            t0 = time.perf_counter()
            res = planner.plan(Q_START, Q_GOAL)
            elapsed = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed)
            successes.append(res.success)
            if res.success:
                arr = np.array(res.path)
                lengths.append(float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))))
            print("  %-22s trial %2d/%d  %s  %7.0f ms" % (
                scene["label"], trial+1, n_trials,
                "OK" if res.success else "--", elapsed))

        sr = sum(successes) / n_trials * 100
        stats = {
            "scene":         scene_name,
            "label":         scene["label"],
            "success_rate":  round(sr, 1),
            "time_mean_ms":  round(float(np.mean(times_ms)), 0),
            "time_std_ms":   round(float(np.std(times_ms)),  0),
            "path_len_mean": round(float(np.mean(lengths)), 4) if lengths else None,
        }
        all_stats.append(stats)
        print("  -> %.0f%% success  mean %.0f ms  std %.0f ms\n" % (
            sr, stats["time_mean_ms"], stats["time_std_ms"]))

    bench_json = results_dir / "benchmarks.json"
    with open(str(bench_json), "w") as f:
        json.dump(all_stats, f, indent=2)
    print("  Saved " + bench_json.name)
    return all_stats


def plot_benchmarks(stats: list, results_dir: Path):
    labels = [s["label"]        for s in stats]
    means  = [s["time_mean_ms"] for s in stats]
    stds   = [s["time_std_ms"]  for s in stats]
    rates  = [s["success_rate"] for s in stats]
    colors = [SCENES[s["scene"]]["color"] for s in stats]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        bars = ax1.bar(labels, means, yerr=stds, capsize=6,
                       color=colors, alpha=0.82, error_kw={"linewidth": 1.5})
        for bar, m, s in zip(bars, means, stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 50,
                     "%.0f ms" % m, ha="center", va="bottom", fontsize=9)
        ax1.set_ylabel("Plan time (ms)")
        ax1.set_title("Planning time by scene")
        ax1.set_ylim(0, max(means) * 1.35)

        bars2 = ax2.bar(labels, rates, color=colors, alpha=0.82)
        for bar, r in zip(bars2, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     "%.0f%%" % r, ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Success rate (%)")
        ax2.set_title("Success rate by scene")
        ax2.set_ylim(0, 115)
        ax2.axhline(100, color="#ccc", linewidth=0.8, linestyle="--")

        fig.suptitle("Informed RRT* — Franka Panda 7-DoF benchmark", fontsize=13)
        fig.tight_layout()
        path = results_dir / "benchmarks.png"
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        print("  Saved " + path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — RESULTS.md
# ─────────────────────────────────────────────────────────────────────────────

def write_results_md(summary: dict, stats: list, results_dir: Path):
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines = [
        "# Pipeline results",
        "",
        "Generated by `pipeline.py` · " + ts,
        "",
        "## Test suite",
        "",
        "| Status | Time |",
        "|--------|------|",
        "| %s | %.1f s |" % (
            "PASSED" if summary["tests_passed"] else "FAILED",
            summary["test_time_s"]),
        "",
        "## Benchmark results",
        "",
        "| Scene | Success | Mean time | Std | Path length |",
        "|-------|---------|-----------|-----|-------------|",
    ]
    for s in stats:
        pl = ("%.3f rad" % s["path_len_mean"]) if s["path_len_mean"] else "n/a"
        lines.append("| %s | %.0f%% | %.0f ms | +/-%.0f ms | %s |" % (
            s["label"], s["success_rate"],
            s["time_mean_ms"], s["time_std_ms"], pl))

    lines += [
        "",
        "## Visualisations",
        "",
        "### Planned paths",
        "",
        "| Free space | Narrow corridor | Cluttered |",
        "|---|---|---|",
        "| ![free](plans/free_space_path.png) | ![corridor](plans/narrow_corridor_path.png) | ![cluttered](plans/cluttered_path.png) |",
        "",
        "### Benchmark chart",
        "",
        "![benchmarks](benchmarks.png)",
        "",
        "### Trajectories",
        "",
        "| Free space | Narrow corridor | Cluttered |",
        "|---|---|---|",
        "| ![free](trajectories/free_space_traj.png) | ![corridor](trajectories/narrow_corridor_traj.png) | ![cluttered](trajectories/cluttered_traj.png) |",
        "",
        "### CBF safety margin",
        "",
        "| Narrow corridor | Cluttered |",
        "|---|---|",
        "| ![cbf_corridor](cbf/narrow_corridor_cbf.png) | ![cbf_cluttered](cbf/cluttered_cbf.png) |",
        "",
    ]
    md_path = results_dir / "RESULTS.md"
    md_path.write_text("\n".join(lines))
    print("  Saved " + md_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true", help="Skip multi-trial benchmarks")
    parser.add_argument("--scene",  default=None, choices=list(SCENES.keys()))
    parser.add_argument("--iter",   type=int, default=2000)
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    plans_dir = make_dir(RESULTS / "plans")
    traj_dir  = make_dir(RESULTS / "trajectories")
    cbf_dir   = make_dir(RESULTS / "cbf")

    summary = {}

    # Step 1 — tests
    summary.update(run_tests(RESULTS))
    if not summary["tests_passed"]:
        print("\n  Tests failed — aborting.")
        sys.exit(1)

    # Steps 2+3 — plan + visualise
    scenes_to_run = [args.scene] if args.scene else list(SCENES.keys())
    plan_results  = {}

    print("\n" + "="*60)
    print("  STEP 2+3 — Planning + visualisation")
    print("="*60)

    for name in scenes_to_run:
        scene = SCENES[name]
        print("\n  Scene: " + scene["label"])
        result, traj, elapsed_ms = plan_scene(name, scene, max_iter=args.iter)
        if result.success:
            print("  SUCCESS  %.0f ms  tree=%d  waypoints=%d" % (
                elapsed_ms, result.tree_size, len(result.path)))
        else:
            print("  FAILED  %.0f ms" % elapsed_ms)

        plan_results[name] = {
            "success":    result.success,
            "elapsed_ms": round(elapsed_ms, 1),
            "tree_size":  result.tree_size,
            "n_waypoints": len(result.path),
            "cost":       round(result.cost, 4) if result.success else None,
        }

        plot_path_3d(name, scene, result, plans_dir)
        plot_trajectory(name, scene, traj, traj_dir)
        plot_cbf_clearance(name, scene, result, traj, cbf_dir)

    summary["plans"] = plan_results

    # Step 4 — benchmarks
    stats = []
    if not args.quick:
        stats = run_benchmarks(RESULTS, n_trials=args.trials)
        print("\n" + "="*60)
        print("  STEP 4b — Benchmark chart")
        print("="*60)
        plot_benchmarks(stats, RESULTS)
    else:
        print("\n  Skipping multi-trial benchmarks (--quick)")

    # Step 5 — summary files
    print("\n" + "="*60)
    print("  STEP 5 — Writing summary")
    print("="*60)
    summary_path = RESULTS / "summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved summary.json")
    if stats:
        write_results_md(summary, stats, RESULTS)

    # Final report
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print("  Tests: %s" % ("PASSED" if summary["tests_passed"] else "FAILED"))
    for name, r in plan_results.items():
        mark = "+" if r["success"] else "x"
        print("  %s %-24s %.0f ms" % (mark, SCENES[name]["label"], r["elapsed_ms"]))
    print("\n  Results written to: " + str(RESULTS))
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
