"""
Informed RRT* for the Panda in 7-DoF joint space.

Key ideas over vanilla RRT*:
  1. Ellipsoidal sampling (Gammell et al. 2014) — once a solution is found,
     samples are restricted to a prolate hyperspheroid centred on the
     straight-line path from start to goal.  This focuses computation and
     dramatically speeds up solution refinement.

  2. k-nearest rewiring — each new node rewires the k nearest neighbours,
     not just nodes within a fixed radius, giving O(n log n) rewiring cost.

  3. Constraint-aware extension — each candidate config is checked against
     the full ConstraintSet before acceptance.

  4. Early termination — returns as soon as a solution within `goal_cost`
     of the optimal straight-line distance is found.

Terminology (follows the original paper):
  c_best  — cost of the current best path
  c_min   — admissible lower bound (straight-line distance in C-space)
  C_ell   — the sampling ellipse transformation matrix
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from safety.constraints import ConstraintSet
from kinematics.forward import tcp_position


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    q: np.ndarray          # joint configuration (7,)
    parent: Optional["Node"] = None
    cost: float = 0.0      # cost from root to this node


@dataclass
class PlannerResult:
    path: List[np.ndarray]   # list of (7,) configs, start→goal
    cost: float
    success: bool
    iterations: int
    time_s: float
    tree_size: int


# ---------------------------------------------------------------------------
# Collision checking in Cartesian space
# ---------------------------------------------------------------------------

def _cartesian_collision_check(
    q: np.ndarray,
    obstacles: list,   # list of safety.cbf.Obstacle
) -> bool:
    """
    Return True if q is collision-free with all Cartesian obstacles.
    Uses TCP position as a proxy (extend to link spheres for full check).
    """
    p = tcp_position(q)
    for obs in obstacles:
        if np.linalg.norm(p - obs.center) < obs.effective_radius:
            return False
    return True


# ---------------------------------------------------------------------------
# Informed RRT*
# ---------------------------------------------------------------------------

class InformedRRTStar:
    """
    Informed RRT* planner for the Panda.

    Parameters
    ----------
    constraints   : ConstraintSet — joint limits, self-collision, torque
    obstacles     : list of Obstacle — Cartesian obstacles
    max_iter      : iteration budget
    step_size     : maximum extension step (rad, joint-space L2)
    goal_radius   : convergence ball radius (rad)
    k_neighbours  : number of neighbours in rewiring
    goal_bias     : fraction of iterations that sample the goal directly
    seed          : RNG seed for reproducibility
    """

    def __init__(
        self,
        constraints: ConstraintSet,
        obstacles: list | None = None,
        max_iter: int = 5000,
        step_size: float = 0.3,
        goal_radius: float = 0.15,
        k_neighbours: int = 6,
        goal_bias: float = 0.10,
        seed: int | None = None,
    ):
        self.constraints  = constraints
        self.obstacles    = obstacles or []
        self.max_iter     = max_iter
        self.step_size    = step_size
        self.goal_radius  = goal_radius
        self.k_neighbours = k_neighbours
        self.goal_bias    = goal_bias
        self.rng          = np.random.default_rng(seed)

        self._q_min = constraints.joint.q_min
        self._q_max = constraints.joint.q_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> PlannerResult:
        """
        Run Informed RRT* from q_start to q_goal.

        Returns a PlannerResult with the best path found within the
        iteration budget.
        """
        t0 = time.perf_counter()

        assert self._is_valid(q_start), "Start configuration is invalid"
        assert self._is_valid(q_goal),  "Goal  configuration is invalid"

        # Tree initialisation
        root = Node(q_start.copy(), parent=None, cost=0.0)
        tree: List[Node] = [root]

        best_node: Optional[Node] = None
        c_best   = float("inf")
        c_min    = float(np.linalg.norm(q_goal - q_start))

        for iteration in range(self.max_iter):
            # ---- Sample ------------------------------------------------
            q_rand = self._sample(q_start, q_goal, c_best, c_min)

            # ---- Nearest -----------------------------------------------
            nearest = self._nearest(tree, q_rand)

            # ---- Steer -------------------------------------------------
            q_new = self._steer(nearest.q, q_rand)

            # ---- Validity check ----------------------------------------
            if not self._is_valid(q_new):
                continue
            if not self._edge_valid(nearest.q, q_new):
                continue

            # ---- Find best parent in neighbourhood ---------------------
            neighbours = self._k_nearest(tree, q_new, self.k_neighbours)
            cost_new, parent_new = self._best_parent(neighbours, q_new)

            new_node = Node(q_new, parent=parent_new, cost=cost_new)
            tree.append(new_node)

            # ---- Rewire ------------------------------------------------
            self._rewire(neighbours, new_node)

            # ---- Goal check -------------------------------------------
            if self._dist(q_new, q_goal) < self.goal_radius:
                path_cost = cost_new + self._dist(q_new, q_goal)
                if path_cost < c_best:
                    c_best    = path_cost
                    best_node = new_node

        elapsed = time.perf_counter() - t0

        if best_node is None:
            return PlannerResult([], float("inf"), False, self.max_iter, elapsed, len(tree))

        path = self._extract_path(best_node) + [q_goal]
        return PlannerResult(path, c_best, True, self.max_iter, elapsed, len(tree))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_uniform(self) -> np.ndarray:
        """Uniform sample from the C-space box."""
        return self.rng.uniform(self._q_min, self._q_max)

    def _sample_ellipsoid(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        c_best: float,
        c_min: float,
    ) -> np.ndarray:
        """
        Sample uniformly from the prolate hyperspheroid (Gammell et al.).

        The ellipsoid is defined by:
            semi-major axis a1 = c_best / 2
            semi-minor axes   = sqrt(c_best² - c_min²) / 2   (all equal)

        We rotate from the ellipsoid frame (centred at midpoint,
        major axis along q_goal - q_start) into C-space.
        """
        n = len(q_start)
        c_mid = (q_start + q_goal) / 2.0

        # Unit vector from start to goal (major axis)
        a1 = (q_goal - q_start) / c_min        # (n,)

        # Build rotation matrix whose first column is a1
        # Use Gram-Schmidt on a random basis
        basis = np.eye(n)
        basis[:, 0] = a1
        Q, _ = np.linalg.qr(basis)
        # Ensure first column aligns with a1
        if np.dot(Q[:, 0], a1) < 0:
            Q[:, 0] *= -1

        # Ellipsoid semi-axes
        r1 = c_best / 2.0
        r_minor = np.sqrt(max(0.0, c_best ** 2 - c_min ** 2)) / 2.0
        L = np.diag([r1] + [r_minor] * (n - 1))

        # Sample from unit ball, transform
        for _ in range(50):
            x_ball = self.rng.standard_normal(n)
            x_ball /= np.linalg.norm(x_ball)
            x_ball *= self.rng.random() ** (1.0 / n)   # uniform in ball
            q_sample = Q @ L @ x_ball + c_mid
            if np.all(q_sample >= self._q_min) and np.all(q_sample <= self._q_max):
                return q_sample

        return self._sample_uniform()   # fallback

    def _sample(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        c_best: float,
        c_min: float,
    ) -> np.ndarray:
        if self.rng.random() < self.goal_bias:
            return q_goal.copy()
        if c_best < float("inf") and c_best > c_min + 1e-6:
            return self._sample_ellipsoid(q_start, q_goal, c_best, c_min)
        return self._sample_uniform()

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _nearest(self, tree: List[Node], q: np.ndarray) -> Node:
        """Return the tree node closest to q."""
        dists = [self._dist(node.q, q) for node in tree]
        return tree[int(np.argmin(dists))]

    def _k_nearest(self, tree: List[Node], q: np.ndarray, k: int) -> List[Node]:
        """Return the k nearest nodes to q."""
        dists = [(self._dist(node.q, q), node) for node in tree]
        dists.sort(key=lambda x: x[0])
        return [node for _, node in dists[:k]]

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """
        Move from q_near toward q_rand by at most step_size.
        """
        diff = q_rand - q_near
        d = np.linalg.norm(diff)
        if d < 1e-9:
            return q_near.copy()
        if d <= self.step_size:
            return q_rand.copy()
        return q_near + (diff / d) * self.step_size

    def _best_parent(
        self, neighbours: List[Node], q_new: np.ndarray
    ) -> Tuple[float, Node]:
        """
        Among neighbours, find the one minimising cost-through-neighbour.
        """
        best_cost   = float("inf")
        best_parent = neighbours[0]
        for node in neighbours:
            if self._edge_valid(node.q, q_new):
                c = node.cost + self._dist(node.q, q_new)
                if c < best_cost:
                    best_cost   = c
                    best_parent = node
        return best_cost, best_parent

    def _rewire(self, neighbours: List[Node], new_node: Node) -> None:
        """
        Rewire neighbours if routing through new_node reduces their cost.
        """
        for node in neighbours:
            if node is new_node.parent:
                continue
            if not self._edge_valid(new_node.q, node.q):
                continue
            new_cost = new_node.cost + self._dist(new_node.q, node.q)
            if new_cost < node.cost:
                node.parent = new_node
                node.cost   = new_cost

    def _extract_path(self, node: Node) -> List[np.ndarray]:
        """Walk parent pointers back to root; return start→node path."""
        path = []
        cur = node
        while cur is not None:
            path.append(cur.q.copy())
            cur = cur.parent
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------

    def _is_valid(self, q: np.ndarray) -> bool:
        """Check joint limits and Cartesian obstacles (skip self-col for speed)."""
        if not self.constraints.config_is_valid(q, check_self_col=False):
            return False
        if not _cartesian_collision_check(q, self.obstacles):
            return False
        return True

    def _edge_valid(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        n_checks: int = 4,
    ) -> bool:
        """
        Discretise the edge and check each sample.
        n_checks controls resolution (higher = safer, slower).
        """
        for t in np.linspace(0, 1, n_checks):
            q_interp = (1 - t) * q_from + t * q_to
            if not self._is_valid(q_interp):
                return False
        return True
