# Mathematical Foundations

*This document derives the key equations implemented in the planner.*

---

## 1. Forward Kinematics — Modified DH Convention

Each joint `i` has four parameters: `(a_{i-1}, d_i, α_{i-1}, θ_i)`.

The homogeneous transform from frame `i-1` to frame `i` is:

```
T_{i-1}^{i} = Rx(α_{i-1}) · Tx(a_{i-1}) · Rz(θ_i) · Tz(d_i)

         ⎡ cθᵢ   -sθᵢ    0     aᵢ₋₁  ⎤
       = ⎢ sθᵢcα  cθᵢcα  -sα  -sα·dᵢ ⎥
         ⎢ sθᵢsα  cθᵢsα   cα   cα·dᵢ ⎥
         ⎣  0      0       0     1   ⎦
```

where `cθ = cos(θ)`, `sθ = sin(θ)`, `cα = cos(α)`, `sα = sin(α)`.

The full FK is the product of all 7 transforms:

```
T_base^tcp = T_0^1(θ₁) · T_1^2(θ₂) · ... · T_6^7(θ₇) · T_tcp_offset
```

The TCP offset `p_off = [0, 0, 0.10]^T` in joint-7 frame gives:

```
p_tcp = R_0^7 · p_off + p_0^7
```

---

## 2. Geometric Jacobian

For a revolute joint `i`, the column of the geometric Jacobian is:

```
Jᵢ = ⎡ zᵢ₋₁ × (p_tcp − pᵢ₋₁) ⎤   (linear velocity)
     ⎣         zᵢ₋₁          ⎦   (angular velocity)
```

where:
- `zᵢ₋₁` = z-axis of frame `i-1` expressed in base frame = `T_0^{i-1}[:3, 2]`
- `pᵢ₋₁` = origin of frame `i-1` in base frame = `T_0^{i-1}[:3, 3]`

The full 6×7 Jacobian:

```
J(q) = [J₁ | J₂ | ... | J₇]
```

maps joint velocities to TCP twist:

```
ξ_tcp = [v_tcp; ω_tcp] = J(q) · q̇       ∈ R^6
```

### Numerical Validation

We validate against finite differences:

```
∂p_tcp / ∂qᵢ ≈ (FK(q + εeᵢ)[:3,3] − FK(q)[:3,3]) / ε
```

Test shows max error < 1e-4 m/rad for ε = 1e-6 (see `test_kinematics.py`).

---

## 3. Damped Pseudoinverse

The standard Moore-Penrose pseudoinverse `J† = Jᵀ(JJᵀ)⁻¹` blows up
near singularities (where `det(JJᵀ) → 0`).

The **damped pseudoinverse** regularises this:

```
J†_λ = Jᵀ (JJᵀ + λ²I)⁻¹
```

- Far from singularities (λ ≪ σ_min): `J†_λ ≈ J†` (exact)
- Near singularities: λ² dominates, magnification is bounded by `1/(2λ)`

We use **adaptive damping** based on the Yoshikawa manipulability measure:

```
w(q) = √(det(J(q)Jᵀ(q)))
```

When `w < w_threshold`:
```
λ(w) = λ_max · (1 − w/w_threshold)²
```

This smoothly increases damping as the arm approaches a singularity.

---

## 4. Redundancy Resolution — Nullspace

The Panda has 7 DoF for a 6D task → 1-dimensional nullspace.

The general IK velocity solution is:

```
q̇ = J†_λ · ẋ_e + (I − J†_λ J) · q̇_0
```

- **First term** (task-space component): achieves the desired TCP motion
- **Second term:** (nullspace component): moves joints without moving TCP

We choose `q̇_0 = k · ∇H(q)` where `H` is the joint-midpoint cost:

```
H(q) = ½ Σᵢ ((qᵢ − q_mid,i) / (q_max,i − q_min,i))²

∇H(q) = −(q − q_mid) / (q_max − q_min)²    (element-wise)
```

This gradient descent in the nullspace continuously pulls all joints
toward their midpoints, maximising manipulability and avoiding limits.

---

## 5. Informed RRT*

### RRT* (Karaman & Frazzoli, 2011)

RRT* guarantees **asymptotic optimality**: as `n → ∞`, the cost of the
returned path converges to the optimal cost almost surely.

Key addition over RRT: **rewiring:** after inserting a new node `x_new`,
check all nodes `x_near` in a ball of radius `r_n`:

```
r_n = γ · (log(n)/n)^(1/d)
```

If `cost(x_new) + dist(x_new, x_near) < cost(x_near)`, reroute through
`x_new`. This is what makes the tree asymptotically optimal.

We use **k-nearest rewiring** (k = 10) instead of a fixed ball, which
gives O(n log n) total rewiring cost.

### Informed RRT* (Gammell et al., 2014)

Once a solution of cost `c_best` is found, the set of configurations
that can possibly improve it is:

```
X_f = { x ∈ X | g*(x_start, x) + g*(x, x_goal) < c_best }
```

where `g*` is the true optimal cost-to-go. Using the Euclidean lower
bound i.e straightline distance, this becomes a **prolate hyperspheroid**:

```
X_ell = { x | dist(x, x_start) + dist(x, x_goal) < c_best }
```

The hyperspheroid has:
- Centre: `x_c = (x_start + x_goal) / 2`
- Major semi-axis: `a₁ = c_best / 2`  (along the start to goal axis)
- Minor semi-axes: `aᵢ = √(c_best² − c_min²) / 2`  (i ≥ 2)

Sampling uniformly in this ellipsoid, then applying rotation matrix `C`
(first column = unit vector from start to goal):

```
x_sample = C · L · x_ball + x_c

where L = diag(a₁, a₂, ..., aₙ)
and   x_ball is a uniform sample from the unit n-ball
```

This focuses computation on the **relevant region** of C-space and
drastically improves convergence speed in narrow-passage problems.

---

## 6. Control Barrier Functions

### Definition

A **Control Barrier Function** `h: X → R` defines a safe set:

```
S = { x ∈ X | h(x) ≥ 0 }
```

`S` is **forward invariant** (once in, stays in) if:

```
ḣ(x, u) + α(h(x)) ≥ 0    ∀ x ∈ S, u ∈ U
```

where `α` is a class-K function. We use `α(s) = k·s`, k > 0.

### Obstacle Avoidance CBF

For a spherical obstacle at `p_obs` with effective radius `r_eff`:

```
h_obs(q) = ‖p_tcp(q) − p_obs‖² − r_eff²
```

`h_obs ≥ 0` means the TCP is outside the obstacle.

The time derivative:

```
ḣ_obs = ∂h/∂q · q̇ = 2(p_tcp − p_obs)ᵀ · J_pos(q) · q̇
```

So `L_g h = 2(p_tcp − p_obs)ᵀ J_pos(q)` — a 1×7 row vector.

### Joint Limit CBFs

For joint `k`:

```
h_k^low(q) = qₖ − q_min,k ≥ 0
h_k^up(q)  = q_max,k − qₖ ≥ 0
```

Time derivatives: `ḣ_k^low = q̇ₖ`, `ḣ_k^up = −q̇ₖ`

### Safety Filter QP

At each timestep, given a desired torque command `τ_des`:

```
min_τ  ‖τ − τ_des‖²

s.t.  ∀ obstacle i:    L_g h_i · τ + L_f h_i + αᵢ · hᵢ ≥ 0
      ∀ joint k (low):  τₖ + L_f h_k^low + α_jnt · h_k^low ≥ 0
      ∀ joint k (up):  −τₖ + L_f h_k^up  + α_jnt · h_k^up  ≥ 0
      τ_min ≤ τ ≤ τ_max
```

This is a Quadratic Program. For real time use (1 kHz), we implement
a gradient projection approximation that handles each constraint
independently, which is sub-optimal but runs in microseconds.

**Safety guarantee**: The CBF condition is a sufficient condition for
set invariance. If `h(x₀) ≥ 0` at time 0 and the QP is feasible at
every timestep, then `h(x(t)) ≥ 0` for all `t ≥ 0` .
The arm can never collide with the obstacle.

---

## 7. B-Spline Trajectory Smoothing

Raw RRT* paths are piecewise-linear in joint space, producing
discontinuous velocity commands. We can make it smooth with a **cubic B-spline**:

```
p(u) = Σᵢ Nᵢ,₃(u) · pᵢ     u ∈ [0, 1]
```

where `Nᵢ,₃` are cubic B-spline basis functions (C2 continuous).
Using `scipy.interpolate.splprep` with `s = 0` (interpolation mode).

### Time Scaling — Trapezoidal Profile

Given a smooth path `q(s)` parametrised by arc length `s ∈ [0, L]`,
we assign time via a trapezoidal velocity profile:

```
Phase 1 (accelerate): q̈ = a_max,   0 ≤ t < t_a
Phase 2 (cruise):     q̈ = 0,       t_a ≤ t < T − t_a
Phase 3 (decelerate): q̈ = −a_max,  T − t_a ≤ t ≤ T
```

The accel duration `t_a = v_peak / a_max` and total time `T` are
chosen so that no joint exceeds its velocity or acceleration limit.
The most constrained joint determines the global time scale.

---

## References

1. Karaman, S. & Frazzoli, E. (2011). *Sampling-based algorithms for optimal motion planning*. IJRR 30(7).
2. Gammell, J.D., Srinivasa, S.S. & Barfoot, T.D. (2014). *Informed RRT*: Optimal sampling-based path planning focused via direct sampling of an admissible ellipsoidal heuristic*. IROS.
3. Ames, A.D. et al. (2019). *Control Barrier Functions: Theory and Applications*. ECC.
4. Nakamura, Y. (1991). *Advanced Robotics: Redundancy and Optimization*. Addison-Wesley.
5. Siciliano, B. et al. (2009). *Robotics: Modelling, Planning and Control*. Springer.
