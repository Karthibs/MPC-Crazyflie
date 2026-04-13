# MPC Drone Flight Controller

A Model Predictive Control (MPC) framework for quadrotor flight simulation, built on top of MuJoCo. The project implements multiple MPC variants — from unconstrained batch MPC to constrained offset-free MPC with Kalman disturbance estimation — all applied to a 12-state linearized hover model of a Crazyflie-style quadrotor.

---

## Overview

The controller linearizes the quadrotor dynamics around hover, discretizes using zero-order hold (matrix exponential), and solves a receding-horizon optimal control problem at each timestep. Several progressively more capable controller architectures are implemented and compared.

---

## Repository Structure

```text
.
├── MPC/
│   ├── BatchMpc.py                  # Unconstrained batch MPC + Kalman variant
│   ├── ConstrainedBatchMPC.py       # Constrained batch MPC (QP-based) + Kalman variant
│   ├── DyPMPC.py                    # Dynamic Programming MPC + Kalman variant
│   ├── drone_environment.py         # MuJoCo quadrotor environment wrapper
│   ├── drone_params.py              # Physical params, linearization, state-space helpers
│   └── chaos_wind_generator.py      # Randomized wind/torque disturbance injection
├── Drone_xml/
│   ├── drone.xml                    # Crazyflie quadrotor MuJoCo model
│   └── scene.xml                    # Scene with ground plane and lighting
├── CstMPC_simple.py                 # Constrained batch MPC (no tracking, manual augmentation)
├── CstMPC_Reftrack.py               # Constrained batch MPC with integral reference tracking
├── CstMPC_Kalman_Reftrack.py        # Constrained offset-free MPC with Kalman filter
├── UnCst_Batch_droneRefTrack.py     # Unconstrained batch MPC with reference tracking
├── UnCst_Dy_KalmanRefTrack.py       # Unconstrained DP MPC with Kalman filter
└── images/                          # Saved simulation plots
```

---

## Drone Model

The simulated drone is a Crazyflie 2.x quadrotor modelled in MuJoCo with:

- **Mass:** 27 g
- **Inertia:** Ixx = Iyy = 2.3951×10⁻⁵ kg·m², Izz = 3.2347×10⁻⁵ kg·m²
- **Actuators:** body thrust (N), roll torque (Nm), pitch torque (Nm), yaw torque (Nm)
- **Sensors:** gyroscope, accelerometer, frame quaternion

The **12-state** linearized model (hover operating point) has the state vector:

```
x = [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
```

and control input:

```
u = [ΔThrust, τ_roll, τ_pitch, τ_yaw]
```

The continuous-time A and B matrices are discretized exactly via the matrix exponential.

---

## Controller Architectures

### 1. Unconstrained Batch MPC (`BatchMpc.py`)
Pre-computes the full receding-horizon gain matrix **K** offline by solving the unconstrained QP analytically:

```
min  X^T Q̄ X + U^T R̄ U    s.t.   X = Ω x₀ + Γ U
```

The closed-form solution `K = -(Γᵀ Q̄ Γ + R̄)⁻¹ Γᵀ Q̄ Ω` is computed once at initialization.

### 2. Batch MPC with Integral Reference Tracking (`BatchMpc.py` — `BatchMPCReferencetracking`)
Augments the state with output-error integrators to achieve zero steady-state error:

```
z_{k+1} = z_k + dt · C_track · x̃_k
x_aug   = [x̃; z]
```

Tracks selectable outputs (e.g. `z`, `roll`, `pitch`, `yaw`) by penalizing both tracking error and integral windup.

### 3. Dynamic Programming MPC (`DyPMPC.py`)
Solves the finite-horizon LQR backward Riccati recursion online. Equivalent gain to Batch MPC but demonstrates the DP formulation. Also includes a Kalman + offset-free variant (`DyPMPCKalmanReferenceTracking`).

### 4. Constrained Batch MPC (`ConstrainedBatchMPC.py`)
Solves the QP **online** at each timestep using [qpsolvers](https://github.com/qpsolvers/qpsolvers) (OSQP backend), enabling hard state and input constraints:

```
min  ½ Uᵀ P U + qᵀ U
s.t. A_ineq U ≤ b₀ + E x₀
```

Constraint matrices for state and input box constraints are assembled via condensed prediction:

```
Y = C̄(Ω x₀ + Γ U)   →   A_ineq U ≤ b + E x₀
```

### 5. Constrained MPC with Integral Reference Tracking (`ConstrainedBatchMPC.py` — `ConstrainedBatchMPCReferenceTracking`)
Combines the QP-based constrained MPC with the integral-augmented error-state formulation. Physical state bounds are automatically shifted to error-state bounds relative to the current reference `x_ref`.

### 6. Constrained Offset-Free MPC with Kalman Filter (`ConstrainedBatchMPC.py` — `ConstrainedBatchMPCKalmanReferenceTracking`)
The most complete controller. Adds a **steady-state Kalman filter** for disturbance estimation to achieve offset-free tracking under persistent unmeasured disturbances:

```
x_{k+1} = A x_k + B u_k + B_d d_k
d_{k+1} = d_k                          (constant disturbance model)
y_k     = C_y x_k + v_k
```

At each step:
1. **Kalman correction** — update `[x̂; d̂]` from measurement `y_k`
2. **Steady-state target** — solve `[A-I, B; H, 0] [xₛ; uₛ] = [-B_d d̂; r]`
3. **Constrained QP** — solve for `ũ = u - uₛ` on deviation dynamics `x̃ = x - xₛ`
4. **Apply** `u = uₛ + ũ`, then Kalman prediction step

---

## State & Input Constraints

Constraints are built as condensed prediction-form inequalities and stacked before each QP solve:

| Constraint | Typical Values |
|---|---|
| Altitude `z` | 0.10 m – 0.45 m |
| Roll angle | ±20° |
| Pitch angle | ±20° |
| Thrust deviation | `[0 – 2·hover_force] - hover_force` |
| Roll torque | ±2×10⁻³ Nm |
| Pitch torque | ±2×10⁻³ Nm |
| Yaw torque | ±1×10⁻³ Nm |

---

## Cost Tuning

The cost matrices penalize deviations across the 12-state vector and 4 inputs:

```python
# State penalty (diagonal Q)
Q_x = [2e-1, 2e-1, 5e3,    # position (x, y, z)
       10,   10,   20,      # velocity (vx, vy, vz)
       40e2, 40e2, 10e2,    # angles   (roll, pitch, yaw)
       5,    5,    5]       # rates    (p, q, r)

# Input penalty (diagonal R)
R = [0.1, 0.1, 0.01, 0.1]  # [thrust, τ_roll, τ_pitch, τ_yaw]
```

Increase **Q** entries for tighter state tracking. Increase **R** entries for smoother control inputs. For integral tracking, the `Q_z` weights on the integrator states govern how aggressively steady-state error is corrected.

---

## Simulation Scripts

| Script | Controller | Features |
|---|---|---|
| `UnCst_Batch_droneRefTrack.py` | Unconstrained Batch MPC | Integral tracking, time-varying reference |
| `UnCst_Dy_KalmanRefTrack.py` | DP MPC + Kalman | Disturbance estimation, velocity tracking |
| `CstMPC_simple.py` | Constrained Batch MPC | Manual augmented state, state+input bounds |
| `CstMPC_Reftrack.py` | Constrained + Integral tracking | Physical bound conversion, QP per step |
| `CstMPC_Kalman_Reftrack.py` | Constrained + Kalman + Offset-free | Full pipeline with measurement noise |

All scripts run a 30-second simulation with a piecewise-constant reference that cycles through roll/pitch commands at 5-second intervals, and save plots to `images/`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install mujoco numpy scipy qpsolvers[osqp] matplotlib
```

---

## Running

```bash
# Most complete controller (constrained + Kalman + offset-free)
python CstMPC_Kalman_Reftrack.py

# Constrained MPC with integral tracking
python CstMPC_Reftrack.py

# Unconstrained baseline
python UnCst_Batch_droneRefTrack.py
```

Plots are saved to `images/` at the end of each run.

---

## Key Design Decisions

- **Deviation inputs:** The MPC always optimizes `u_dev = u - u_hover`. Hover thrust is added back before passing to the simulator, keeping the linearization assumption valid.
- **Constraint shifting:** When using offset-free MPC, state and input bounds are automatically shifted relative to the computed steady-state targets `(xₛ, uₛ)` so the QP is always solved in deviation coordinates.
- **Kalman filter:** Solved offline via the discrete algebraic Riccati equation (DARE) for the augmented `[x; d]` system. Only the correction step runs online.
- **Controllability/Observability:** Verified at startup for both the base 12-state system and the augmented integral/disturbance-extended systems.

---

## License

Add your license here.
