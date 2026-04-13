# MPC Drone Flight Controller

A Model Predictive Control (MPC) framework for quadrotor flight simulation built on top of **MuJoCo**.  
The project implements multiple MPC variants, ranging from **unconstrained batch MPC** to **constrained offset-free MPC with Kalman disturbance estimation**, all applied to a **12-state linearized hover model** of a **Crazyflie quadrotor**.

The controller linearizes the quadrotor dynamics around hover, discretizes the continuous-time model using **zero-order hold** via the matrix exponential, and solves a **receding-horizon optimal control problem** at each timestep.

---

## Repository Structure

```text
.
├── MPC/
│   ├── BatchMpc.py                  # Unconstrained batch MPC + Kalman variant
│   ├── ConstrainedBatchMPC.py       # Constrained batch MPC (QP-based) + Kalman variant
│   ├── DyPMPC.py                    # Dynamic Programming MPC + Kalman variant
│   ├── drone_environment.py         # MuJoCo quadrotor environment wrapper
│   ├── drone_params.py              # Physical parameters, linearization, state-space helpers
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

The simulated platform is a **Crazyflie 2.x quadrotor** modeled in MuJoCo.

The **12-state** linearized hover model uses the state vector

$$
x =
\begin{bmatrix}
x & y & z & v_x & v_y & v_z & \phi & \theta & \psi & p & q & r
\end{bmatrix}^\top
$$

where:

- \(x, y, z\) are position
- \(v_x, v_y, v_z\) are linear velocities
- \(\phi, \theta, \psi\) are roll, pitch, and yaw
- \(p, q, r\) are body angular rates

The control input is

$$
u =
\begin{bmatrix}
\Delta T & \tau_\phi & \tau_\theta & \tau_\psi
\end{bmatrix}^\top
$$

where:

- \(\Delta T\) is the deviation from hover thrust
- \(\tau_\phi, \tau_\theta, \tau_\psi\) are roll, pitch, and yaw torques

---

## Controller Architectures

### 1. Unconstrained Batch MPC (`BatchMpc.py`)

This controller precomputes the full receding-horizon feedback gain offline by solving the unconstrained finite-horizon quadratic program analytically:

$$
\min_U \; X^\top \bar{Q} X + U^\top \bar{R} U
\quad \text{subject to} \quad
X = \Omega x_0 + \Gamma U
$$

Substituting the prediction model into the cost gives the closed-form optimal control law

$$
U^\star = K x_0
$$

with

$$
K = -\left(\Gamma^\top \bar{Q} \Gamma + \bar{R}\right)^{-1}\Gamma^\top \bar{Q}\Omega
$$

This gain is computed once during initialization and reused online.

---

### 2. Batch MPC with Integral Reference Tracking  
(`BatchMpc.py` — `BatchMPCReferenceTracking`)

To remove steady-state tracking error, the controller augments the system with integrators on selected output errors.

The integrator dynamics are

$$
z_{k+1} = z_k + \Delta t \, C_{\text{track}} \tilde{x}_k
$$

and the augmented state is

$$
x_{\text{aug}} =
\begin{bmatrix}
\tilde{x} \\
z
\end{bmatrix}
$$

where:

- \(\tilde{x} = x - x_{\text{ref}}\) is the tracking error state
- \(z\) is the integral of the tracked output error

This allows the controller to track outputs such as \(z\), roll, pitch, or yaw with zero steady-state error while also penalizing integral windup through the cost function.

---

### 3. Dynamic Programming MPC (`DyPMPC.py`)

This implementation solves the finite-horizon optimal control problem using the backward Riccati recursion.

Although it is equivalent to the unconstrained batch formulation, it demonstrates the dynamic programming viewpoint of MPC and provides a useful alternative implementation.

The repository also includes a Kalman-based offset-free version:

- `DyPMPCKalmanReferenceTracking`

---

### 4. Constrained Batch MPC (`ConstrainedBatchMPC.py`)

This controller solves the constrained quadratic program online at every timestep using [`qpsolvers`](https://github.com/qpsolvers/qpsolvers) with the **OSQP** backend.

The optimization problem is

$$
\min_U \; \frac{1}{2} U^\top P U + q^\top U
$$

subject to

$$
A_{\text{ineq}} U \le b_0 + E x_0
$$

State and input constraints are assembled in condensed prediction form. Using the predicted output equation

$$
Y = \bar{C}(\Omega x_0 + \Gamma U)
$$

the controller constructs linear inequality constraints of the form

$$
A_{\text{ineq}} U \le b + E x_0
$$

This enables hard enforcement of state and actuator bounds.

---

### 5. Constrained MPC with Integral Reference Tracking  
(`ConstrainedBatchMPC.py` — `ConstrainedBatchMPCReferenceTracking`)

This variant combines:

- constrained QP-based MPC
- integral reference tracking
- physical bound handling in error coordinates

Because the optimization is performed in deviation variables, physical state bounds are automatically shifted relative to the current reference \(x_{\text{ref}}\), so that the constraints remain consistent in the tracking-error formulation.

---

### 6. Constrained Offset-Free MPC with Kalman Filter  
(`ConstrainedBatchMPC.py` — `ConstrainedBatchMPCKalmanReferenceTracking`)

This is the most complete controller in the repository.

It augments the system with a disturbance model and uses a **steady-state Kalman filter** to estimate unmeasured constant disturbances, enabling **offset-free tracking** under persistent model mismatch or external disturbances.

The disturbance-augmented model is

$$
x_{k+1} = A x_k + B u_k + B_d d_k
$$

$$
d_{k+1} = d_k
$$

$$
y_k = C_y x_k + v_k
$$

At each timestep, the controller performs the following sequence:

1. **Kalman correction**  
   Update the estimate \(\begin{bmatrix}\hat{x} & \hat{d}\end{bmatrix}^\top\) using the latest measurement \(y_k\)

2. **Steady-state target computation**  
   Solve

   $$
   \begin{bmatrix}
   A - I & B \\
   H     & 0
   \end{bmatrix}
   \begin{bmatrix}
   x_s \\
   u_s
   \end{bmatrix}
   =
   \begin{bmatrix}
   -B_d \hat{d} \\
   r
   \end{bmatrix}
   $$

   to obtain the steady-state target \((x_s, u_s)\)

3. **Constrained QP in deviation coordinates**  
   Solve for

   $$
   \tilde{u} = u - u_s
   $$

   using deviation dynamics based on

   $$
   \tilde{x} = x - x_s
   $$

4. **Apply control**

   $$
   u = u_s + \tilde{u}
   $$

5. **Kalman prediction**  
   Propagate the estimator forward to the next timestep

This structure gives robust tracking performance even in the presence of persistent disturbances such as wind or modeling error.

---

## State and Input Constraints

Constraints are assembled in condensed prediction form and stacked before each QP solve.

| Constraint | Typical Values |
|---|---|
| Altitude \(z\) | \(0.10 \text{ m} \le z \le 0.45 \text{ m}\) |
| Roll angle \(\phi\) | \(\pm 20^\circ\) |
| Pitch angle \(\theta\) | \(\pm 20^\circ\) |
| Thrust deviation \(\Delta T\) | \([0, 2T_{\text{hover}}] - T_{\text{hover}}\) |
| Roll torque \(\tau_\phi\) | \(\pm 2 \times 10^{-3}\ \text{Nm}\) |
| Pitch torque \(\tau_\theta\) | \(\pm 2 \times 10^{-3}\ \text{Nm}\) |
| Yaw torque \(\tau_\psi\) | \(\pm 1 \times 10^{-3}\ \text{Nm}\) |

---

## Cost Tuning

The quadratic cost penalizes both state deviation and control effort.

```python
# State penalty (diagonal Q)
Q_x = [2e-1, 2e-1, 5e3,    # position: x, y, z
       10,   10,   20,     # velocity: vx, vy, vz
       40e2, 40e2, 10e2,   # angles: roll, pitch, yaw
       5,    5,    5]      # rates: p, q, r

# Input penalty (diagonal R)
R = [0.1, 0.1, 0.01, 0.1]  # [thrust, tau_roll, tau_pitch, tau_yaw]
```

In general:

- increasing entries in \(Q\) enforces tighter state tracking
- increasing entries in \(R\) produces smoother, less aggressive control inputs
- for integral tracking, the \(Q_z\) weights determine how strongly steady-state error is corrected

---

## Simulation Scripts

| Script | Controller | Features |
|---|---|---|
| `UnCst_Batch_droneRefTrack.py` | Unconstrained Batch MPC | Integral tracking, time-varying reference |
| `UnCst_Dy_KalmanRefTrack.py` | DP MPC + Kalman | Disturbance estimation, velocity tracking |
| `CstMPC_simple.py` | Constrained Batch MPC | Manual augmented state, state and input bounds |
| `CstMPC_Reftrack.py` | Constrained MPC + Integral tracking | Physical bound conversion, online QP solve |
| `CstMPC_Kalman_Reftrack.py` | Constrained MPC + Kalman + Offset-free | Full pipeline with measurement noise |

All scripts run a **30-second simulation** with a piecewise-constant reference that cycles through roll and pitch commands every **5 seconds**, and save plots to the `images/` directory.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install mujoco numpy scipy qpsolvers[osqp] matplotlib
```

---

## Running the Controllers

```bash
# Most complete controller: constrained + Kalman + offset-free
python CstMPC_Kalman_Reftrack.py

# Constrained MPC with integral tracking
python CstMPC_Reftrack.py

# Unconstrained baseline
python UnCst_Batch_droneRefTrack.py
```

Simulation plots are saved to `images/` at the end of each run.

---

## Key Design Decisions

### Deviation Inputs
The MPC optimizes deviation inputs

$$
u_{\text{dev}} = u - u_{\text{hover}}
$$

Hover thrust is added back before applying the command to the MuJoCo simulator.  
This keeps the controller consistent with the hover linearization.

### Constraint Shifting
In the offset-free formulation, state and input bounds are automatically shifted relative to the computed steady-state targets \((x_s, u_s)\), so that the QP is solved entirely in deviation coordinates.

### Kalman Filter Design
The Kalman filter is obtained offline by solving the discrete algebraic Riccati equation (DARE) for the augmented disturbance-estimation model. Only the estimator recursion is executed online.

### Controllability and Observability Checks
The project verifies controllability and observability at startup for:

- the base 12-state hover model
- the integral-augmented tracking model
- the disturbance-augmented offset-free model

This helps ensure that each controller formulation is well-posed before simulation begins.

---

## License

Add your license here.
