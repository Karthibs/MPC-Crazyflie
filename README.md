# MPC-Crazyflie

Model Predictive Control (MPC) simulations for a Crazyflie-like quadrotor in MuJoCo.

## Repository structure

```text
.
├── CstMPC_simple.py
├── CstMPC_Reftrack.py
├── CstMPC_Kalman_Reftrack.py
├── UnCst_Batch_droneRefTrack.py
├── UnCst_Dy_KalmanRefTrack.py
├── MPC/
│   ├── ConstrainedBatchMPC.py
│   ├── BatchMpc.py
│   ├── DyPMPC.py
│   ├── drone_environment.py
│   ├── drone_params.py
│   └── chaos_wind_generator.py
├── Drone_xml/
│   ├── scene.xml
│   ├── drone.xml
│   └── assets/
└── images/
```

## What is included

- Constrained batch MPC for quadrotor stabilization and reference tracking
- Offset-free constrained MPC with Kalman disturbance estimation
- Unconstrained batch/dynamic-programming MPC variants
- MuJoCo-based simulation environment and drone model linearization utilities

## Requirements

Python 3.9+ and these packages:

- `numpy`
- `scipy`
- `matplotlib`
- `mujoco`
- `qpsolvers`
- a QP backend for `qpsolvers` (for example `osqp`)

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install numpy scipy matplotlib mujoco qpsolvers osqp
```

## Setup note

Some modules in `MPC/` use legacy absolute imports (for example `from drone_params import ...`).
Before running scripts, set:

```bash
export PYTHONPATH="$PWD/MPC:$PYTHONPATH"
```

## Running examples

From repository root:

```bash
python CstMPC_simple.py
python CstMPC_Reftrack.py
python CstMPC_Kalman_Reftrack.py
python UnCst_Batch_droneRefTrack.py
python UnCst_Dy_KalmanRefTrack.py
```

## Outputs

Simulation plots are saved to `images/`, for example:

- `drone_constrained_mpc_reference_tracking_wic.png`
- `drone_constrained_mpc_inputs_wic.png`
- `drone_constrained_kalman_mpc_tracking.png`
- `drone_constrained_kalman_mpc_inputs.png`
- `UnCst_KalmanRefTrack_Dy.png`

## Notes

- The MuJoCo scene path used by the environment is `Drone_xml/scene.xml`.
- The constrained MPC scripts rely on QP solving through `qpsolvers.solve_qp`.
- `python -m pytest -q` is currently not available in this repository because `pytest` is not installed/configured here.
