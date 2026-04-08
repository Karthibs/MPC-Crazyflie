import numpy as np
import scipy.linalg as la
from scipy.linalg import expm
from MPC.drone_environment import QuadrotorEnv, XML_FILE
from MPC.BatchMpc import BatchMPCReferencetracking
from MPC.chaos_wind_generator import ChaosWindGenerator
from MPC.drone_params import (
    DroneParams,
    make_hover_12state_continuous_model
)


if __name__ == "__main__":

    env = QuadrotorEnv(XML_FILE)
    chaos_gen = ChaosWindGenerator(force_intensity=0.0, torque_intensity=0, body_name="Drone")

    params = DroneParams.from_env(env, gravity=9.81)
    m = params.mass
    g = params.gravity
    Ix, Iy, Iz = params.Ix, params.Iy, params.Iz
    hover_force = params.hover_force

    lin_model = make_hover_12state_continuous_model(params, dt=env.dt)
    Ad, Bd, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D

    n = Ad.shape[0]
    dt = env.dt

    Q_x = np.array([
        2e-4, 2e-4, 9e1,   # position error (x,y,z)
        5e-1, 5e-1, 8e-1,   # velocity error (vx,vy,vz)
        2e4, 2e4, 5e4,   # angle error (roll,pitch,yaw)
        1e2, 1e2, 1e2,   # rate error (p,q,r)
    ])
    Q_z = np.array([1e2, 8e1, 8e1, 1.2e1])  # integral-of-position-error weights
    R_vec = np.array([0.1, 0.1, 0.01, 0.1])
    N_horizon = 10

    tracker = BatchMPCReferencetracking(
        Ad,
        Bd,
        dt=dt,
        horizon=N_horizon,
        Q_x=Q_x,
        Q_z=Q_z,
        R=R_vec,
        outputs_to_track=('z', "roll", "pitch", "yaw"),
    )

    zref = 0.4

    # --- Simulation Loop ---
    sim_time, duration = 0.0, 40.0
    next_print = 0.0

    while sim_time < duration:
        if sim_time <= 5:
            c = np.array([[0], [0], [0]])
        elif sim_time <= 10:
            c = np.array([[0.5], [0], [0]])
        elif sim_time <= 15:
            c = np.array([[0], [0.5], [0]])
        elif sim_time <= 20:
            c = np.array([[-0.5], [0], [0]])
        elif sim_time <= 25:
            c = np.array([[0], [-0.5], [0]])
        elif sim_time <= 30:
            c = np.array([[0.5], [0], [0]])
        elif sim_time <= 35:
            c = np.array([[0], [0.5], [0]])
        else:
            c = np.zeros((3, 1))

        tracker.set_position_ref(np.array([zref, c[0][0], c[1][0], c[2][0]]), [2, 6, 7, 8])
        
        chaos_gen.apply_disturbance(env.model, env.data)

        pos, vel, roll, p, pitch, q, yaw, r = env.get_state()
        x_current = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], roll, pitch, yaw, p, q, r])


        u_mpc = tracker.compute_action(x_current)

        U_total = np.array([
            (m * g) +u_mpc[0], # Total Vertical Force in Newtons
            u_mpc[1],           # Total Roll Torque in Nm
            u_mpc[2],           # Total Pitch Torque in Nm
            u_mpc[3]          # Total Yaw Torque in Nm
        ])
        

        running = env.step(U_total)
        rpy_deg = np.degrees([roll, pitch, yaw])
        ang_vel_deg = np.degrees([p, q, r])

        if sim_time >= next_print:
            print(
                    f"[{sim_time:5.2f}s] "
                    f"Inputs (N): {U_total.round(2)} | "
                    f"Position (m): X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f} (ref Z={zref:.2f}) | "
                    f"Angles (deg): Roll={rpy_deg[0]:.2f}, Pitch={rpy_deg[1]:.2f}, Yaw={rpy_deg[2]:.2f} | "
                    )
            next_print += 0.5

        if not running: break
        sim_time += dt

    env.close()