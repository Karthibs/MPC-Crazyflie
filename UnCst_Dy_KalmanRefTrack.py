import numpy as np
from MPC.drone_environment import QuadrotorEnv, XML_FILE
from MPC.DyPMPC import DyPMPCKalmanReferenceTracking
from MPC.chaos_wind_generator import ChaosWindGenerator
from MPC.drone_params import (
    DroneParams,
    make_hover_12state_continuous_model
)
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use("Agg")


if __name__ == "__main__":
    env = QuadrotorEnv(XML_FILE)
    chaos_gen = ChaosWindGenerator(force_intensity=0, torque_intensity=0, body_name="Drone")

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
        2e1, 2e1, 5e3,   # position
        10e3, 10e3, 5e2,      # velocity
        4e-2, 4e-2, 1e-2,   # roll pitch yaw
        5, 5, 5          # angular rates
    ])

    R_vec = np.array([0.1, 0.1, 0.01, 0.1])
    N_horizon = 50

    B_d = np.zeros((12,3))
    B_d[3,0] = 1
    B_d[4,1] = 1
    B_d[5,2] = 1
    n_x = Ad.shape[0]
    n_d = B_d.shape[1]
    n_y = C.shape[0]
    Qw = 1e-7 * np.eye(n_x + n_d)
    Rv = 1e-5 * np.eye(n_y)

    controller = DyPMPCKalmanReferenceTracking(
        Ad,
        Bd,
        dt=dt,
        horizon=N_horizon,
        Qx=Q_x,
        Ru=R_vec,
        outputs_to_track=('z', "vx", "vy", "vz"),
        C_y=np.eye(Ad.shape[0]),
        B_d=B_d, 
        Rv = Rv,
        Qw=Qw 
    )

    Rv = 1e-5 * np.eye(n_y)
    print(f"Measurement Noise Covariance Rv:\n{Rv}")
    noise_rng = np.random.default_rng()

    zref = 0.4
    

    # --- Simulation Loop ---
    sim_time, duration = 0.0, 30.0
    next_print = 0.0

    time_log = []
    y_log = []
    xhat_log = []
    ref_log = []

    while sim_time < duration:

        if sim_time <= 5:
            c = np.array([[0.5], [0], [0]])
        elif sim_time <= 10:
            c = np.array([[0], [0.5], [0]])
        elif sim_time <= 15:
            c = np.array([[-0.5], [0], [0]])
        elif sim_time <= 20:
            c = np.array([[0], [-0.5], [0]])
        elif sim_time <= 25:
            c = np.array([[0.5], [0], [0]])
        elif sim_time <= 30:
            c = np.array([[0], [0.5], [0]])
        else:
            c = np.zeros((3, 1))            

        controller.set_reference(np.array([zref, c[0][0], c[1][0], c[2][0]]))
        chaos_gen.apply_disturbance(env.model, env.data)

        pos, vel, roll, p, pitch, q, yaw, r = env.get_state()
        x_current = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], roll, pitch, yaw, p, q, r])


        v_k = noise_rng.multivariate_normal(
            mean=np.zeros(controller.n_y),
            cov=Rv
        )
        
        # Measurement model y = Cx + v
        y_meas = controller.C_y @ x_current + v_k

        u_mpc = controller.step(y_meas)
        time_log.append(sim_time)
        y_log.append(y_meas)
        xhat_log.append(controller.x_aug_hat[:controller.n_x])
        ref_log.append(controller.r)

        U_total = np.array([
            (m * g) +u_mpc[0], # Total Vertical Force in Newtons
            u_mpc[1],           # Total Roll Torque in Nm
            u_mpc[2],           # Total Pitch Torque in Nm
            u_mpc[3]          # Total Yaw Torque in Nm
        ])



        running = env.step(U_total)
        rpy_deg = np.degrees([roll, pitch, yaw])
        ang_vel_deg = np.degrees([p, q, r])
        xpedd = controller.x_aug_hat[:controller.n_x]

        if sim_time >= next_print:
            print(
                    f"[{sim_time:5.2f}s]"
                    # f"Inputs (N): {U_total.round(2)} | "
                    f"Position (m): X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}) | "
                    f"Position (m): X={y_meas[0]:.2f}, Y={y_meas[1]:.2f}, Z={y_meas[2]:.2f}) | "
                    f"Position (m): X={xpedd[0]:.2f}, Y={xpedd[1]:.2f}, Z={xpedd[2]:.2f}) | "
                    # f"Angles (deg): Roll={rpy_deg[0]:.2f}, Pitch={rpy_deg[1]:.2f}, Yaw={rpy_deg[2]:.2f} | "
                    )
            next_print += 0.1

        if not running: break
        sim_time += dt

    env.close()

    time_log = np.array(time_log)
    y_log = np.array(y_log)
    xhat_log = np.array(xhat_log)
    ref_log = np.array(ref_log)
    states_idx = [2,3,4,5]
    state_names = ["z", "vx", "vy", "vz"]  
    
    plt.figure(figsize=(10,8))

    for i, idx in enumerate(states_idx):

        plt.subplot(2,2,i+1)

        plt.plot(time_log, y_log[:,idx], label="Measured (y_meas)", linestyle="--")
        plt.plot(time_log, xhat_log[:,idx], label="Estimated (x_hat)")
        plt.plot(time_log, ref_log[:,i], label="Reference", linewidth=2)

        plt.title(state_names[i])
        plt.xlabel("Time (s)")
        plt.ylabel(state_names[i])
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"images/UnCst_KalmanRefTrack_Dy.png")
    # plt.show()  