import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MPC.drone_environment import QuadrotorEnv, XML_FILE
from MPC.chaos_wind_generator import ChaosWindGenerator
from MPC.drone_params import (
    DroneParams,
    make_hover_12state_continuous_model,
)

from MPC.ConstrainedBatchMPC import ConstrainedBatchMPCKalmanReferenceTracking


if __name__ == "__main__":
    env = QuadrotorEnv(XML_FILE)
    chaos_gen = ChaosWindGenerator(
        force_intensity=0.0,
        torque_intensity=0.0,
        body_name="Drone",
    )

    params = DroneParams.from_env(env, gravity=9.81)
    m = params.mass
    g = params.gravity
    hover_force = params.hover_force

    lin_model = make_hover_12state_continuous_model(params, dt=env.dt)
    Ad, Bd, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D

    n = Ad.shape[0]
    dt = env.dt

    Q_x = np.array([
        2e-1, 2e-1, 5e3,    # position
        10,   10,   20,     # velocity
        40e2, 40e2, 10e2,   # roll, pitch, yaw
        5,    5,    5       # angular rates
    ])

    R_vec = np.array([0.1, 0.1, 0.01, 0.1])
    N_horizon = 10


    B_d = np.zeros((12, 3))
    B_d[6, 0] = 1.0
    B_d[7, 1] = 1.0
    B_d[8, 2] = 1.0

    n_x = Ad.shape[0]
    n_d = B_d.shape[1]
    n_y = C.shape[0]

    Qw = 1e-5 * np.eye(n_x + n_d)
    Rv = 1e-7 * np.eye(n_y)

    controller = ConstrainedBatchMPCKalmanReferenceTracking(
        Ad,
        Bd,
        dt=dt,
        horizon=N_horizon,
        Qx=Q_x,
        Ru=R_vec,
        outputs_to_track=("z", "roll", "pitch", "yaw"),
        C_y=np.eye(Ad.shape[0]),
        B_d=B_d,
        Rv=Rv,
        Qw=Qw,
    )

    print(f"\nMeasurement Noise Covariance Rv:\n{Rv}")
    noise_rng = np.random.default_rng()


    # Hard state bounds on actual physical states
    state_constraints = [
        {
            "states": ("z", "roll", "pitch"),
            "lower": np.array([
                0.10,                    # z_min
                -np.deg2rad(20.0),       # roll_min
                -np.deg2rad(20.0),       # pitch_min
            ]),
            "upper": np.array([
                0.45,                    # z_max
                np.deg2rad(20.0),        # roll_max
                np.deg2rad(20.0),        # pitch_max
            ]),
            "terminal_only": False,
        }
    ]


    thrust_min_actual = 0.0
    thrust_max_actual = 2.0 * hover_force

    roll_tau_max = 2.0e-3
    pitch_tau_max = 2.0e-3
    yaw_tau_max = 1.0e-3

    u_lower_dev = np.array([
        thrust_min_actual - hover_force,
        -roll_tau_max,
        -pitch_tau_max,
        -yaw_tau_max,
    ])

    u_upper_dev = np.array([
        0.1,
        roll_tau_max,
        pitch_tau_max,
        yaw_tau_max,
    ])


    env.reset(height=0.30)
    controller.reset(x0=np.zeros(n_x), d0=np.zeros(n_d))


    sim_time = 0.0
    duration = 30.0
    next_print = 0.0

    time_log = []
    u_log = []
    y_log = []
    xhat_log = []
    ref_log = []

    while sim_time < duration:
        if sim_time <= 5:
            c = np.array([0.2, 0.0, 0.0])
        elif sim_time <= 10:
            c = np.array([0.0, 0.2, 0.0])
        elif sim_time <= 15:
            c = np.array([-0.2, 0.0, 0.0])
        elif sim_time <= 20:
            c = np.array([0.0, -0.2, 0.0])
        elif sim_time <= 25:
            c = np.array([0.2, 0.0, 0.0])
        elif sim_time <= 30:
            c = np.array([0.0, 0.2, 0.0])
        else:
            c = np.zeros(3)

        zref = 0.4
        controller.set_reference(np.array([zref, c[0], c[1], c[2]]))

        chaos_gen.apply_disturbance(env.model, env.data)

        pos, vel, roll, p, pitch, q, yaw, r = env.get_state()
        x_current = np.array([
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            roll, pitch, yaw,
            p, q, r
        ])

        v_k = noise_rng.multivariate_normal(
            mean=np.zeros(controller.n_y),
            cov=Rv
        )
        y_meas = controller.C_y @ x_current + v_k

        try:
            # Constrained offset-free MPC step
            u_mpc = controller.step(
                y_meas,
                state_constraints=state_constraints,
                input_lower=u_lower_dev,
                input_upper=u_upper_dev,
            )
        except RuntimeError as e:
            print(f"\nQP failed at t={sim_time:.3f}s: {e}")
            break

        # Add hover thrust back to the first channel
        U_total = np.array([
            hover_force + u_mpc[0],   # total thrust
            u_mpc[1],                 # roll torque
            u_mpc[2],                 # pitch torque
            u_mpc[3],                 # yaw torque
        ])

        running = env.step(U_total)
        if not running:
            break

        # Logging
        time_log.append(sim_time)
        y_log.append(y_meas.copy())
        xhat_log.append(controller.x_aug_hat[:controller.n_x].copy())
        ref_log.append(controller.r.copy())
        u_log.append(u_mpc.copy())

        if sim_time >= next_print:
            x_est = controller.x_aug_hat[:controller.n_x]
            print(
                f"[{sim_time:5.2f}s] "
                f"Est Pos: X={x_est[0]:.2f}, Y={x_est[1]:.2f}, Z={x_est[2]:.2f} | "
                f"Ref: Z={controller.r[0]:.2f}, Roll={np.degrees(controller.r[1]):.2f}deg, "
                f"Pitch={np.degrees(controller.r[2]):.2f}deg, Yaw={np.degrees(controller.r[3]):.2f}deg | "
                f"Rates(rad/s): p={p:.2f}, q={q:.2f}, r={r:.2f}"
            )
            next_print += 0.1

        sim_time += dt

    env.close()


    if len(time_log) > 0:
        time_log = np.array(time_log)
        y_log = np.array(y_log)
        xhat_log = np.array(xhat_log)
        u_log = np.array(u_log)
        ref_log = np.array(ref_log)

        states_idx = [2, 6, 7, 8]
        state_names = ["z", "roll", "pitch", "yaw"]

        plt.figure(figsize=(14, 8))

        for i, idx in enumerate(states_idx):
            plt.subplot(2, 2, i + 1)
            plt.plot(time_log, y_log[:, idx], label="Measured (y_meas)", linestyle="--")
            plt.plot(time_log, xhat_log[:, idx], label="Estimated (x_hat)")
            plt.plot(time_log, ref_log[:, i], label="Reference", linewidth=2)

            plt.title(state_names[i])
            plt.xlabel("Time (s)")
            plt.ylabel(state_names[i])
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig("images/drone_constrained_kalman_mpc_tracking.png")

        plt.figure(figsize=(14, 8))
        input_names = ["delta_thrust", "tau_roll", "tau_pitch", "tau_yaw"]
        for i in range(u_log.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.plot(time_log, u_log[:, i], label=input_names[i])
            plt.title(input_names[i])
            plt.xlabel("Time (s)")
            plt.ylabel("Control Input")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.savefig("images/drone_constrained_kalman_mpc_inputs.png")

        