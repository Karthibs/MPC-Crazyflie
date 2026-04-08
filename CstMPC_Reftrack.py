import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MPC.drone_environment import QuadrotorEnv, XML_FILE
from MPC.chaos_wind_generator import ChaosWindGenerator
from MPC.drone_params import (
    DroneParams,
    make_hover_12state_continuous_model
)
from MPC.ConstrainedBatchMPC import (
    ConstrainedBatchMPCReferenceTracking,
    stack_condensed_constraints,
)


if __name__ == "__main__":
    env = QuadrotorEnv(XML_FILE)
    chaos_gen = ChaosWindGenerator(
        force_intensity=0.0,
        torque_intensity=0.0,
        body_name="Drone"
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
        2e-1, 2e-1, 5e3,   # position
        10,   10,   20,    # velocity
        40e2, 40e2, 10e2,  # roll, pitch, yaw
        5,    5,    5      # angular rates
    ])

    Q_z = np.array([8e1, 8e1, 8e1, 1.2e1])   # integral-of-output-error weights
    R_vec = np.array([0.1, 0.1, 0.01, 0.1])
    N_horizon = 10

    tracker = ConstrainedBatchMPCReferenceTracking(
        Ad,
        Bd,
        dt=dt,
        horizon=N_horizon,
        Q_x=Q_x,
        Q_z=Q_z,
        R=R_vec,
        outputs_to_track=("z", "roll", "pitch", "yaw"),
    )

    z_min, z_max = 0.10, 0.45
    roll_min, roll_max = -np.deg2rad(20.0), np.deg2rad(20.0)
    pitch_min, pitch_max = -np.deg2rad(11.0), np.deg2rad(11.0)
    yaw_min, yaw_max = -0.015, 0.015  # no yaw constraints


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
        0.01,
        roll_tau_max,
        pitch_tau_max,
        yaw_tau_max,
    ])

    A_input, E_input, b_input = tracker.build_input_box_constraints(
        lower=u_lower_dev,
        upper=u_upper_dev,
    )

    sim_time = 0.0
    duration = 30.0
    next_print = 0.0

    time_log = []
    x_log = []
    u_log = []
    ref_log = []

    # Optional: start safely inside the feasible set
    env.reset(height=0.30)
    tracker.reset()

    while sim_time < duration:
        # Time-varying reference
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

        zref = 0.40

        # Here we track: (z, roll, pitch, yaw)
        x_ref = np.zeros(tracker.n_x)
        x_ref[[2, 6, 7, 8]] = np.array([zref, c[0], c[1], c[2]])
        tracker.set_x_ref(x_ref)


        A_state, E_state, b_state = tracker.build_state_box_constraints(
            states=("z", "roll", "pitch", "yaw"),
            lower=np.array([z_min, roll_min, pitch_min, yaw_min]),
            upper=np.array([z_max, roll_max, pitch_max, yaw_max]),
            terminal_only=False,
            bounds_are_actual_states=True,
        )

        A_ineq, E_x0, b0 = stack_condensed_constraints(
            (A_state, E_state, b_state),
            (A_input, E_input, b_input),
        )


        chaos_gen.apply_disturbance(env.model, env.data)

        pos, vel, roll, p, pitch, q, yaw, r = env.get_state()
        x_current = np.array([
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            roll, pitch, yaw,
            p, q, r
        ])


        try:
            u_dev = tracker.compute_action(
                x_current=x_current,
                A_ineq=A_ineq,
                E_x0=E_x0,
                b0=b0,
            )
        except RuntimeError as e:
            print(f"\nQP failed at t={sim_time:.3f}s: {e}")
            break


        U_total = np.array([
            hover_force + u_dev[0],
            u_dev[1],
            u_dev[2],
            u_dev[3],
        ])

        running = env.step(U_total)
        if not running:
            break


        time_log.append(sim_time)
        x_log.append(x_current.copy())
        u_log.append(u_dev.copy())
        ref_log.append(np.array([zref, c[0], c[1], c[2]]))

        rpy_deg = np.degrees([roll, pitch, yaw])

        if sim_time >= next_print:
            print(
                f"[{sim_time:5.2f}s] "
                f"Pos: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f} | "
                f"Ref: Z={zref:.2f}, Roll={np.degrees(c[0]):.2f}deg, "
                f"Pitch={np.degrees(c[1]):.2f}deg, Yaw={np.degrees(c[2]):.2f}deg | "
                f"Angles(deg): Roll={rpy_deg[0]:.2f}, Pitch={rpy_deg[1]:.2f}, Yaw={rpy_deg[2]:.2f}"
            )
            next_print += 0.1

        sim_time += dt

    env.close()


    if len(time_log) > 0:
        time_log = np.array(time_log)
        x_log = np.array(x_log)
        u_log = np.array(u_log)
        ref_log = np.array(ref_log)

        states_idx = [2, 6, 7, 8]
        state_names = ["z", "roll", "pitch", "yaw"]

        plt.figure(figsize=(10, 8))

        for i, idx in enumerate(states_idx):
            plt.subplot(2, 2, i + 1)
            plt.plot(time_log, x_log[:, idx], label="State")
            plt.plot(time_log, ref_log[:, i], label="Reference", linewidth=2)
            plt.title(state_names[i])
            plt.xlabel("Time (s)")
            plt.ylabel(state_names[i])
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig("images/drone_constrained_mpc_reference_tracking_wic.png")

        plt.figure(figsize=(10, 8))
        input_names = ["Thrust Dev", "Roll Torque", "Pitch Torque", "Yaw Torque"]
        for i in range(u_log.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.plot(time_log, u_log[:, i], label=input_names[i])
            plt.title(input_names[i])
            plt.xlabel("Time (s)")
            plt.ylabel("Control Input")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.savefig("images/drone_constrained_mpc_inputs_wic.png") 