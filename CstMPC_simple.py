import numpy as np

from MPC.drone_environment import QuadrotorEnv, XML_FILE
from MPC.chaos_wind_generator import ChaosWindGenerator
from MPC.drone_params import (
    DroneParams,
    STATE_ORDER_12,
    controllability_rank,
    make_hover_12state_continuous_model,
    observability_rank,
    output_matrix,
)


from MPC.ConstrainedBatchMPC import (
    ConstrainedBatchMPC,
    build_condensed_state_box_constraints,
    build_condensed_input_box_constraints,
    stack_condensed_constraints,)


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
    print(f"Drone Mass: {m:.4f} kg, Hover Force: {hover_force:.4f} N")

    lin_model = make_hover_12state_continuous_model(params, dt=env.dt)
    Ad, Bd, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D
    n = Ad.shape[0]
    dt = env.dt

    # ---------------------------------------------------------
    # 1. CHECK CONTROLLABILITY / OBSERVABILITY
    # ---------------------------------------------------------
    rank_c = controllability_rank(Ad, Bd)
    print(f"Controllability Matrix Rank: {rank_c}")
    print("The system is FULLY CONTROLLABLE." if rank_c == n else f"The system is NOT fully controllable. ({rank_c}/{n})")

    rank_o = observability_rank(Ad, C)
    print(f"\nObservability Matrix Rank: {rank_o}")
    print("The system is FULLY OBSERVABLE." if rank_o == n else f"The system is NOT fully observable. ({rank_o}/{n})")


    outputs_to_track = ("z", "roll", "pitch", "yaw")
    C_track = output_matrix(list(outputs_to_track))

    n_x = Ad.shape[0]
    n_u = Bd.shape[1]
    n_z = C_track.shape[0]

    A_aug = np.zeros((n_x + n_z, n_x + n_z))
    B_aug = np.zeros((n_x + n_z, n_u))

    A_aug[:n_x, :n_x] = Ad
    B_aug[:n_x, :] = Bd
    A_aug[n_x:, :n_x] = dt * C_track
    A_aug[n_x:, n_x:] = np.eye(n_z)


    Q_x = np.array([
        2e-4, 2e-4, 6e1,   # x, y, z
        5e-1, 5e-1, 8e-1,  # vx, vy, vz
        2e4, 2e4, 5e4,     # roll, pitch, yaw
        1e2, 1e2, 1e2,     # p, q, r
    ])
    Q_z = np.array([8e1, 8e1, 8e1, 1.2e1])   # integral error weights
    R_vec = np.array([0.1, 0.1, 0.01, 0.1])
    N_horizon = 10

    Q_aug = np.concatenate([Q_x, Q_z])

    mpc = ConstrainedBatchMPC(
        A=A_aug,
        B=B_aug,
        N=N_horizon,
        Q_diag=Q_aug,
        R_diag=R_vec,
    )

    n_aug = A_aug.shape[0]
    rank_c_aug = controllability_rank(A_aug, B_aug)
    C_aug_obs = np.hstack([C_track, np.zeros((C_track.shape[0], n_z))])
    rank_o_aug = observability_rank(A_aug, C_aug_obs)

    print(f"\nAugmented Controllability Rank: {rank_c_aug} (n={n_aug})")
    print(f"Augmented Observability Rank: {rank_o_aug} (n={n_aug})")

    x_ref = np.zeros(n_x)
    x_ref[[2, 6, 7, 8]] = np.array([0.4, 0.0, 0.0, 0.3])  # z, roll, pitch, yaw
    z_int = np.zeros(n_z)

    # STATE CONSTRAINTS
    #
    # The controller optimizes the error-state x_tilde = x - x_ref.
    # So if you want ACTUAL state constraints, convert them to
    # error-state bounds:
    #
    #   lower_tilde = lower_actual - x_ref_subset
    #   upper_tilde = upper_actual - x_ref_subset
    # ---------------------------------------------------------


    aug_state_order = STATE_ORDER_12 + tuple(f"int_{name}" for name in outputs_to_track)


    constrained_states = ("z", "roll", "pitch")

    actual_lower = np.array([
        0,                    # z_min
        -np.deg2rad(20.0),       # roll_min
        -np.deg2rad(20.0),       # pitch_min
    ])
    actual_upper = np.array([
        10.405,                    # z_max
        np.deg2rad(20.0),        # roll_max
        np.deg2rad(20.0),        # pitch_max
    ])

    ref_subset = np.array([
        x_ref[2],  # z_ref
        x_ref[6],  # roll_ref
        x_ref[7],  # pitch_ref
    ])

    lower_tilde = actual_lower - ref_subset
    upper_tilde = actual_upper - ref_subset

    A_state, E_state, b_state = build_condensed_state_box_constraints(
        Omega=mpc.Omega,
        Gamma=mpc.Gamma,
        horizon=mpc.N,
        states=constrained_states,
        lower=lower_tilde,
        upper=upper_tilde,
        state_order=aug_state_order,
        terminal_only=False,   # enforce over whole horizon
    )

    # -------------------------
    # Input constraints
    #
    # MPC output is deviation input:
    #   u_dev = [delta_thrust, tau_roll, tau_pitch, tau_yaw]
    #
    # Actual applied thrust is:
    #   thrust = hover_force + delta_thrust
    #
    # So thrust bounds must be converted to deviation bounds.
    # -------------------------
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
        thrust_max_actual - hover_force,
        roll_tau_max,
        pitch_tau_max,
        yaw_tau_max,
    ])

    A_input, E_input, b_input = build_condensed_input_box_constraints(
        n_inputs=mpc.m,
        n_states=mpc.n,
        horizon=mpc.N,
        lower=u_lower_dev,
        upper=u_upper_dev,
    )

    A_ineq, E_x0, b0 = stack_condensed_constraints(
        (A_state, E_state, b_state),
        (A_input, E_input, b_input),
    )


    sim_time = 0.0
    duration = 10.0
    next_print = 0.0

    try:
        while sim_time < duration:
            chaos_gen.apply_disturbance(env.model, env.data)


            pos, vel, roll, p, pitch, q, yaw, r = env.get_state()
            x_current = np.array([
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                roll, pitch, yaw,
                p, q, r
            ])

            x_tilde = x_current - x_ref
            z_int = z_int + dt * (C_track @ x_tilde)
            x_aug = np.concatenate([x_tilde, z_int])

            try:
                u_dev = mpc.compute_action(
                    x_current=x_aug,
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

            rpy_deg = np.degrees([roll, pitch, yaw])

            if sim_time >= next_print:
                print(
                    f"[{sim_time:5.2f}s] | "
                    f"dev Inputs: {u_dev.round(4)} | "
                    f"Inputs: {U_total.round(4)} | "
                    f"Pos: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f} "
                    f"(ref Z={x_ref[2]:.2f}) | "
                    f"Angles(deg): Roll={rpy_deg[0]:.2f}, Pitch={rpy_deg[1]:.2f}, Yaw={rpy_deg[2]:.2f}"
                )
                next_print += 0.5

            sim_time += dt

    finally:
        env.close()