import numpy as np
import scipy.linalg as la

from drone_params import STATE_ORDER_12, output_matrix

class BatchMPC:
    """
    Batch Model Predictive Control for Linear Time-Invariant (LTI) Systems.
    This implementation pre-calculates the gain matrix K offline to 
    ensure real-time performance (Step 5 is just a matrix multiplication).
    """
    def __init__(self, A, B, N, Q_diag, R_diag):
        n = A.shape[0]  # Number of states
        m = B.shape[1]  # Number of inputs        
        Q = np.diag(Q_diag)
        R = np.diag(R_diag)
        Pf = Q  # Terminal cost (could be solved via Riccati for better stability)

        # --- Step 1: Construct Prediction Matrices (Omega and Gamma) ---
        # Omega maps the current state to future states: X_future = Omega * x0
        # Omega = np.zeros((n * N, n))
        Omega = np.vstack([np.linalg.matrix_power(A, i) for i in range(1, N + 1)])
        # Gamma maps future inputs to future states: X_future = Gamma * U_future
        Gamma = np.zeros((n * N, m * N))

        for i in range(N):
            for j in range(N):
                if i >= j:
                    Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B
            
        # for i in range(1, N + 1):
        #     Omega[(i-1)*n : i*n, :] = np.linalg.matrix_power(A, i)
        #     for j in range(1, i + 1):
        #         Gamma[(i-1)*n : i*n, (j-1)*m : j*m] = np.linalg.matrix_power(A, i-j) @ B
        
        # --- Step 2: Construct Augmented Weight Matrices ---
        Q_list = [Q] * (N - 1) + [Pf]
        # Q_bar = la.block_diag(*([Q] * (N-1)), Pf)
        Q_bar = la.block_diag(*Q_list)
        R_bar = la.block_diag(*([R] * N))
        
        # --- Step 3: Compute the Offline Batch Gain ---
        # Solving the Quadratic Programming (QP) problem analytically:
        # min J = X^T Q X + U^T R U  s.t. X = Omega*x0 + Gamma*U
        H = Gamma.T @ Q_bar @ Gamma + R_bar
        M = Gamma.T @ Q_bar @ Omega
        
        # K_full calculates the entire input sequence for the horizon N
        # K_full = -np.linalg.solve(H, M)
        K_full = -np.linalg.inv(H) @ M  # More numerically stable than solve for large matrices
        
        # --- Step 4: Extract the Receding Horizon Gain ---
        # We only apply the first control action in the sequence
        self.K_batch = K_full[0:m, :]
        print(f"Batch MPC Gain K:\n{self.K_batch.round(4)}")
       
    def compute_action(self, x_current):
        """Calculates the optimal control action for the current state."""
        return self.K_batch @ x_current


class BatchMPCReferencetracking:
    """

    # ---------------------------------------------------------
    # REFERENCE TRACKING WITH INTEGRAL ACTION
    # ---------------------------------------------------------
    Batch MPC with output reference tracking + integral action.

    Implements the augmentation shown in your notes:
        z_{k+1} = z_k + dt * (y_k - r)
        y_k = C_track x_k

    We run the MPC on the augmented *error state*:
        x_tilde = x - x_ref
        z_{k+1} = z_k + dt * (C_track x_tilde)
        x_aug = [x_tilde; z]
    """

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        *,
        dt: float,
        horizon: int,
        Q_x: np.ndarray,
        Q_z: np.ndarray,
        R: np.ndarray,
        outputs_to_track: tuple[str, ...] = ("x", "y", "z"),
        state_order: tuple[str, ...] = STATE_ORDER_12,
    ):
        self.dt = float(dt)
        self.C_track = output_matrix(list(outputs_to_track), state_order=state_order)

        self.n_x = int(Ad.shape[0])
        self.n_u = int(Bd.shape[1])
        self.n_z = int(self.C_track.shape[0])

        self.A_aug = np.zeros((self.n_x + self.n_z, self.n_x + self.n_z))
        self.B_aug = np.zeros((self.n_x + self.n_z, self.n_u))

        self.A_aug[: self.n_x, : self.n_x] = Ad
        self.B_aug[: self.n_x, :] = Bd
        self.A_aug[self.n_x :, : self.n_x] = self.dt * self.C_track
        self.A_aug[self.n_x :, self.n_x :] = np.eye(self.n_z)

        Q_vec = np.concatenate([np.asarray(Q_x, dtype=float).ravel(), np.asarray(Q_z, dtype=float).ravel()])
        self.mpc = BatchMPC(self.A_aug, self.B_aug, int(horizon), Q_vec, np.asarray(R, dtype=float).ravel())

        self.x_ref = np.zeros(self.n_x)
        self.z_int = np.zeros(self.n_z)

    def reset(self) -> None:
        self.z_int[:] = 0.0

    # def set_x_ref(self, x_ref: np.ndarray) -> None:
    #     x_ref = np.asarray(x_ref, dtype=float).ravel()
    #     if x_ref.size != self.n_x:
    #         raise ValueError(f"x_ref must have length {self.n_x}, got {x_ref.size}")
    #     self.x_ref = x_ref.copy()

    def set_position_ref(self, ref_states: np.ndarray, states: np.ndarray) -> None:
        """Convenience setter when state order is the default 12-state."""
        ref_states = np.asarray(ref_states, dtype=float).ravel()
        states = np.asarray(states)
        # print(f"Setting position reference: {ref_states}")
        self.x_ref[states] = ref_states



    def compute_action(self, x_current: np.ndarray) -> np.ndarray:
        """Returns the MPC control for the current state and updates the integrator."""
        x_current = np.asarray(x_current, dtype=float).ravel()
        if x_current.size != self.n_x:
            raise ValueError(f"x_current must have length {self.n_x}, got {x_current.size}")

        x_tilde = x_current - self.x_ref
        self.z_int = self.z_int + self.dt * (self.C_track @ x_tilde)
        x_aug = np.concatenate([x_tilde, self.z_int])
        return self.mpc.compute_action(x_aug)


class BatchMPCKalmanReferenceTracking:
    """Offset-free Batch MPC with disturbance estimation and reference tracking.

    Implements the algorithm in your screenshot ("Offset-Free MPC with Disturbance
    Rejection and Reference Tracking") for a discrete-time plant:

        x_{k+1} = A x_k + B u_k + B_d d_k
        d_{k+1} = d_k
        y_k     = C_y x_k

    At each step:
      1) Kalman correction -> estimates x_hat, d_hat
      2) Steady-state target generator -> (x_s, u_s)
      3) Control law: u = u_s + K (x_hat - x_s)
      4) Kalman prediction -> x_aug_pred for next step

    Notes:
      - This is an unconstrained MPC feedback (BatchMPC gain K).
      - Choose H (tracked outputs) and reference r accordingly.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        *,
        horizon: int,
        Qx: np.ndarray,
        Ru: np.ndarray,
        dt: float,
        outputs_to_track: tuple[str, ...] = ("x", "y", "z", "yaw"),
        state_order: tuple[str, ...] = STATE_ORDER_12,
        C_y: np.ndarray | None = None,
        B_d: np.ndarray | None = None,
        Qw: np.ndarray | None = None,
        Rv: np.ndarray | None = None,
    ):
        self.dt = float(dt)

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.n_x = int(self.A.shape[0])
        self.n_u = int(self.B.shape[1])

        if self.A.shape != (self.n_x, self.n_x):
            raise ValueError("A must be square")
        if self.B.shape[0] != self.n_x:
            raise ValueError("B must have same number of rows as A")

        # Measurement model y = C_y x
        if C_y is None:
            self.C_y = np.eye(self.n_x)
        else:
            self.C_y = np.asarray(C_y, dtype=float)

        self.n_y = int(self.C_y.shape[0])
        if self.C_y.shape[1] != self.n_x:
            raise ValueError("C_y must have shape (n_y, n_x)")

        # Disturbance model x_{k+1} += B_d d; d_{k+1} = d
        if B_d is None:
            self.B_d = np.ones((self.n_x,1))
        else:
            self.B_d = np.asarray(B_d, dtype=float)

        print(f"Bd matrix:\n{self.B_d}")
        
        print(f"B_d shape: {self.B_d.shape}")

        self.n_d = int(self.B_d.shape[1])

        print(f"n_x: {self.n_x}, n_u: {self.n_u}, n_y: {self.n_y}, n_d: {self.n_d}")
        if self.B_d.shape[0] != self.n_x:
            raise ValueError("B_d must have shape (n_x, n_d)")

        # Tracked output constraint H x_s = r
        self.H = output_matrix(list(outputs_to_track), state_order=state_order)
        print(f"H matrix:\n{self.H}")
        self.n_r = int(self.H.shape[0])
        print(f"n_r: {self.n_r}")

        # 1) Controller gain K (unconstrained Batch MPC feedback)
        self._controller = BatchMPC(self.A, self.B, int(horizon), np.asarray(Qx, dtype=float).ravel(), np.asarray(Ru, dtype=float).ravel())
        self.K = self._controller.K_batch

        # 2) Estimator gain L via steady-state Kalman filter on augmented system
        # x_aug = [x; d]
        # x_aug+ = A_aug x_aug + B_aug u
        self.A_aug = np.block(
            [
                [self.A, self.B_d],
                [np.zeros((self.n_d, self.n_x)), np.eye(self.n_d)],
            ]
        )
        self.B_aug = np.vstack([self.B, np.zeros((self.n_d, self.n_u))])
        self.C_aug = np.hstack([self.C_y, np.zeros((self.n_y, self.n_d))])

        if Qw is None:
            Qw = 1e-4 * np.eye(self.n_x + self.n_d)
        if Rv is None:
            Rv = 1e-3 * np.eye(self.n_y)
        self.Qw = np.asarray(Qw, dtype=float)
        self.Rv = np.asarray(Rv, dtype=float)

        if self.Qw.shape != (self.n_x + self.n_d, self.n_x + self.n_d):
            raise ValueError("Qw must have shape (n_x+n_d, n_x+n_d)")
        if self.Rv.shape != (self.n_y, self.n_y):
            raise ValueError("Rv must have shape (n_y, n_y)")

        P = la.solve_discrete_are(self.A_aug.T, self.C_aug.T, self.Qw, self.Rv)
        S = self.C_aug @ P @ self.C_aug.T + self.Rv
        self.L = P @ self.C_aug.T @ np.linalg.inv(S)

        # 3) Target generator: solve steady-state equations for (x_s, u_s)
        # [A-I, B; H, 0] [x_s; u_s] = [-B_d d_hat; r]
        # This matrix is square only when n_r == n_u. If not square, we use
        # a minimum-norm least-squares solution via pseudoinverse.
        self._S_ss = np.block(
            [
                [self.A - np.eye(self.n_x), self.B],
                [self.H, np.zeros((self.n_r, self.n_u))],
            ]
        )

        if self._S_ss.shape[0] == self._S_ss.shape[1]:
            try:
                self._S_ss_solver = np.linalg.inv(self._S_ss)
            except np.linalg.LinAlgError:
                self._S_ss_solver = np.linalg.pinv(self._S_ss)
        else:
            self._S_ss_solver = np.linalg.pinv(self._S_ss)

        # Online state
        self.x_aug_hat = np.zeros(self.n_x + self.n_d)
        self.x_aug_pred = np.zeros(self.n_x + self.n_d)
        self.r = np.zeros(self.n_r)

    def reset(self, *, x0: np.ndarray | None = None, d0: np.ndarray | None = None) -> None:
        if x0 is None:
            x0 = np.zeros(self.n_x)
        if d0 is None:
            d0 = np.zeros(self.n_d)
        x0 = np.asarray(x0, dtype=float).ravel()
        d0 = np.asarray(d0, dtype=float).ravel()
        if x0.size != self.n_x:
            raise ValueError(f"x0 must have length {self.n_x}")
        if d0.size != self.n_d:
            raise ValueError(f"d0 must have length {self.n_d}")
        self.x_aug_hat = np.concatenate([x0, d0])
        self.x_aug_pred = self.x_aug_hat.copy()

    def set_reference(self, r: np.ndarray) -> None:
        r = np.asarray(r, dtype=float).ravel()
        if r.size != self.n_r:
            raise ValueError(f"reference r must have length {self.n_r}")
        self.r = r.copy()

    def steady_state_targets(self, d_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d_hat = np.asarray(d_hat, dtype=float).ravel()
        if d_hat.size != self.n_d:
            raise ValueError(f"d_hat must have length {self.n_d}")
        rhs = np.concatenate([-self.B_d @ d_hat, self.r])
        sol = self._S_ss_solver @ rhs
        x_s = sol[: self.n_x]
        u_s = sol[self.n_x :]
        return x_s, u_s

    def step(self, y_meas: np.ndarray) -> np.ndarray:
        """Run one offset-free MPC step.

        Args:
            y_meas: measured output (n_y,). If you measure full state, pass x.
        Returns:
            u: control action (n_u,)
        """
        y_meas = np.asarray(y_meas, dtype=float).ravel()
        if y_meas.size != self.n_y:
            raise ValueError(f"y_meas must have length {self.n_y}")

        # 1) Kalman correction
        # print(f"y_meas: {y_meas.shape}, x_aug_pred: {self.x_aug_pred.shape}, C_aug: {self.C_aug.shape}")
        innovation = y_meas - (self.C_aug @ self.x_aug_pred)
        self.x_aug_hat = self.x_aug_pred + self.L @ innovation

        x_hat = self.x_aug_hat[: self.n_x]
        d_hat = self.x_aug_hat[self.n_x :]

        # 2) Targets
        x_s, u_s = self.steady_state_targets(d_hat)

        # 3) Control law
        u = u_s + (self.K @ (x_hat - x_s))

        # 4) Prediction
        self.x_aug_pred = self.A_aug @ self.x_aug_hat + self.B_aug @ u
        return u



    # Backwards-compatible alias
    # BatchMPCReferenceTracking = BatchMPCOffsetFreeReferenceTracking