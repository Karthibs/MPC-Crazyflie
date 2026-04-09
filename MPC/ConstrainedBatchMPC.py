import numpy as np
import scipy.linalg as la

from MPC.drone_params import STATE_ORDER_12, output_matrix
from qpsolvers import solve_qp

def build_condensed_state_box_constraints(
    *,
    Omega: np.ndarray,
    Gamma: np.ndarray,
    horizon: int,
    states: tuple[str, ...] | list[str],
    lower: np.ndarray | list[float] | float | None = None,
    upper: np.ndarray | list[float] | float | None = None,
    state_order: tuple[str, ...] = STATE_ORDER_12,
    terminal_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build condensed constraints on selected predicted states.

    Returns matrices (A_ineq, E_x0, b0) such that the QP constraints are

        A_ineq @ U <= b0 + E_x0 @ x0

    where
        U = [u_k, u_{k+1}, ..., u_{k+N-1}] stacked.

    The constrained predicted outputs are
        Y = Cbar @ X = Cbar @ (Omega x0 + Gamma U)

    and box constraints
        lower <= Y <= upper

    are converted into affine inequalities in U.
   
    given current state x0, compute:
        b_ineq = b0 + E_x0 @ x0

    and solve:
        A_ineq @ U <= b_ineq
    """
    Omega = np.asarray(Omega, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    if Omega.ndim != 2 or Gamma.ndim != 2:
        raise ValueError("Omega and Gamma must be 2D arrays")

    nN, n = Omega.shape
    if nN % horizon != 0:
        raise ValueError("Omega row dimension must be divisible by horizon")

    n = Omega.shape[1]
    n_state = nN // horizon

    if n_state != len(state_order):
        raise ValueError(
            f"State dimension mismatch: Omega implies {n_state} states, "
            f"but state_order has length {len(state_order)}"
        )

    Csel = output_matrix(states, state_order=state_order)  # shape: (p, n)
    p = Csel.shape[0]

    if terminal_only:
        # Select only x_{k+N}
        Cbar = np.zeros((p, nN))
        Cbar[:, (horizon - 1) * n_state : horizon * n_state] = Csel
        dim_y = p
    else:
        Cbar = np.kron(np.eye(horizon), Csel)
        dim_y = p * horizon

    Y_Gamma = Cbar @ Gamma
    Y_Omega = Cbar @ Omega

    def _expand_bound(bound, name: str) -> np.ndarray | None:
        if bound is None:
            return None

        arr = np.asarray(bound, dtype=float).reshape(-1)

        if arr.size == 1:
            return np.full(dim_y, arr.item(), dtype=float)

        if arr.size == p:
            if terminal_only:
                return arr.copy()
            return np.tile(arr, horizon)

        if arr.size == dim_y:
            return arr.copy()

        raise ValueError(
            f"{name} must be scalar, length {p}, or length {dim_y}; "
            f"got shape {np.asarray(bound).shape}"
        )

    upper_vec = _expand_bound(upper, "upper")
    lower_vec = _expand_bound(lower, "lower")

    if upper_vec is None and lower_vec is None:
        raise ValueError("At least one of lower or upper must be provided")

    A_blocks = []
    E_blocks = []
    b_blocks = []

    # Upper bound: Y <= upper
    # Cbar * Gamma * U <= upper - Cbar * Omega * x0
    if upper_vec is not None:
        A_blocks.append(Y_Gamma)
        E_blocks.append(-Y_Omega)
        b_blocks.append(upper_vec)

    # Lower bound: Y >= lower  <=>  -Y <= -lower
    # -Cbar * Gamma * U <= -lower + Cbar * Omega * x0
    if lower_vec is not None:
        A_blocks.append(-Y_Gamma)
        E_blocks.append(Y_Omega)
        b_blocks.append(-lower_vec)

    A_ineq = np.vstack(A_blocks)
    E_x0 = np.vstack(E_blocks)
    b0 = np.concatenate(b_blocks)

    return A_ineq, E_x0, b0

def build_condensed_input_box_constraints(
    *,
    n_inputs: int,
    n_states: int,
    horizon: int,
    lower: np.ndarray | list[float] | float | None = None,
    upper: np.ndarray | list[float] | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build input box constraints directly on U:
        lower <= U <= upper

    Returns (A_ineq, E_x0, b0) such that
        A_ineq @ U <= b0 + E_x0 @ x0
    with E_x0 = 0 because these constraints do not depend on x0.
    """
    dim_u = n_inputs * horizon
    I = np.eye(dim_u)

    def _expand(bound, name: str):
        if bound is None:
            return None
        arr = np.asarray(bound, dtype=float).reshape(-1)

        if arr.size == 1:
            return np.full(dim_u, arr.item(), dtype=float)
        if arr.size == n_inputs:
            return np.tile(arr, horizon)
        if arr.size == dim_u:
            return arr.copy()

        raise ValueError(
            f"{name} must be scalar, length {n_inputs}, or length {dim_u}; "
            f"got shape {np.asarray(bound).shape}"
        )

    upper_vec = _expand(upper, "upper")
    lower_vec = _expand(lower, "lower")

    if upper_vec is None and lower_vec is None:
        raise ValueError("At least one of lower or upper must be provided")

    A_blocks = []
    b_blocks = []

    if upper_vec is not None:
        A_blocks.append(I)
        b_blocks.append(upper_vec)

    if lower_vec is not None:
        A_blocks.append(-I)
        b_blocks.append(-lower_vec)

    A_ineq = np.vstack(A_blocks)
    b0 = np.concatenate(b_blocks)
    E_x0 = np.zeros((A_ineq.shape[0], n_states))  # can be replaced later if you prefer matching x0 dim

    return A_ineq, E_x0, b0

def stack_condensed_constraints(
    *constraints: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack multiple condensed inequalities of the form
        A_i U <= b_i + E_i x0
    into one:
        A U <= b + E x0
    """
    A_list, E_list, b_list = zip(*constraints)
    A = np.vstack(A_list)
    E = np.vstack(E_list)
    b = np.concatenate(b_list)
    return A, E, b

class ConstrainedBatchMPC:
    def __init__(self, A, B, N, Q_diag, R_diag):
        n = A.shape[0]
        m = B.shape[1]

        Q = np.diag(Q_diag)
        R = np.diag(R_diag)
        Pf = Q

        Omega = np.vstack([np.linalg.matrix_power(A, i) for i in range(1, N + 1)])
        Gamma = np.zeros((n * N, m * N))
        for i in range(N):
            for j in range(N):
                if i >= j:
                    Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B

        Q_bar = la.block_diag(*([Q] * (N - 1) + [Pf]))
        R_bar = la.block_diag(*([R] * N))

        self.n = n
        self.m = m
        self.N = N
        self.Omega = Omega
        self.Gamma = Gamma
        self.Q_bar = Q_bar
        self.R_bar = R_bar

        self.P = 2 * (Gamma.T @ Q_bar @ Gamma + R_bar)
        self.P = (self.P + self.P.T) / 2  # ensure symmetry
        self.Fq = 2 * (self.Gamma.T @ self.Q_bar @ self.Omega)

    def compute_action(self, x_current, A_ineq, E_x0, b0):
        x_current = np.asarray(x_current, dtype=float).ravel()
        q = self.Fq @ x_current
        b_ineq = b0 + E_x0 @ x_current

        U_star = solve_qp(P=self.P, q=q, G=A_ineq, h=b_ineq, solver="osqp")
        if U_star is None:
            raise RuntimeError("QP solver failed or problem is infeasible.")
        return U_star[:self.m]

class ConstrainedBatchMPCReferenceTracking:
    """
    Constrained batch MPC with output reference tracking + integral action.

    This is the constrained analogue of BatchMPCReferencetracking:
        x_tilde = x - x_ref
        z_{k+1} = z_k + dt * (C_track @ x_tilde)
        x_aug = [x_tilde; z]

    The inner MPC is a QP-based constrained batch MPC.
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
        self.state_order = tuple(state_order)
        self.outputs_to_track = tuple(outputs_to_track)

        self.C_track = output_matrix(list(outputs_to_track), state_order=state_order)

        self.n_x = int(Ad.shape[0])
        self.n_u = int(Bd.shape[1])
        self.n_z = int(self.C_track.shape[0])

        # Augmented tracking model
        self.A_aug = np.zeros((self.n_x + self.n_z, self.n_x + self.n_z))
        self.B_aug = np.zeros((self.n_x + self.n_z, self.n_u))

        self.A_aug[: self.n_x, : self.n_x] = Ad
        self.B_aug[: self.n_x, :] = Bd
        self.A_aug[self.n_x :, : self.n_x] = self.dt * self.C_track
        self.A_aug[self.n_x :, self.n_x :] = np.eye(self.n_z)

        Q_vec = np.concatenate([
            np.asarray(Q_x, dtype=float).ravel(),
            np.asarray(Q_z, dtype=float).ravel(),
        ])

        self.mpc = ConstrainedBatchMPC(
            self.A_aug,
            self.B_aug,
            int(horizon),
            Q_vec,
            np.asarray(R, dtype=float).ravel(),
        )

        self.x_ref = np.zeros(self.n_x)
        self.z_int = np.zeros(self.n_z)

        # Augmented state names:
        # actual plant error-state names + integral-state names
        self.aug_state_order = self.state_order + tuple(
            f"int_{name}" for name in self.outputs_to_track
        )

    def reset(self) -> None:
        self.z_int[:] = 0.0

    def set_x_ref(self, x_ref: np.ndarray) -> None:
        x_ref = np.asarray(x_ref, dtype=float).ravel()
        if x_ref.size != self.n_x:
            raise ValueError(f"x_ref must have length {self.n_x}, got {x_ref.size}")
        self.x_ref = x_ref.copy()

    def set_position_ref(self, ref_states: np.ndarray, states: np.ndarray) -> None:
        """
        Convenience setter, same style as your unconstrained class.
        Example:
            tracker.set_position_ref(np.array([0.4, 0.0, 0.0, 0.3]), np.array([2, 6, 7, 8]))
        """
        ref_states = np.asarray(ref_states, dtype=float).ravel()
        states = np.asarray(states)
        if ref_states.size != states.size:
            raise ValueError("ref_states and states must have the same length")
        self.x_ref[states] = ref_states

    def _shift_actual_bounds_to_error_bounds(
        self,
        bounds,
        states: tuple[str, ...] | list[str],
        *,
        terminal_only: bool,
    ):
        """
        Convert actual-state bounds into error-state bounds:
            x_tilde = x - x_ref
        so
            lower_tilde = lower_actual - x_ref_subset
            upper_tilde = upper_actual - x_ref_subset
        """
        if bounds is None:
            return None

        states = tuple(states)
        p = len(states)
        dim_y = p if terminal_only else p * self.mpc.N

        name_to_index = {name: i for i, name in enumerate(self.state_order)}
        ref_subset = np.array([self.x_ref[name_to_index[s]] for s in states], dtype=float)
        ref_full = ref_subset if terminal_only else np.tile(ref_subset, self.mpc.N)

        arr = np.asarray(bounds, dtype=float).reshape(-1)

        if arr.size == 1:
            arr = np.full(dim_y, arr.item(), dtype=float)
        elif arr.size == p:
            arr = arr.copy() if terminal_only else np.tile(arr, self.mpc.N)
        elif arr.size == dim_y:
            arr = arr.copy()
        else:
            raise ValueError(
                f"Bounds must be scalar, length {p}, or length {dim_y}; got shape {np.asarray(bounds).shape}"
            )

        return arr - ref_full

    def build_state_box_constraints(
        self,
        *,
        states: tuple[str, ...] | list[str],
        lower=None,
        upper=None,
        terminal_only: bool = False,
        bounds_are_actual_states: bool = True,
    ):
        """
        Build state constraints for the constrained tracking MPC.

        If bounds_are_actual_states=True, you provide bounds on the physical states x,
        and they are automatically converted to bounds on x_tilde = x - x_ref.

        If bounds_are_actual_states=False, bounds are assumed to already be given
        directly for the augmented optimization state.
        """
        lower_use = lower
        upper_use = upper

        if bounds_are_actual_states:
            lower_use = self._shift_actual_bounds_to_error_bounds(
                lower, states, terminal_only=terminal_only
            )
            upper_use = self._shift_actual_bounds_to_error_bounds(
                upper, states, terminal_only=terminal_only
            )

        return build_condensed_state_box_constraints(
            Omega=self.mpc.Omega,
            Gamma=self.mpc.Gamma,
            horizon=self.mpc.N,
            states=states,
            lower=lower_use,
            upper=upper_use,
            state_order=self.aug_state_order,
            terminal_only=terminal_only,
        )

    def build_input_box_constraints(self, *, lower=None, upper=None):
        """
        Build direct box constraints on the optimization input sequence U.
        These are usually deviation-input constraints if your controller output is u_dev.
        """
        return build_condensed_input_box_constraints(
            n_inputs=self.mpc.m,
            n_states=self.mpc.n,
            horizon=self.mpc.N,
            lower=lower,
            upper=upper,
        )

    def compute_action(
        self,
        x_current: np.ndarray,
        *,
        A_ineq: np.ndarray,
        E_x0: np.ndarray,
        b0: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the constrained MPC action for the current state and updates the integrator.
        """
        x_current = np.asarray(x_current, dtype=float).ravel()
        if x_current.size != self.n_x:
            raise ValueError(f"x_current must have length {self.n_x}, got {x_current.size}")

        x_tilde = x_current - self.x_ref
        self.z_int = self.z_int + self.dt * (self.C_track @ x_tilde)
        x_aug = np.concatenate([x_tilde, self.z_int])

        return self.mpc.compute_action(
            x_current=x_aug,
            A_ineq=A_ineq,
            E_x0=E_x0,
            b0=b0,
        )


class ConstrainedBatchMPCKalmanReferenceTracking:
    """
    Constrained batch MPC with disturbance estimation and reference tracking.

    Plant:
        x_{k+1} = A x_k + B u_k + B_d d_k
        d_{k+1} = d_k
        y_k     = C_y x_k
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
        self.state_order = tuple(state_order)

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

        # Disturbance model x_{k+1} += B_d d, d_{k+1}=d
        if B_d is None:
            self.B_d = np.ones((self.n_x, 1))
        else:
            self.B_d = np.asarray(B_d, dtype=float)

        if self.B_d.shape[0] != self.n_x:
            raise ValueError("B_d must have shape (n_x, n_d)")

        self.n_d = int(self.B_d.shape[1])

        # Tracked output matrix H x_s = r
        self.H = output_matrix(list(outputs_to_track), state_order=state_order)
        self.n_r = int(self.H.shape[0])

        # Inner constrained regulator on deviation dynamics
        self.mpc = ConstrainedBatchMPC(
            self.A,
            self.B,
            int(horizon),
            np.asarray(Qx, dtype=float).ravel(),
            np.asarray(Ru, dtype=float).ravel(),
        )

        # Estimator model for augmented [x; d]
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

        # Steady-state target solver:
        # [A-I, B; H, 0] [x_s; u_s] = [-B_d d_hat; r]
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

        # Online estimator state
        self.x_aug_hat = np.zeros(self.n_x + self.n_d)
        self.x_aug_pred = np.zeros(self.n_x + self.n_d)

        # Current reference
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
        x_s = sol[:self.n_x]
        u_s = sol[self.n_x:]
        return x_s, u_s

    def _shift_actual_state_bounds(self, states, lower, upper, x_s, terminal_only=False):
        """
        Convert actual physical state bounds into deviation-state bounds:
            x_tilde = x - x_s
        so:
            lower_tilde = lower_actual - x_s_subset
            upper_tilde = upper_actual - x_s_subset
        """
        states = tuple(states)
        name_to_index = {name: i for i, name in enumerate(self.state_order)}
        subset = np.array([x_s[name_to_index[s]] for s in states], dtype=float)

        p = len(states)
        dim_y = p if terminal_only else p * self.mpc.N
        subset_full = subset if terminal_only else np.tile(subset, self.mpc.N)

        def _expand(bound):
            if bound is None:
                return None
            arr = np.asarray(bound, dtype=float).reshape(-1)
            if arr.size == 1:
                arr = np.full(dim_y, arr.item(), dtype=float)
            elif arr.size == p:
                arr = arr.copy() if terminal_only else np.tile(arr, self.mpc.N)
            elif arr.size == dim_y:
                arr = arr.copy()
            else:
                raise ValueError(
                    f"Bounds must be scalar, length {p}, or length {dim_y}; got shape {np.asarray(bound).shape}"
                )
            return arr

        lower_use = _expand(lower)
        upper_use = _expand(upper)

        if lower_use is not None:
            lower_use = lower_use - subset_full
        if upper_use is not None:
            upper_use = upper_use - subset_full

        return lower_use, upper_use

    def _shift_actual_input_bounds(self, lower, upper, u_s):
        """
        Convert actual input bounds into deviation-input bounds:
            u_tilde = u - u_s
        so:
            lower_tilde = lower_actual - u_s
            upper_tilde = upper_actual - u_s
        """
        dim_u = self.n_u * self.mpc.N
        u_s = np.asarray(u_s, dtype=float).ravel()
        u_s_full = np.tile(u_s, self.mpc.N)

        def _expand(bound):
            if bound is None:
                return None
            arr = np.asarray(bound, dtype=float).reshape(-1)
            if arr.size == 1:
                arr = np.full(dim_u, arr.item(), dtype=float)
            elif arr.size == self.n_u:
                arr = np.tile(arr, self.mpc.N)
            elif arr.size == dim_u:
                arr = arr.copy()
            else:
                raise ValueError(
                    f"Bounds must be scalar, length {self.n_u}, or length {dim_u}; got shape {np.asarray(bound).shape}"
                )
            return arr

        lower_use = _expand(lower)
        upper_use = _expand(upper)

        if lower_use is not None:
            lower_use = lower_use - u_s_full
        if upper_use is not None:
            upper_use = upper_use - u_s_full

        return lower_use, upper_use

    def step(
        self,
        y_meas: np.ndarray,
        *,
        state_constraints: list[dict] | None = None,
        input_lower=None,
        input_upper=None,
    ) -> np.ndarray:
        """
        Run one constrained offset-free MPC step.

        Parameters
        ----------
        y_meas : ndarray
            Measured output (n_y,)
        state_constraints : list[dict] | None
            Example:
                [
                    {
                        "states": ("z", "roll", "pitch"),
                        "lower": np.array([0.1, -0.35, -0.35]),
                        "upper": np.array([0.45, 0.35, 0.35]),
                        "terminal_only": False,
                    }
                ]
            These bounds are interpreted as ACTUAL physical-state bounds and
            internally shifted to deviation-state bounds using x_s.
        input_lower, input_upper :
            ACTUAL physical input bounds, internally shifted to deviation-input
            bounds using u_s.
        """
        y_meas = np.asarray(y_meas, dtype=float).ravel()
        if y_meas.size != self.n_y:
            raise ValueError(f"y_meas must have length {self.n_y}")

        # 1) Kalman correction
        innovation = y_meas - (self.C_aug @ self.x_aug_pred)
        self.x_aug_hat = self.x_aug_pred + self.L @ innovation

        x_hat = self.x_aug_hat[:self.n_x]
        d_hat = self.x_aug_hat[self.n_x:]

        # 2) Steady-state targets
        x_s, u_s = self.steady_state_targets(d_hat)

        # 3) Build shifted constraints in deviation variables
        constraint_blocks = []

        if state_constraints is not None:
            for spec in state_constraints:
                states = spec["states"]
                lower = spec.get("lower", None)
                upper = spec.get("upper", None)
                terminal_only = spec.get("terminal_only", False)

                lower_tilde, upper_tilde = self._shift_actual_state_bounds(
                    states=states,
                    lower=lower,
                    upper=upper,
                    x_s=x_s,
                    terminal_only=terminal_only,
                )

                block = build_condensed_state_box_constraints(
                    Omega=self.mpc.Omega,
                    Gamma=self.mpc.Gamma,
                    horizon=self.mpc.N,
                    states=states,
                    lower=lower_tilde,
                    upper=upper_tilde,
                    state_order=self.state_order,
                    terminal_only=terminal_only,
                )
                constraint_blocks.append(block)

        if input_lower is not None or input_upper is not None:
            lower_tilde, upper_tilde = self._shift_actual_input_bounds(
                lower=input_lower,
                upper=input_upper,
                u_s=u_s,
            )

            block = build_condensed_input_box_constraints(
                n_inputs=self.mpc.m,
                n_states=self.mpc.n,
                horizon=self.mpc.N,
                lower=lower_tilde,
                upper=upper_tilde,
            )
            constraint_blocks.append(block)

        if len(constraint_blocks) == 0:
            raise ValueError("At least one constraint block must be provided for constrained MPC.")

        A_ineq, E_x0, b0 = stack_condensed_constraints(*constraint_blocks)

        # 4) Constrained control on deviation variables
        x_tilde = x_hat - x_s
        u_tilde = self.mpc.compute_action(
            x_current=x_tilde,
            A_ineq=A_ineq,
            E_x0=E_x0,
            b0=b0,
        )

        u = u_s + u_tilde

        # 5) Prediction
        self.x_aug_pred = self.A_aug @ self.x_aug_hat + self.B_aug @ u
        return u