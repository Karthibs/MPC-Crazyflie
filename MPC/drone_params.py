from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from scipy.linalg import expm


@dataclass(frozen=True)
class DroneParams:
    """Physical parameters used by the controller."""

    mass: float
    inertia: np.ndarray  # [Ix, Iy, Iz]
    gravity: float = 9.81

    @property
    def Ix(self) -> float:
        return float(self.inertia[0])

    @property
    def Iy(self) -> float:
        return float(self.inertia[1])

    @property
    def Iz(self) -> float:
        return float(self.inertia[2])

    @property
    def hover_force(self) -> float:
        return float(self.mass * self.gravity)

    @classmethod
    def from_env(cls, env, gravity: float = 9.81) -> "DroneParams":
        """Builds params from a QuadrotorEnv-like object (must expose mass/inertia)."""
        return cls(
            mass=float(env.mass),
            inertia=np.asarray(env.inertia, dtype=float).copy(),
            gravity=float(gravity),
        )


@dataclass(frozen=True)
class DroneLinearModel:
    """Discrete- or continuous-time linear model container."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


STATE_ORDER_12: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r",
)


def output_matrix(
    outputs: Sequence[str],
    *,
    state_order: Sequence[str] = STATE_ORDER_12,
) -> np.ndarray:
    """Build a selection matrix C such that y = C x.

    Example:
        C_pos = output_matrix(["x", "y", "z"])  # 3x12
    """
    name_to_index = {name: i for i, name in enumerate(state_order)}
    indices = [name_to_index[name] for name in outputs]
    C = np.zeros((len(indices), len(state_order)))
    for row, idx in enumerate(indices):
        C[row, idx] = 1.0
    return C


def make_hover_12state_continuous_model(params: DroneParams, dt: float) -> DroneLinearModel:
    """12-state hover-linearized continuous model.

    State order:
        [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    Input order:
        [U1(thrust), U2(roll torque), U3(pitch torque), U4(yaw torque)]
    """
    m = float(params.mass)
    g = float(params.gravity)
    Ix, Iy, Iz = params.Ix, params.Iy, params.Iz

    A = np.zeros((12, 12))
    # position dynamics
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0

    # velocity dynamics (small-angle)
    A[3, 7] = g
    A[4, 6] = -g

    # attitude kinematics
    A[6, 9] = 1.0
    A[7, 10] = 1.0
    A[8, 11] = 1.0

    B = np.zeros((12, 4))
    # thrust effect
    B[5, 0] = 1.0 / m
    # torques
    B[9, 1] = 1.0 / Ix
    B[10, 2] = 1.0 / Iy
    B[11, 3] = 1.0 / Iz

    C = np.eye(12)
    D = np.zeros((12, 4))


    dt = dt
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    M = np.zeros((n_states + n_inputs, n_states + n_inputs))
    M[:n_states, :n_states] = A
    M[:n_states, n_states:] = B

    # exp(M * dt) = [[Ad, Bd],
    #                [0,   I ]]
    Md = expm(M * dt)

    Ad = Md[:n_states, :n_states]
    Bd = Md[:n_states, n_states:]
    return DroneLinearModel(A=Ad, B=Bd, C=C, D=D)


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the controllability matrix: [B, AB, A^2B, ..., A^(n-1)B]."""
    n = A.shape[0]
    blocks = [B]
    for i in range(1, n):
        blocks.append(np.linalg.matrix_power(A, i) @ B)
    return np.hstack(blocks)


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute the observability matrix: [C; CA; CA^2; ...; CA^(n-1)]."""
    n = A.shape[0]
    blocks = [C]
    for i in range(1, n):
        blocks.append(C @ np.linalg.matrix_power(A, i))
    return np.vstack(blocks)


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(controllability_matrix(A, B)))


def observability_rank(A: np.ndarray, C: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(observability_matrix(A, C)))
