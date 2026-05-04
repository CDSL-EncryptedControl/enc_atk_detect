"""Observer and offline-matrix builder used by simulation.ipynb."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy import linalg, signal

from enc_func import Mod

STATE_DIM = 24
OUTPUT_DIM = 6
PIVOT_INDEX = 19
KEEP_INDICES = np.array([idx for idx in range(STATE_DIM) if idx != PIVOT_INDEX], dtype=int)
T1_TEMPLATE = np.eye(STATE_DIM, dtype=object)[KEEP_INDICES]
IDENTITY_6_OBJECT = np.eye(OUTPUT_DIM, dtype=object)


@dataclass(frozen=True)
class PlantConfig:
    J1: float = 0.01
    J2: float = 0.01
    J3: float = 0.01
    k1: float = 1.37
    k2: float = 1.37
    b1: float = 0.007
    b2: float = 0.007
    b3: float = 0.007
    Ts: float = 0.1


@dataclass(frozen=True)
class LQRConfig:
    Q_diag: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    R_diag: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class FGHConfig:
    plant: PlantConfig = PlantConfig()
    lqr: LQRConfig = LQRConfig()


@dataclass
class ModelData:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    K: np.ndarray
    F_bar: np.ndarray
    G_bar: np.ndarray
    H: np.ndarray
    Phi_pinv: np.ndarray


@dataclass
class SimulationData:
    model: ModelData
    offline: dict[str, np.ndarray]


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(A.shape[0])])


def build_plant_matrices(config: PlantConfig = PlantConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A0 = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-config.k1 / config.J1, -config.b1 / config.J1, config.k1 / config.J1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [
            config.k1 / config.J2,
            0.0,
            -(config.k1 + config.k2) / config.J2,
            -config.b2 / config.J2,
            config.k2 / config.J2,
            0.0,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, config.k2 / config.J3, 0.0, -config.k2 / config.J3, -config.b3 / config.J3],
    ], dtype=np.float64)
    B0 = np.array([[0.0], [1.0 / config.J1], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
    C = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    ], dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    Ad, Bd, Cd, _, _ = signal.cont2discrete((A0, B0, C, D), config.Ts, method="zoh")
    return Ad, Bd, Cd


def discrete_lqr_gain(A: np.ndarray, B: np.ndarray, config: LQRConfig = LQRConfig()) -> np.ndarray:
    Q = np.diag(np.asarray(config.Q_diag, dtype=np.float64))
    R = np.diag(np.asarray(config.R_diag, dtype=np.float64))
    X = linalg.solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A)


@lru_cache(maxsize=None)
def _build_model_data_cached(config: FGHConfig) -> ModelData:
    A, B, C = build_plant_matrices(config.plant)
    K = discrete_lqr_gain(A, B, config.lqr)

    sensor_rows = [C[idx : idx + 1, :] for idx in range(C.shape[0])]
    F_raw, G_raw, H_raw, phi_obs_list = [], [], [], []

    for Ci in sensor_rows:
        Oi = observability_matrix(A, Ci)
        li_i = int(np.linalg.matrix_rank(Oi))
        phi_i_obs = linalg.qr(Oi[:li_i, :].T, mode="economic")[0].T
        phi_i_p = np.linalg.pinv(phi_i_obs)
        F_raw.append(phi_i_obs @ A @ phi_i_p)
        G_raw.append(phi_i_obs @ B)
        H_raw.append(Ci @ phi_i_p)
        phi_obs_list.append(phi_i_obs)

    phi_final, F_can, G_can, H_can = [], [], [], []
    for Fi, Gi, Hi, phi_i_obs in zip(F_raw, G_raw, H_raw, phi_obs_list):
        li_i = Fi.shape[0]
        Oi_small = observability_matrix(Fi, Hi)
        e_last = np.zeros((li_i, 1), dtype=np.float64)
        e_last[-1, 0] = 1.0
        basis = np.linalg.solve(Oi_small, e_last)
        inv_T = np.zeros((li_i, li_i), dtype=np.float64)
        v = basis.copy()
        for idx in range(li_i):
            inv_T[:, idx : idx + 1] = v
            v = Fi @ v
        T_i = np.linalg.inv(inv_T)
        T_i_inv = np.linalg.inv(T_i)
        F_can.append(T_i @ Fi @ T_i_inv)
        G_can.append(T_i @ Gi)
        H_can.append(Hi @ T_i_inv)
        phi_final.append(T_i @ phi_i_obs)

    L_list = [Fi[:, [-1]] for Fi in F_can]
    F_bar = linalg.block_diag(*[Fi - Li @ Hi for Fi, Li, Hi in zip(F_can, L_list, H_can)])
    G_bar = np.hstack([np.vstack(G_can), linalg.block_diag(*L_list)])

    Phi = np.vstack(phi_final)
    Phi_pinv = np.linalg.pinv(Phi)
    li_vec = np.array([phi_i.shape[0] for phi_i in phi_final], dtype=int)
    cum_li = np.concatenate(([0], np.cumsum(li_vec)))

    H_blocks = []
    for idx_tuple in combinations(range(len(phi_final)), 3):
        Pk = np.linalg.pinv(np.vstack([phi_final[idx] for idx in idx_tuple]))
        Hk = np.zeros((A.shape[0], cum_li[-1]), dtype=np.float64)
        col_start = 0
        for sensor_idx in idx_tuple:
            lj = li_vec[sensor_idx]
            Hk[:, cum_li[sensor_idx] : cum_li[sensor_idx + 1]] = Pk[:, col_start : col_start + lj]
            col_start += lj
        H_blocks.append(Hk - Phi_pinv)

    H = np.vstack(H_blocks)
    return ModelData(A=A, B=B, C=C, K=K, F_bar=F_bar, G_bar=G_bar, H=H, Phi_pinv=Phi_pinv)


def build_model_data(config: FGHConfig | None = None) -> ModelData:
    return _build_model_data_cached(FGHConfig() if config is None else config)


def build_TV(H1: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H1 = np.asarray(H1, dtype=object).reshape(STATE_DIM)
    T1 = T1_TEMPLATE.copy()
    T2 = H1.reshape(1, STATE_DIM)
    inv_h_pivot = pow(int(H1[PIVOT_INDEX]) % q, -1, q)
    V = np.zeros((STATE_DIM, STATE_DIM), dtype=object)
    V[KEEP_INDICES, np.arange(STATE_DIM - 1)] = 1
    keep_values = np.array([int(H1[idx]) % q for idx in KEEP_INDICES], dtype=object)
    V[PIVOT_INDEX, : STATE_DIM - 1] = ((-keep_values) * inv_h_pivot) % q
    V[PIVOT_INDEX, STATE_DIM - 1] = inv_h_pivot % q
    return T1, T2, V[:, : STATE_DIM - 1].copy(), V[:, STATE_DIM - 1].reshape(STATE_DIM, 1)


def float_to_object_int(arr_float: np.ndarray) -> np.ndarray:
    rounded = np.rint(np.asarray(arr_float, dtype=np.float64))
    return np.array([int(value) for value in rounded.flat], dtype=object).reshape(rounded.shape)


def compute_offline_mats(env, s=100000, num_channels=60, config: FGHConfig | None = None, model=None):
    model = build_model_data(config) if model is None else model
    scale = int(s)
    F_bar = float_to_object_int(model.F_bar)
    G_bar = float_to_object_int(scale * model.G_bar)
    H_bar = float_to_object_int(scale * model.H)
    Phi_pinv_bar = float_to_object_int(scale * model.Phi_pinv)

    T1_all, T2_all, V2_all, S_xi_all, S_v_all, Psi_all, Sigma_all, Sigma_pinv_all = [], [], [], [], [], [], [], []
    for channel_idx in range(num_channels):
        H1 = H_bar[channel_idx, :]
        T1, T2, V1, V2 = build_TV(H1, env.q)
        H1_row = H1.reshape(1, -1)
        S_1 = Mod(T1 @ F_bar @ V1, env.q)
        S_3 = Mod(T1 @ G_bar, env.q)
        Psi = Mod(H1_row @ F_bar @ V1, env.q)
        Sigma = Mod(H1_row @ G_bar, env.q)
        Sigma_pinv = np.zeros((OUTPUT_DIM, 1), dtype=object)
        Sigma_pinv[0, 0] = pow(int(Sigma[0, 0]), -1, env.q)
        T1_all.append(T1)
        T2_all.append(T2)
        V2_all.append(V2)
        S_xi_all.append(Mod(S_1 - S_3 @ Sigma_pinv @ Psi, env.q))
        S_v_all.append(Mod(S_3 @ (IDENTITY_6_OBJECT - Sigma_pinv @ Sigma), env.q))
        Psi_all.append(Psi)
        Sigma_all.append(Sigma)
        Sigma_pinv_all.append(Sigma_pinv)

    return {
        "F_bar": F_bar,
        "G_bar": G_bar,
        "H_bar": H_bar,
        "Phi_pinv_bar": Phi_pinv_bar,
        "T1_all": np.stack(T1_all, axis=0),
        "T2_all": np.stack(T2_all, axis=0),
        "V2_all": np.stack(V2_all, axis=0),
        "S_xi_all": np.stack(S_xi_all, axis=0),
        "S_v_all": np.stack(S_v_all, axis=0),
        "Psi_all": np.stack(Psi_all, axis=0),
        "Sigma_all": np.stack(Sigma_all, axis=0),
        "Sigma_pinv_all": np.array(Sigma_pinv_all, dtype=object),
    }


def prepare_simulation_data(env, s_quant=100000, num_channels=60, config: FGHConfig | None = None) -> SimulationData:
    model = build_model_data(config)
    offline = compute_offline_mats(env, s_quant, num_channels=num_channels, config=config, model=model)
    return SimulationData(model=model, offline=offline)
