"""Python translation of the MATLAB residual-observer construction in FGH.m."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import io, linalg, signal

from enc_func import Mod

STATE_DIM = 24
OUTPUT_DIM = 6
PIVOT_INDEX = 19
KEEP_INDICES = np.array([idx for idx in range(STATE_DIM) if idx != PIVOT_INDEX], dtype=int)
T1_TEMPLATE = np.eye(STATE_DIM, dtype=object)[KEEP_INDICES]
IDENTITY_6_OBJECT = np.eye(OUTPUT_DIM, dtype=object)


@dataclass(frozen=True)
class PlantConfig:
    """Physical plant parameters and sample time."""

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
    """LQR weighting matrices represented by their diagonal entries."""

    Q_diag: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    R_diag: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class FGHConfig:
    """Top-level configuration for the plant discretization and controller design."""

    plant: PlantConfig = PlantConfig()
    lqr: LQRConfig = LQRConfig()


@dataclass
class ModelData:
    """Container for the plant, observer, and reconstruction matrices."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    K: np.ndarray
    F_bar: np.ndarray
    G_bar: np.ndarray
    H: np.ndarray
    Phi_pinv: np.ndarray
    Phi: np.ndarray
    li_vec: np.ndarray
    subset_phi_bar_pinvs: tuple[np.ndarray, ...]


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Build the standard discrete-time observability matrix."""

    n = A.shape[0]
    blocks = [C @ np.linalg.matrix_power(A, k) for k in range(n)]
    return np.vstack(blocks)


def build_plant_matrices(config: PlantConfig = PlantConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the discrete-time plant used in the MATLAB script."""

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
    """Solve the discrete-time Riccati equation and return the MATLAB-sign gain."""

    Q = np.diag(np.asarray(config.Q_diag, dtype=np.float64))
    R = np.diag(np.asarray(config.R_diag, dtype=np.float64))
    X = linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A)
    return -K


@lru_cache(maxsize=1)
def _build_model_data_cached(config: FGHConfig) -> ModelData:
    """Reproduce the observer construction and residual map from FGH.m."""

    A, B, C = build_plant_matrices(config.plant)
    K = discrete_lqr_gain(A, B, config.lqr)

    sensor_rows = [C[idx : idx + 1, :] for idx in range(C.shape[0])]

    phi_obs_list = []
    F_raw = []
    G_raw = []
    H_raw = []

    for Ci in sensor_rows:
        Oi = observability_matrix(A, Ci)
        li_i = int(np.linalg.matrix_rank(Oi))
        Oki = Oi[:li_i, :]

        Qobs, _ = linalg.qr(Oki.T, mode="economic")
        phi_i_obs = Qobs.T
        phi_i_p = np.linalg.pinv(phi_i_obs)

        F_raw.append(phi_i_obs @ A @ phi_i_p)
        G_raw.append(phi_i_obs @ B)
        H_raw.append(Ci @ phi_i_p)
        phi_obs_list.append(phi_i_obs)

    phi_final = []
    F_can = []
    G_can = []
    H_can = []

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
    F_bar_blocks = [Fi - Li @ Hi for Fi, Li, Hi in zip(F_can, L_list, H_can)]
    G_left = np.vstack(G_can)
    G_right = linalg.block_diag(*L_list)
    G_bar = np.hstack([G_left, G_right])
    F_bar = linalg.block_diag(*F_bar_blocks)

    Phi = np.vstack(phi_final)
    Phi_pinv = np.linalg.pinv(Phi)

    li_vec = np.array([phi_i.shape[0] for phi_i in phi_final], dtype=int)
    cum_li = np.concatenate(([0], np.cumsum(li_vec)))

    phi_bar_pinv_list = []
    sensor_combinations = list(combinations(range(len(phi_final)), 3))
    for idx_tuple in sensor_combinations:
        phi_bar = np.vstack([phi_final[idx] for idx in idx_tuple])
        phi_bar_pinv_list.append(np.linalg.pinv(phi_bar))

    H_k = []
    z_dim = int(cum_li[-1])
    for idx_tuple, Pk in zip(sensor_combinations, phi_bar_pinv_list):
        Hk = np.zeros((A.shape[0], z_dim), dtype=np.float64)
        col_start = 0

        for sensor_idx in idx_tuple:
            lj = li_vec[sensor_idx]
            cols_local = slice(col_start, col_start + lj)
            cols_global = slice(cum_li[sensor_idx], cum_li[sensor_idx + 1])
            Hk[:, cols_global] = Pk[:, cols_local]
            col_start += lj

        H_k.append(Hk)

    H = np.zeros((A.shape[0] * len(H_k), z_dim), dtype=np.float64)
    for block_idx, Hk in enumerate(H_k):
        rows = slice(A.shape[0] * block_idx, A.shape[0] * (block_idx + 1))
        H[rows, :] = Hk - Phi_pinv

    return ModelData(
        A=A,
        B=B,
        C=C,
        K=K,
        F_bar=F_bar,
        G_bar=G_bar,
        H=H,
        Phi_pinv=Phi_pinv,
        Phi=Phi,
        li_vec=li_vec,
        subset_phi_bar_pinvs=tuple(phi_bar_pinv_list),
    )


def build_model_data(config: FGHConfig | None = None) -> ModelData:
    """Build the observer model for the provided configuration."""

    if config is None:
        config = FGHConfig()
    return _build_model_data_cached(config)


def build_TV(H1: np.ndarray, q: int):
    """Build the channel-specific coordinate transform and its modular inverse."""

    H1 = np.asarray(H1, dtype=object).reshape(STATE_DIM)
    T1 = T1_TEMPLATE.copy()
    T2 = H1.reshape(1, STATE_DIM)
    T = np.vstack([T1, T2])

    h_pivot = int(H1[PIVOT_INDEX]) % q
    if h_pivot == 0:
        raise ValueError(f"H1[{PIVOT_INDEX}] is zero modulo q and cannot be used as the pivot entry.")

    inv_h_pivot = pow(h_pivot, -1, q)
    V = np.zeros((STATE_DIM, STATE_DIM), dtype=object)

    V[KEEP_INDICES, np.arange(STATE_DIM - 1)] = 1
    keep_values = np.array([int(H1[idx]) % q for idx in KEEP_INDICES], dtype=object)
    V[PIVOT_INDEX, : STATE_DIM - 1] = ((-keep_values) * inv_h_pivot) % q
    V[PIVOT_INDEX, STATE_DIM - 1] = inv_h_pivot % q

    V1 = V[:, : STATE_DIM - 1].copy()
    V2 = V[:, STATE_DIM - 1].reshape(STATE_DIM, 1)
    return T1, T2, T, V, V1, V2


def float_to_object_int(arr_float: np.ndarray) -> np.ndarray:
    """Round a numeric array and return an object-dtype array of Python integers."""

    rounded = np.rint(np.asarray(arr_float, dtype=np.float64))
    if not np.isfinite(rounded).all():
        raise ValueError("Quantization produced a non-finite value.")

    values = [int(value) for value in rounded.flat]
    return np.array(values, dtype=object).reshape(rounded.shape)


def load_fgh_matrices(config: FGHConfig | None = None) -> dict[str, np.ndarray]:
    """Return the observer matrices generated by the Python translation."""

    model = build_model_data(config)
    return {
        "F_bar": model.F_bar,
        "G_bar": model.G_bar,
        "H": model.H,
        "Phi_pinv": model.Phi_pinv,
    }


@lru_cache(maxsize=None)
def _quantized_base_matrices(scale: int, config: FGHConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache the quantized observer matrices for repeated offline preprocessing."""

    data = load_fgh_matrices(config)
    F_bar = float_to_object_int(data["F_bar"])
    G_bar = float_to_object_int(scale * data["G_bar"])
    H_bar = float_to_object_int(scale * data["H"])
    Phi_pinv_bar = float_to_object_int(scale * data["Phi_pinv"])
    return F_bar, G_bar, H_bar, Phi_pinv_bar


def compute_offline_mats(env, s=100000, num_channels=60, config: FGHConfig | None = None, model=None):
    """Quantize the FGH.py matrices and derive all per-channel offline terms."""

    if config is None:
        config = FGHConfig()
    scale = int(s)
    if model is not None:
        F_bar = float_to_object_int(model.F_bar)
        G_bar = float_to_object_int(scale * model.G_bar)
        H_bar = float_to_object_int(scale * model.H)
        Phi_pinv_bar = float_to_object_int(scale * model.Phi_pinv)
    else:
        F_bar, G_bar, H_bar, Phi_pinv_bar = _quantized_base_matrices(scale, config)

    T1_all = []
    T2_all = []
    V1_all = []
    V2_all = []
    S_xi_all = []
    S_v_all = []
    Psi_all = []
    Sigma_all = []
    Sigma_pinv_all = []

    for channel_idx in range(num_channels):
        H1 = H_bar[channel_idx, :]
        T1, T2, _, _, V1, V2 = build_TV(H1, env.q)

        H1_row = H1.reshape(1, -1)
        S_1 = Mod(T1 @ F_bar @ V1, env.q)
        S_3 = Mod(T1 @ G_bar, env.q)
        Psi = Mod(H1_row @ F_bar @ V1, env.q)
        Sigma = Mod(H1_row @ G_bar, env.q)

        sigma0 = int(Sigma[0, 0])
        if sigma0 == 0:
            raise ValueError(f"[channel {channel_idx}] Sigma[0,0] is zero modulo q.")

        Sigma_pinv = np.zeros((OUTPUT_DIM, 1), dtype=object)
        Sigma_pinv[0, 0] = pow(sigma0, -1, env.q)

        S_xi = Mod(S_1 - S_3 @ Sigma_pinv @ Psi, env.q)
        S_v = Mod(S_3 @ (IDENTITY_6_OBJECT - Sigma_pinv @ Sigma), env.q)

        T1_all.append(T1)
        T2_all.append(T2)
        V1_all.append(V1)
        V2_all.append(V2)
        S_xi_all.append(S_xi)
        S_v_all.append(S_v)
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
        "V1_all": np.stack(V1_all, axis=0),
        "V2_all": np.stack(V2_all, axis=0),
        "S_xi_all": np.stack(S_xi_all, axis=0),
        "S_v_all": np.stack(S_v_all, axis=0),
        "Psi_all": np.stack(Psi_all, axis=0),
        "Sigma_all": np.stack(Sigma_all, axis=0),
        "Sigma_pinv_all": np.array(Sigma_pinv_all, dtype=object),
    }


def save_offline_mats(offline, filename="matrices.npz"):
    """Persist the precomputed offline matrices for later reuse."""

    np.savez(filename, **offline)


def _to_object_int(arr):
    """Convert a loaded numeric array to Python integers backed by object dtype."""

    if arr.dtype == object:
        return arr
    return arr.astype(int).astype(object)


def load_offline_mats(filename="matrices.npz"):
    """Load the cached offline matrices and restore object dtype where needed."""

    data = np.load(filename, allow_pickle=True)
    offline = {}

    int_keys = {
        "F_bar",
        "G_bar",
        "H_bar",
        "Phi_pinv_bar",
        "T1_all",
        "T2_all",
        "V1_all",
        "V2_all",
        "S_xi_all",
        "S_v_all",
        "Psi_all",
        "Sigma_all",
    }

    for key in data.files:
        if key in int_keys:
            offline[key] = _to_object_int(data[key])
        elif key == "Sigma_pinv_all":
            offline[key] = np.array([_to_object_int(sigma) for sigma in data[key]], dtype=object)
        else:
            offline[key] = data[key]

    return offline


def save_outputs(data: ModelData, filename: str = "FGHPhi_pinv.mat") -> None:
    """Write the matrices needed by the Python encryption pipeline."""

    io.savemat(
        filename,
        {
            "F_bar": data.F_bar,
            "G_": data.G_bar,
            "H": data.H,
            "Phi_pinv": data.Phi_pinv,
        },
    )


def run_example_simulation(
    data: ModelData,
    iterations: int = 100,
    Ts: float = 0.1,
    show_plots: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the same plaintext closed-loop simulation that appears in FGH.m."""

    n = data.A.shape[0]
    x = np.ones((n, 1), dtype=np.float64)
    z_hat = np.zeros((data.F_bar.shape[0], 1), dtype=np.float64)

    xp_hist = [x.copy()]
    z_hist = [z_hat.copy()]
    y_hist = []
    r_hist = []

    for step_idx in range(iterations):
        if 50 < step_idx + 1 < 60:
            attack = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=np.float64)
        else:
            attack = np.zeros((data.C.shape[0], 1), dtype=np.float64)

        y_k = data.C @ x + attack
        r_k = data.H @ z_hat
        u_k = data.K @ x

        x = data.A @ x + data.B @ u_k
        z_hat = data.F_bar @ z_hat + data.G_bar @ np.vstack([u_k, y_k])

        y_hist.append(y_k)
        r_hist.append(r_k)
        xp_hist.append(x.copy())
        z_hist.append(z_hat.copy())

    xp = np.hstack(xp_hist)
    z_hat_arr = np.hstack(z_hist)
    y = np.hstack(y_hist)
    r = np.hstack(r_hist)

    t = Ts * np.arange(iterations)
    t_x = Ts * np.arange(iterations + 1)
    x_hat = data.Phi_pinv @ z_hat_arr

    plt.figure()
    plt.plot(t, y.T)
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y")
    plt.title("Plant outputs y")

    fig, axes = plt.subplots(6, 1, sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(t_x, xp[idx, :], "b", linewidth=1.2)
        ax.plot(t_x, x_hat[idx, :], "r--", linewidth=1.2)
        ax.grid(True)
        ax.set_ylabel(f"x_{idx + 1}")

    axes[0].set_title("Plant states x_i vs estimated states x_hat_i")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()

    r_inf = np.linalg.norm(r, ord=np.inf, axis=0)
    plt.figure()
    plt.plot(t, r_inf)
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("||r||_inf")
    plt.title("Residual norm ||r(k)||_inf")
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return xp, x_hat, r


def compute_bounds(
    data: ModelData,
    s_quant: float = 1e5,
    r_quant: float = 1e5,
    xp0: np.ndarray | None = None,
    z_hat0: np.ndarray | None = None,
) -> dict[str, float]:
    """Reproduce the parameter and bound calculations from FGH.m."""

    n = data.A.shape[0]
    p = data.C.shape[0]
    m = data.B.shape[1]

    if xp0 is None:
        xp0 = np.ones((n, 1), dtype=np.float64)
    if z_hat0 is None:
        z_hat0 = np.zeros((data.F_bar.shape[0], 1), dtype=np.float64)

    z_tilde = np.linalg.norm(data.Phi @ xp0 - z_hat0, ord=np.inf)
    l_max = float(np.max(data.li_vec))
    inf_norms = [np.linalg.norm(Pk, ord=np.inf) for Pk in data.subset_phi_bar_pinvs]
    kappa = max(np.linalg.norm(data.Phi_pinv, ord=np.inf), max(inf_norms))
    G_norm = np.linalg.norm(data.G_bar, ord=np.inf)

    KC = np.vstack([data.K, data.C])
    vmax = np.linalg.norm(KC, ord=np.inf) * np.linalg.norm(xp0, ord=np.inf)
    M = max(1.0, np.linalg.norm(data.H, ord=np.inf)) * (
        np.linalg.norm(z_hat0, ord=np.inf) + l_max * np.linalg.norm(data.G_bar, ord=np.inf) * vmax
    )
    eini = 1.0 / (2.0 * r_quant * s_quant)
    ez = (m + p) / (2.0 * s_quant) * vmax + np.linalg.norm(data.G_bar, ord=np.inf) / (2.0 * r_quant)
    ez += (m + p) / (4.0 * r_quant * s_quant)
    epsilonz = eini + l_max * ez
    epsilon = 2.0 * (kappa * epsilonz + data.Phi.shape[0] / (2.0 * s_quant) * (M + epsilonz))

    transient = kappa * z_tilde
    L = 2.0 * (kappa * s_quant + 12.0) * (1.0 + 6.0 * (s_quant * G_norm + 4.0))

    return {
        "z_tilde": float(z_tilde),
        "l_max": float(l_max),
        "kappa": float(kappa),
        "G_norm": float(G_norm),
        "vmax": float(vmax),
        "M": float(M),
        "eini": float(eini),
        "ez": float(ez),
        "epsilonz": float(epsilonz),
        "epsilon": float(epsilon),
        "transient": float(transient),
        "L": float(L),
    }


def print_numpy(name: str, M: np.ndarray, prec: int = 4) -> None:
    """Print a matrix in a NumPy-friendly literal format."""

    fmt = f"{{:.{prec}f}}"
    M = np.asarray(M)

    if M.ndim == 1 or 1 in M.shape:
        flat = M.reshape(-1)
        body = ", ".join(fmt.format(value) for value in flat)
        print(f"{name} = np.array([ {body} ], dtype=np.float64)  # shape ({flat.size},)")
        print()
        return

    print(f"{name} = np.array([")
    for row_idx, row in enumerate(M):
        body = ", ".join(fmt.format(value) for value in row)
        suffix = "," if row_idx < M.shape[0] - 1 else ""
        print(f"    [ {body} ]{suffix}")
    print("], dtype=np.float64)")
    print()


def compare_with_matlab_export(data: ModelData, filename: str = "FGHPhi_pinv.mat") -> dict[str, float]:
    """Report the max-abs error against the existing MATLAB export, if present."""

    ref = io.loadmat(filename)
    return {
        "F_bar": float(np.max(np.abs(data.F_bar - ref["F_bar"]))),
        "G_": float(np.max(np.abs(data.G_bar - ref["G_"]))),
        "H": float(np.max(np.abs(data.H - ref["H"]))),
        "Phi_pinv": float(np.max(np.abs(data.Phi_pinv - ref["Phi_pinv"]))),
    }


def main() -> None:
    """Build the matrices, save them, compare with MATLAB, and run the demo."""

    data = build_model_data()

    try:
        errors = compare_with_matlab_export(data)
        for name, err in errors.items():
            print(f"{name} max abs diff: {err:.3e}")
    except FileNotFoundError:
        pass

    save_outputs(data)
    print_numpy("A", data.A, 6)
    print_numpy("B", data.B, 6)
    print_numpy("C", data.C, 0)
    print_numpy("K", data.K, 6)
    bounds = compute_bounds(data)
    for name, value in bounds.items():
        print(f"{name} = {value:.6e}")
    run_example_simulation(data)


if __name__ == "__main__":
    main()
