"""Core modular arithmetic and encryption helpers used by the simulations."""

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncryptionConfig:
    """Immutable configuration for the toy encryption scheme."""

    p: int = int(2**65)
    L: int = int(2**44)
    r: float = 19.2
    N: int = int(2**12)
    q_offset: int = 31


class Params:
    """Container for the public parameters of the toy encryption scheme."""

    def __init__(self, config: EncryptionConfig | None = None, **overrides):
        # Parameter set used by the current encrypted observer experiments.
        if config is None:
            config = EncryptionConfig()
        if overrides:
            config = EncryptionConfig(**{**config.__dict__, **overrides})

        self.config = config
        self.p = int(config.p)
        self.L = int(config.L)
        self.r = float(config.r)
        self.N = int(config.N)
        self.q = self.p * self.L - int(config.q_offset)
        self.invL = pow(self.L % self.q, -1, self.q)


def Seret_key(env):
    """Sample a ternary secret key in {-1, 0, 1}^N."""

    key = [random.choice([-1, 0, 1]) for _ in range(env.N)]
    return np.array(key, dtype=object).reshape(-1, 1)


def Mod(x, p):
    """Apply centered modular reduction element-wise over scalars or arrays."""

    x_arr = np.asarray(x, dtype=object)
    out = np.empty(x_arr.shape, dtype=object)
    half_p = p // 2

    it = np.nditer(x_arr, flags=["multi_index", "refs_ok", "zerosize_ok"], op_flags=["readonly"])
    for value in it:
        reduced = int(value.item()) % p
        if reduced >= half_p:
            reduced -= p
        out[it.multi_index] = reduced

    return out


def rand_mod_q(shape, q):
    """Sample an object-dtype array with entries chosen uniformly from [0, q)."""

    total = int(np.prod(shape))
    values = [random.randrange(q) for _ in range(total)]
    return np.array(values, dtype=object).reshape(shape)


def Enc_state(z_hat_bar, sk, env, T1, T2, V2):
    """Encrypt the initial observer state for one residual channel."""

    n = 24
    N = env.N

    z_hat_bar = np.asarray(z_hat_bar, dtype=object).reshape(n, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)

    # Fresh randomness for the state ciphertext.
    A = rand_mod_q((n, N), env.q)
    e = np.random.randint(-env.r, env.r + 1, size=(n, 1)).astype(object)

    b_ini = Mod(A @ sk + e, env.q)
    b_tilde = Mod(T2 @ b_ini, env.q)
    b_xi_ini = Mod(T1 @ b_ini, env.q)
    b_prime = Mod(V2 @ b_tilde, env.q)

    # The first column carries the masked message term.
    c0 = Mod(env.L * z_hat_bar + b_ini - b_prime, env.q)
    c_state = np.hstack([c0, A, b_prime])

    return c_state, b_xi_ini


def Enc_t(v, sk, b_xi, Sigma_pinv, Sigma, Psi, env):
    """Encrypt the current input/output vector with dynamic masking."""

    n_v = 6
    N = env.N

    v = np.asarray(v, dtype=object).reshape(n_v, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)
    b_xi = np.asarray(b_xi, dtype=object).reshape(23, 1)
    Sigma = np.asarray(Sigma, dtype=object).reshape(1, 6)
    Sigma_pinv = np.asarray(Sigma_pinv, dtype=object).reshape(6, 1)
    Psi = np.asarray(Psi, dtype=object).reshape(1, 23)

    Av = rand_mod_q((n_v, N), env.q)
    e = np.random.randint(-env.r, env.r + 1, size=(n_v, 1)).astype(object)

    b_v = Mod(Av @ sk + e, env.q)
    b_prime = Mod(Sigma_pinv @ (Sigma @ b_v + Psi @ b_xi), env.q)

    # The message term stays in the first ciphertext column.
    c0 = Mod(env.L * v + b_v - b_prime, env.q)
    c_t = np.hstack([c0, Av, b_prime])

    return c_t, b_v


def Dec(ciphertext, sk, env):
    """Decrypt a ciphertext matrix and return the scaled integer message."""

    dec_sk = np.vstack([
        np.ones((1, 1), dtype=object),
        -np.asarray(sk, dtype=object).reshape(-1, 1),
        np.ones((1, 1), dtype=object),
    ])
    return Mod(np.asarray(ciphertext, dtype=object) @ dec_sk, env.q)
