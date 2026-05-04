<<<<<<< HEAD
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
=======
import numpy as np
import random

# 환경 변수 env 와 암호화 함수 Mod / Enc_state / Enc_t / Dec 

# class Params:
#     def __init__(self):
#         self.p = int(2**54)   # p 
#         self.L = int(2**10)   # L 
#         self.r = 19           # 오류 범위 # 균등분포
#         self.N = 5            # 키 차원
#         self.q = self.p * self.L - 59    # 2^64 근처 소수 18446744073709551557
#         L_mod = self.L % self.q
#         self.invL = pow(L_mod, -1, self.q)


# 환경 변수 설정
# class Params:
#     def __init__(self):
#         # 여기서 K의 최댓값은 355,700,000
#         self.p = int(2**118)  # p 
#         self.L = int(2**10)  # L 
#         self.r = 10         # 오류 범위
#         self.N = 8    # 키 차원 
#         self.q = self.p * self.L -159 # 근처 소수
#         L_mod = self.L % self.q
#         self.invL = pow(L_mod, -1, self.q)

class Params:
    def __init__(self):
        # 여기서 K의 최댓값은 355,700,000
        # self.p = int(2**99)  # p 
        # self.L = int(2**10)  # L 
        self.p = int(2**65)  # p 
        self.L = int(2**44)  # L 
        self.r = 19.2         # 오류 범위
        self.N = int(2**12)    # 키 차원 
        self.q = self.p * self.L -31 # 근처 소수
        L_mod = self.L % self.q
        self.invL = pow(L_mod, -1, self.q)



def Seret_key(env):
    # -1 0 1 균등 분포
    sk = np.array([random.choice([-1, 0, 1]) for _ in range(env.N)], dtype=object)
    # print(sk)
    return sk.reshape(-1, 1)  # N x 1 형태로 반환

def Mod(x, p):
    # -q/2 q/2 인 mod 연산
    x_arr = np.asarray(x, dtype=object)

    def centered(v):
        v_int = int(v)
        r = v_int % p
        if r >= p // 2:
            r -= p
        return r

    return np.vectorize(centered, otypes=[object])(x_arr)

def rand_mod_q(shape, q):
    arr = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        arr[idx] = random.randrange(q)
    return arr


def Enc_state(z_hat_bar, sk, env, T1, T2, V2):
    """
    Enc_state, j 인덱스에 따라 60개

    입력:
        z_hat_bar : (24x1) 양자화된 초기 옵저버 상태
        sk        : (N x 1) 비밀키
        env       : Params()
        T1        : 23x24 
        T2        : 1x24  
        V2        : 24x1  

    출력:
        C_state   : 24 x (N+2) 암호문
        b_xi_ini  : 23 x 1 Enc_t 를 위한 동적 마스킹 파트
    """
    n = 24
    N = env.N

    z_hat_bar = np.asarray(z_hat_bar, dtype=object).reshape(n, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)

    # 1) A, e 
    A = rand_mod_q((n, N), env.q)   # 24 x N, in [0, q-1]
    e = np.random.randint(-env.r, env.r + 1, size=(n, 1)).astype(object)  # 6x1 

    # 2) b_ini = A sk + e
    b_ini = Mod(A @ sk + e, env.q)

    # 3) b_tilde, b_xi_ini
    b_tilde = T2 @ b_ini           # 1x1
    b_tilde = Mod(b_tilde, env.q)

    b_xi_ini = T1 @ b_ini          # 23x1
    b_xi_ini = Mod(b_xi_ini, env.q)

    # 4) b_prime = V2 @ b_tilde
    b_prime = V2 @ b_tilde         # 24x1
    b_prime = Mod(b_prime, env.q)

    # 5) 첫 번째 컬럼
    C0 = Mod(env.L * z_hat_bar + b_ini - b_prime, env.q)  # 24x1

    # 6) 최종 암호문
    C_state = np.hstack([C0, A, b_prime])         # 24 x (N+2)

    return C_state, b_xi_ini


def Enc_t(v, sk, b_xi, Sigma_pinv, Sigma, Psi, env):
    """
    Enc_t (동적 암호화)

    입력:
        v           : 6x1 벡터 (양자화된 [u;y])
        sk          : N x 1 비밀키
        b_xi        : 23 x 1 식 (21) 연산을 위해 밖으로 뺌
        Sigma_pinv  : 6 x 1
        Sigma       : 1 x 6
        Psi         : 1 x 23
        env         : Params()

    출력:
        C_t         : 6 x (N+2) 암호문 [v + b_v - b_prime | Av | b_prime]  # b_prime := Sigma_pinv @ b_tilde
        b_v         : 식 (21) 연산을 위해 밖으로 뺌
    """
    n_v = 6
    N = env.N

    # 데이터 타입 맞추기
    v = np.asarray(v, dtype=object).reshape(n_v, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)
    b_xi = np.asarray(b_xi, dtype=object).reshape(23, 1)
    Sigma = np.asarray(Sigma, dtype=object).reshape(1, 6)
    Sigma_pinv = np.asarray(Sigma_pinv, dtype=object).reshape(6, 1)
    Psi = np.asarray(Psi, dtype=object).reshape(1, 23)

    # 1) Av, e
    Av = rand_mod_q((n_v, N), env.q)   # 6 x N, entries in [0, q-1]
    e = np.random.randint(-env.r, env.r + 1, size=(n_v, 1)).astype(object)  # 6x1  에러      
    # e = np.zeros((n_v, 1), dtype=object)    # 테스트용 에러 x

    # 2) b_v = Av sk + e (6x1)
    b_v = Mod(Av @ sk + e, env.q) 

    # 3) b_prime := Sigma_pinv @ (Sigma @ b_v + Psi @ b_xi) (6x1)
    b_prime = Mod(Sigma_pinv @(Sigma @ b_v + Psi @ b_xi), env.q)

    # 4) 첫번째 열
    c0 = Mod(env.L * v + b_v - b_prime, env.q)

    # 5) 최종 암호문: [c0 | Av | b_prime] (6 x (N+2))
    C_t = np.hstack([c0, Av, b_prime])

    return C_t, b_v


def Dec(ciphertext, sk, env):
    
    dec_sk = np.vstack([
        np.ones((1, 1), dtype=object),
        -sk,
        np.ones((1, 1), dtype=object)
    ])
    
    m_bar = Mod(ciphertext@ dec_sk, env.q)  # (h x 1)

    return m_bar


>>>>>>> d1406c7f94c7158a13a21976b5b41889ee7551bd
