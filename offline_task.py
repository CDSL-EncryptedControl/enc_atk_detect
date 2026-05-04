import numpy as np
from scipy.io import loadmat
from enc_func import *


# 좌표변환 행렬 구하기
def build_TV(H1, q):
    """
    입력:  
        H1 : 1×24 numpy object vector  
        q  : modulus

    출력:  
        T1, T2, T, V, V1, V2

    설계:
      - pivot column = 20번째 열 (0-based index 19)
      - T1 : 23x24, 20열만 제외한 나머지 23개 열에 대한 선택행렬
      - T2 = H1
      - T  = [T1; T2]
      - V  = T^{-1} (mod q) 를 closed-form으로 직접 구성
      - V  = [V1 V2] with V1: 24x23, V2: 24x1
    """

    H1 = np.asarray(H1, dtype=object).reshape(24,)
    n = 24
    pivot = 19  # 0-based index, = 20번째 열

    # -------------------------
    # 1) T1 만들기 : "pivot 열만 제외한 표준 기저"
    # -------------------------
    T1 = np.zeros((n - 1, n), dtype=object)  # 23x24
    cols = list(range(n))
    cols.remove(pivot)   # [0,1,...,18,20,21,22,23]
    for row, col in enumerate(cols):
        T1[row, col] = 1

    # 2) T2 = H1
    T2 = H1.reshape(1, n)

    # 3) T = [T1; T2]
    T = np.vstack([T1, T2])

    # -------------------------
    # 4) V = T^{-1} (mod q) 
    # -------------------------
    V = np.zeros((n, n), dtype=object)

    # pivot 값 및 역원
    h_p = int(H1[pivot]) % q
    if h_p == 0:
        raise ValueError("H1[19] (20번째 원소) ≡ 0 (mod q) 입니다. pivot으로 사용할 수 없습니다.")
    inv_h_p = pow(h_p, -1, q)

    for j in range(n - 1):  # 0..22
        orig_idx = j if j < pivot else j + 1  # 0-based
        h_orig = int(H1[orig_idx]) % q

        for i in range(n):
            if i == pivot:
                # pivot 행: -h_orig / h_p (mod q)
                V[i, j] = (-h_orig * inv_h_p) % q
            else:
                # identity mapping
                if i == orig_idx:
                    V[i, j] = 1
                else:
                    V[i, j] = 0

    # 마지막 열 (23, 즉 24번째 열):
    #   모든 행 0, 단 pivot 행에만 1/h_p
    for i in range(n):
        V[i, n - 1] = 0
    V[pivot, n - 1] = inv_h_p % q

    # -------------------------
    # 5) V1, V2 분리
    # -------------------------
    V1 = V[:, :n - 1].copy()          # 24x23
    V2 = V[:, n - 1].reshape(n, 1)    # 24x1

    return T1, T2, T, V, V1, V2


# 큰 수를 다루기 위해서 추가 됨 (gpt가 해줬습니다...)
def float_to_object_int(arr_float: np.ndarray) -> np.ndarray:
    """
    float 배열을 element-wise로 Python int로 변환해서
    dtype=object 배열로 리턴.
    (np.int64 한 번 거치지 않아서 overflow warning 안 뜸)
    """
    arr_rounded = np.rint(arr_float)   # 먼저 반올림
    out = np.empty(arr_rounded.shape, dtype=object)

    it = np.nditer(arr_rounded, flags=['multi_index'])
    for x in it:
        v = float(x)
        if not np.isfinite(v):
            raise ValueError(f"Quantization produced non-finite value: {v}")
        out[it.multi_index] = int(v)   # Python big-int

    return out



def compute_offline_mats(env, s=100000, num_channels=60):
    # data = loadmat('FGH_data.mat')
    data = loadmat('FGHPhi_pinv')
    F_ = data['F_bar']
    G_ = data['G_']
    H_ = data['H']
    Phi_pinv_ = data['Phi_pinv']

    print(s)

    # 양자화 (여기까지는 float)
    F_bar_float = F_                  # F는 스케일 안 올리는 설계라면 이대로 두고,
    G_bar_float = s * G_
    H_bar_float = s * H_
    Phi_pinv_float = s * Phi_pinv_

    # 🔹 float -> Python int (object)로 직접 변환
    F_bar = float_to_object_int(F_bar_float)         # 24x24
    G_bar = float_to_object_int(G_bar_float)         # 24x6
    H_bar = float_to_object_int(H_bar_float)         # 60x24
    Phi_pinv_bar = float_to_object_int(Phi_pinv_float)  # 24x60

    # 채널 수만큼 저장할 배열 예시 (axis 0 = 채널 index)
    T1_all = []
    T2_all = []
    V1_all = []
    V2_all = []
    S_xi_all = []
    S_v_all = []
    Psi_all = []
    Sigma_all = []
    Sigma_pinv_all = []

    for j in range(num_channels):
        # j 인덱스마다, H의 행벡터에 대하여 좌표변환 행렬 T, V 찾기
        H1 = H_bar[j, :].copy()

        T1, T2, T, V, V1, V2 = build_TV(H1, env.q)

        # 식(19) 식
        S_1  = Mod(T1 @ F_bar @ V1, env.q)
        S_2  = Mod(T1 @ F_bar @ V2, env.q)
        S_3  = Mod(T1 @ G_bar,      env.q)

        Psi   = Mod(H1.reshape(1, -1) @ F_bar @ V1, env.q)
        Gamma = Mod(H1.reshape(1, -1) @ F_bar @ V2, env.q)   # 안 쓰더라도 일단 계산
        Sigma = Mod(H1.reshape(1, -1) @ G_bar,      env.q)

        sigma0 = int(Sigma[0, 0])
        if sigma0 == 0:
            raise ValueError(f"[채널 {j}] Sigma[0,0] ≡ 0 (mod q), pinv 구성 불가")
        inv_sigma0 = pow(sigma0, -1, env.q)
        Sigma_pinv = np.zeros((6, 1), dtype=object)
        Sigma_pinv[0, 0] = inv_sigma0

        S_xi = Mod(S_1 - S_3 @ Sigma_pinv @ Psi, env.q)
        S_v = Mod(S_3 @ (np.eye(6, dtype=object) - Sigma_pinv @ Sigma), env.q)

        T1_all.append(T1)
        T2_all.append(T2)
        V1_all.append(V1)
        V2_all.append(V2)
        S_xi_all.append(S_xi)
        S_v_all.append(S_v)
        Psi_all.append(Psi)
        Sigma_all.append(Sigma)
        Sigma_pinv_all.append(Sigma_pinv)

    # axis 0 에 채널 index가 오도록 stack
    T1_all = np.stack(T1_all, axis=0)
    T2_all = np.stack(T2_all, axis=0)
    V1_all = np.stack(V1_all, axis=0)
    V2_all = np.stack(V2_all, axis=0)
    S_xi_all = np.stack(S_xi_all, axis=0)
    S_v_all = np.stack(S_v_all, axis=0)
    Psi_all = np.stack(Psi_all, axis=0)
    Sigma_all = np.stack(Sigma_all, axis=0)
    # Sigma_pinv_all 은 리스트 형태 그대로 저장 (채널별 6x1)

    offline = {
        "F_bar": F_bar,
        "G_bar": G_bar,
        "H_bar": H_bar,
        "Phi_pinv_bar": Phi_pinv_bar,
        "T1_all": T1_all,
        "T2_all": T2_all,
        "V1_all": V1_all,
        "V2_all": V2_all,
        "S_xi_all": S_xi_all,
        "S_v_all": S_v_all,
        "Psi_all": Psi_all,
        "Sigma_all": Sigma_all,
        "Sigma_pinv_all": np.array(Sigma_pinv_all, dtype=object),
    }
    return offline


# ==========================
# npz 파일 읽고 / 쓰기
# ==========================

def save_offline_mats(offline, filename="matrices.npz"):
    np.savez(filename, **offline)


def _to_object_int(arr):
    """
    offline_mats 로드 시, 모든 오프라인 행렬을
    big-int 모듈러 연산이 가능한 dtype=object 정수 배열로 변환.
    """
    # 이미 object면 그대로
    if arr.dtype == object:
        return arr
    # 아니면 int -> object
    return arr.astype(int).astype(object)


def load_offline_mats(filename="offline_mats.npz"):
    data = np.load(filename, allow_pickle=True)

    offline = {}

    # 모듈러 연산에 직접 들어가는 애들 전부 object-int로 변환
    int_keys = [
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
    ]

    for k in data.files:
        if k in int_keys:
            offline[k] = _to_object_int(data[k])
        elif k == "Sigma_pinv_all":
            # (num_channels,) object 배열, 각 원소 6x1
            sigma_list = data[k]
            sigma_obj = []
            for sigma in sigma_list:
                sigma_obj.append(_to_object_int(sigma))
            offline[k] = np.array(sigma_obj, dtype=object)
        else:
            # 나머지는 그대로 보존
            offline[k] = data[k]

    return offline


if __name__ == "__main__":
    env = Params()

    # 1) 오프라인 행렬 계산 및 저장
    offline = compute_offline_mats(env)
    save_offline_mats(offline)
    print("오프라인 행렬 저장\n")

    # 디버그

    # 2) 바로 다시 읽어서 object-int로 잘 들어오는지 확인
    offline_loaded = load_offline_mats("matrices.npz")


    F_bar = offline_loaded["F_bar"]
    G_bar = offline_loaded["G_bar"]
    H_bar = offline_loaded["H_bar"]

    print("===== dtypes after load_offline_mats (should be object) =====")
    print("F_bar.dtype:", F_bar.dtype)
    print("G_bar.dtype:", G_bar.dtype)
    print("H_bar.dtype:", H_bar.dtype)

    # # === 채널 0의 오프라인 행렬들 출력 ===
    # ch = 0
    # print(f"\n===== Channel {ch} offline matrices (loaded) =====")

    # T1_all         = offline_loaded["T1_all"]
    # T2_all         = offline_loaded["T2_all"]
    # V1_all         = offline_loaded["V1_all"]
    # V2_all         = offline_loaded["V2_all"]
    # S_xi_all       = offline_loaded["S_xi_all"]
    # S_v_all        = offline_loaded["S_v_all"]
    # Psi_all        = offline_loaded["Psi_all"]
    # Sigma_all      = offline_loaded["Sigma_all"]
    # Sigma_pinv_all = offline_loaded["Sigma_pinv_all"]  # (num_channels,) object

    # print("\nT1 dtype:", T1_all.dtype)
    # print("T2 dtype:", T2_all.dtype)
    # print("V1 dtype:", V1_all.dtype)
    # print("V2 dtype:", V2_all.dtype)
    # print("S_xi dtype:", S_xi_all.dtype)
    # print("S_v dtype:", S_v_all.dtype)
    # print("Psi dtype:", Psi_all.dtype)
    # print("Sigma dtype:", Sigma_all.dtype)
    # print("Sigma_pinv_all dtype:", Sigma_pinv_all.dtype)

    # print("\nT1[0]:")
    # print(T1_all[ch])

    # print("\nT2[0]:")
    # print(T2_all[ch])

    # print("\nV1[0]:")
    # print(V1_all[ch])

    # print("\nV2[0]:")
    # print(V2_all[ch])

    # print("\nS_xi[0]:")
    # print(S_xi_all[ch])

    # print("\nS_v[0]:")
    # print(S_v_all[ch])

    # print("\nPsi[0]:")
    # print(Psi_all[ch])

    # print("\nSigma[0]:")
    # print(Sigma_all[ch])

    # print("\nSigma_pinv[0]:")
    # print(Sigma_pinv_all[ch])
