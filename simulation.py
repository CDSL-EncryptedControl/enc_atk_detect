import numpy as np
from scipy.io import loadmat
from enc_func import *
import matplotlib.pyplot as plt   
import time  
from offline_task import *

# 파라미터 생성
Ts = 0.1  # 샘플링타임 1초
env = Params()  # 환경 변수 
sk = Seret_key(env) # 비밀키

'''
스케일 정하기
'''

# 양자화 파라미터 # 10^10
r_quant = 10000000000
s_quant = 10000000000

# 1) 오프라인 행렬 계산 및 저장
offline = compute_offline_mats(env, s_quant)
save_offline_mats(offline)
print("오프라인 행렬 저장\n")



# 오프라인 행렬들 준비  #변경사항: Phi_pinv \in R 을 오프라인 행렬로 변경
# offline = np.load("offline_mats.npz", allow_pickle=True)
offline = np.load("matrices.npz", allow_pickle=True)

F_bar = offline["F_bar"]          # 24x24
G_bar = offline["G_bar"]          # 24x6
H_bar = offline["H_bar"]          # 60x24
Phi_pinv_bar = offline["Phi_pinv_bar"]
T1_all = offline["T1_all"]        # 60x23x24
T2_all = offline["T2_all"]        # 60x1x24z
V1_all = offline["V1_all"]        # 60x24x23
V2_all = offline["V2_all"]        # 60x24x1
S_xi_all = offline["S_xi_all"]    # 60x23x23
S_v_all  = offline["S_v_all"]     # 60x23x6
Psi_all  = offline["Psi_all"]     # 60x1x23
Sigma_all = offline["Sigma_all"]  # 60x1x6
Sigma_pinv_all = offline["Sigma_pinv_all"]  # 60x6x1 (dtype=object)


# Ts = 0.1, 이산화된 플랜트 A B C 행렬
A = np.array([
    [ 0.993182, 0.009942, 0.006811, 0.000023, 0.000008, 0.000000 ],
    [ -1.358992, 0.986222, 1.355884, 0.006795, 0.003109, 0.000008 ],
    [ 0.006811, 0.000023, 0.986379, 0.009920, 0.006811, 0.000023 ],
    [ 1.355884, 0.006795, -2.711767, 0.979435, 1.355884, 0.006795 ],
    [ 0.000008, 0.000000, 0.006811, 0.000023, 0.993182, 0.009942 ],
    [ 0.003109, 0.000008, 1.355884, 0.006795, -1.358992, 0.986222 ]
], dtype=np.float64)


B = np.array([ 0.004983, 0.994236, 0.000006, 0.002271, 0.000000, 0.000002 ], dtype=np.float64)  # shape (6,)

C = np.array([
    [ 1, 0, 0, 0, 0, 0 ],
    [ 0, 0, 1, 0, 0, 0 ],
    [ 0, 0, 0, 0, 1, 0 ],
    [ 1, 0, -1, 0, 0, 0 ],
    [ 0, 0, 1, 0, -1, 0 ]
], dtype=np.float64)

K = np.array([ -7.357054, -0.695315, 8.115979, -0.267057, -1.768888, -0.081939 ], dtype=np.float64)  # shape (6,)


## 시뮬레이션 세팅

iter = 300
n_channels = 60 # j_index
execution_times = []  # 실행 시간을 저장할 리스트


'''
# 초기값
'''
xp0 = np.array([[1], [1], [1], [1], [1], [1]])          # 플랜트 초기값
# z_hat0 = np.full((24, 1), 0.1, dtype=float)                         # z_hat 초기값
z_hat0 = np.full((24, 1), 0.0, dtype=float)   


# 플롯 저장용
attack_arr = np.zeros(iter)   # 각 k에서 주입한 공격 신호 저장
attack_start = iter // 2      # 절반 시점부터 공격 시작
xp = [xp0]
u = []
y = []
x_hat_list = []

# residue: 각 채널 r_j(k)의 실수 복원값을 저장: (iter, 60)
residue = np.zeros((iter, n_channels))

# 옵저버 초기 값
z_hat_bar = np.round(z_hat0 * r_quant * s_quant).astype(int)

# j 인덱스 state
Z_hat_list = []   # 길이 60, 각 원소는 24 x (N+2)
b_xi_list = []    # 길이 60, 각 원소는 23 x 1

for j in range(n_channels):
    T1_j = T1_all[j]           # 23x24
    T2_j = T2_all[j]           # 1x24
    V2_j = V2_all[j]           # 24x1
    Z_hat_j, b_xi_j = Enc_state(z_hat_bar, sk, env, T1_j, T2_j, V2_j)
    Z_hat_list.append(Z_hat_j)  # 초기값
    b_xi_list.append(b_xi_j)    # 초기값


Z_hat_ref0 = Z_hat_list[0]                        # j=0 채널 하나 사용
X_hat_cipher0 = Mod(Phi_pinv_bar @ Z_hat_ref0, env.q)
x_hat_int0 = Dec(X_hat_cipher0, sk, env)
x_hat0 = x_hat_int0 / (r_quant * s_quant * s_quant * env.L)

x_hat_list = [x_hat0]  # 이제 x_hat_list[0] = x_hat(0)


#  Simulation loop

for k in range(iter):
    # ==== 시간 측정 시작 ====
    t_start = time.perf_counter()

    # 1) 플랜트 출력 y_k = C x_k
    y_k = C @ xp[-1]          # 5x1, float64
    y.append(y_k)

    # 2) 피드백 제어 입력 u_k = K x_k
    u_k = float(K @ xp[-1])   # 스칼라
    u.append(u_k)

    # 3) v = [u; y] (6x1, float)
    v = np.vstack([
        np.array([[u_k]]),  # 1x1
        y_k                 # 5x1
    ])

    # 공격신호 
    '''
    추후 공격 신호에 변화...?
    '''
    if k >= attack_start:
        crit = (k-attack_start)/iter 
        if 0.1> crit :
            attack  = 1
        elif 0.35>crit >= 0.3:
            attack = -1
        elif 0.4>crit >= 0.35:
            attack = 0
        else:
            attack = 0
    else:
        attack = 0.0
    
    attack_vector = np.zeros((6, 1), dtype=float)
    
    # 3번 센서
    attack_vector[3, 0] = attack  
    attack_arr[k] = attack        
    
    # # 공격 신호 넣기 # 실수 위에서 넣는 것으로 변경
    v += attack_vector   # 3번째 센서에만 공격

    # 인덱스 별 r_j, Z_hat_j, 동적 마스킹 파트 업데이트
    for j in range(n_channels):

        # 오프라인 행렬
        H_j = H_bar[j, :].reshape(1, 24)              # 1x24
        S_xi_j = S_xi_all[j]                          # 23x23
        S_v_j  = S_v_all[j]                           # 23x6
        Psi_j  = Psi_all[j]                           # 1x23
        Sigma_j = Sigma_all[j]                        # 1x6
        Sigma_pinv_j = Sigma_pinv_all[j]              # 6x1

        Z_hat_j = Z_hat_list[j]                       # 24 x (N+2)
        b_xi_j  = b_xi_list[j]                        # 23 x 1

        # residue R_j = H_j @ Z_hat_j (1 x (N+2))
        R_j = Mod(H_j @ Z_hat_j, env.q)           # 1 x (N+2)

        # 첫 번째 항만 잔차로 사용 (스칼라, mod q 정수로 정리)
        bbr_j = Mod(R_j[0, 0], env.q)              # Python int, mod q

        # L^-1 곱셈
        r_bar_j = Mod(bbr_j * env.invL, env.q)          
        r_j = float(r_bar_j) / (r_quant * s_quant * s_quant)

        # 실수 residue
        residue[k, j] = r_j
        
        # 출력 암호화 # 기존과 다르게 큰 수를 다루기 위해서 np.vectorize(lambda x: int(round(x)), otypes=[object])(v_scaled) 이런 꼴의 코드가 추가...
        v_scaled = v * r_quant
        v_bar = np.vectorize(lambda x: int(round(x)), otypes=[object])(v_scaled)
        V_j, b_v_j = Enc_t(v_bar, sk, b_xi_j, Sigma_pinv_j, Sigma_j, Psi_j, env)


        # 암호화된 옵저버 상태 업데이트: Z_hat_{k+1}^j
        Z_hat_j_next = Mod(F_bar @ Z_hat_j + G_bar @ V_j, env.q)

        '''
        수정된 암호스킴의 업데이트
        '''
        # (e) 마스킹 상태 업데이트: b_xi_{k+1}^j = S_xi_j b_xi_j + S_v_j b_v_j
        b_xi_j_next = S_xi_j @ b_xi_j + S_v_j @ b_v_j
        b_xi_j_next = Mod(b_xi_j_next, env.q)

        # 리스트 저장
        Z_hat_list[j] = Z_hat_j_next
        b_xi_list[j]  = b_xi_j_next

    # 5-1) z(t) \in 24x1 로 상태 추정
    # Phi_pinv_bar  : 6 x 24  -> X_hat_cipher : 6 x (N+2)
    Z_hat_ref = Z_hat_list[0]  # 하나 가져오기 (Z_hat은 마스킹 파트만 다르고 메세지는 같음)
    X_hat_cipher = Mod(Phi_pinv_bar @ Z_hat_ref, env.q)  # 6 x (N+2)

    # 복호화
    x_hat_int = Dec(X_hat_cipher, sk, env)
    x_hat_list.append(x_hat_int/ (r_quant * s_quant * s_quant * env.L))

    # 6) 플랜트 상태 업데이트: x_{k+1} = A x_k + B u_k
    xp_next = A @ xp[-1] + B.reshape(-1, 1) * u_k   # B: (6,) -> (6,1)
    xp.append(xp_next)

    # ==== 시간 측정 끝 ====
    t_end = time.perf_counter()
    execution_times.append(t_end - t_start)



'''
루프 실행 시간 통계, N = 2000일 때, 루프 당 평균 약 3초 / j 하나 당 루프 시간 0.05초
'''
execution_times = np.array(execution_times)
print(f"Iteration time min  : {execution_times.min():.6f} s")
print(f"Iteration time max  : {execution_times.max():.6f} s")
print(f"Iteration time mean : {execution_times.mean():.6f} s")



'''
결과 플롯
'''

# 시간축
t_x = np.arange(len(xp))        # 상태는 k=0..iter → 길이 iter+1
t_u = np.arange(len(u))         # u, y, residue는 k=0..iter-1 → 길이 iter



# 입력 u: (iter,)
u_arr = np.array(u)             # (iter,)

# 출력 y: 5x(iter)
y_arr = np.hstack(y)            # 5 x iter

xp_arr = np.hstack(xp)              # 6 x (iter+1), x(0..iter)
x_hat_arr = np.hstack(x_hat_list)   # 6 x (iter+1), x_hat(0..iter)

t_k = np.arange(iter + 1)           # 0..iter

fig, axes = plt.subplots(6, 1, figsize=(8, 10), sharex=True)

for i_state in range(6):
    ax = axes[i_state]
    ax.plot(t_k, xp_arr[i_state, :],        label=f"x[{i_state}] (true)")
    ax.plot(t_k, x_hat_arr[i_state, :], '--', label=f"x_hat[{i_state}] (est)")
    ax.set_ylabel(f"x{i_state+1}")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    if i_state == 0:
        ax.set_title("Plant States xp vs Estimated States x_hat (including k=0)")

axes[-1].set_xlabel("time")

plt.tight_layout()

# 공격 신호 (attack)
plt.figure(figsize=(8, 4))
plt.plot(t_u, attack_arr)
plt.xlabel("time")
plt.ylabel("attack")
plt.title("Injected Attack on sensor 3")
plt.grid(True)

# 각 시간 k마다 전체 60 채널에 대한 ∞-norm 계산
r_inf = np.max(np.abs(residue), axis=1)  # shape: (iter,)

'''
임계 값 설정
'''
thr = 1.0   # threshold 값 설정

# 플롯
plt.figure(figsize=(8, 4))
plt.plot(t_u, r_inf, label="||r||∞")
plt.axhline(thr, color='k', linestyle='--', label="threshold")  # 검은 점선 임계값
plt.title("Residual ∞-norm")
plt.xlabel("time")
plt.ylabel("||r||∞")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 안 가리게 여백 조정
plt.show()
