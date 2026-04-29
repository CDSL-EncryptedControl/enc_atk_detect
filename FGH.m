
clear; clc; close all; format short;

%% 플랜트
J1=0.01; J2=0.01; J3=0.01; k1=1.37; k2=1.37;
b1=0.007; b2=0.007; b3=0.007;


A0=[ 0 1 0 0 0 0;
   -k1/J1 -b1/J1 k1/J1 0 0 0;
    0 0 0 1 0 0;
    k1/J2 0 -(k1+k2)/J2 -b2/J2 k2/J2 0;
    0 0 0 0 0 1;
    0 0 k2/J3 0 -k2/J3 -b3/J3 ];

B0=[0;1/J1;0;0;0;0];

C=[ 1 0 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 1 0;
    1 0 -1 0 0 0;
    0 0 1 0 -1 0 ];

% 센서 5개
C1 = C(1,:);
C2 = C(2,:);
C3 = C(3,:);
C4 = C(4,:);
C5 = C(5,:);

D=zeros(5,1);

Ts = 0.1;
sysd=c2d(ss(A0,B0,C,D),Ts);

A=sysd.A; B=sysd.B; C=sysd.C;

% feedback gain 

Q = eye(6);
R = eye(1);
[~, K, ~] = idare(A,B,Q,R,[],[]);
K = -K;


% 사이즈
n=6; p=5; m=1;


% 각 센서의 관측성 확인
l1 = rank(obsv(A, C1));
l2 = rank(obsv(A, C2));
l3 = rank(obsv(A, C3));
l4 = rank(obsv(A, C4));
l5 = rank(obsv(A, C5));


%% 칼만 분해

% 센서별로 시스템 나누기

Ci_list = {C1; C2; C3; C4; C5};

Phi_obs = cell(5,1);     % 가관측 기저 Φ_i^(obs) (li x n)
Phi_p   = cell(5,1);     % Φ_i^(obs)의 우역행렬 (n x li)
F_raw   = cell(5,1);     % 축소된 F_i (observable 서브시스템)
G_raw   = cell(5,1);
H_raw   = cell(5,1);
li      = zeros(5,1);    % 각 센서별 관측가능 차원(rank)


for i = 1:5
    Ci = Ci_list{i};

    % full observability matrix
    Oi = obsv(A, Ci);
    li(i) = rank(Oi);

    % 앞 li(i)개의 row만 사용 (observability index까지)
    Oki = Oi(1:li(i), :);          % li(i) x n

    % row-space 직교기저 만들기
    [Q,~] = qr(Oki.', 0);          % Q: n x li(i), columns orthonormal

    % observable subspace basis (행기저)
    Phi_i_obs = Q.';               % li(i) x n
    Phi_i_p   = pinv(Phi_i_obs);   % n x li(i), 우역행렬 (Phi_i_obs*Phi_i_p ≈ I_li)

    % observable subsystem (z_i = Phi_i_obs * x)
    F_raw{i} = Phi_i_obs * A * Phi_i_p;   % li x li
    G_raw{i} = Phi_i_obs * B;            % li x m
    H_raw{i} = Ci * Phi_i_p;             % 1  x li

    Phi_obs{i} = Phi_i_obs;
    Phi_p{i}   = Phi_i_p;
end


%% 캐노니컬 폼 
Phi_final = cell(5,1);   % x -> z_i,can 매핑 (Phi_i = T_i * Phi_i^(obs))
F_can     = cell(5,1);   % 관측형 canonical F_i
G_can     = cell(5,1);
H_can     = cell(5,1);
T_obs     = cell(5,1);   % z_i -> z_i,can 좌표변환 행렬 T_i

for i = 1:5
    Fi = F_raw{i};
    Gi = G_raw{i};
    Hi = H_raw{i};

    li_i = size(Fi,1);

    % Fi,Hi에 대한 관측성 행렬
    Oi_small = obsv(Fi, Hi);      % li_i x li_i (Fi,Hi가 observable이면 full-rank)

    % basis = inv(Oi_small) * e_last
    e_last = zeros(li_i,1); 
    e_last(end) = 1;

    basis = Oi_small \ e_last;    % = inv(Oi_small)*e_last

    % inv(T_i) = [basis, Fi*basis, Fi^2*basis, ..., Fi^(li-1)*basis]
    inv_T = zeros(li_i);
    v = basis;
    for k = 1:li_i
        inv_T(:,k) = v;
        v = Fi * v;
    end

    T_i = inv(inv_T);             % 좌표변환 행렬 T_i

    % canonical representation (z_i,can = T_i * z_i)
    F_can{i} = T_i * Fi / T_i;
    G_can{i} = T_i * Gi;
    H_can{i} = Hi / T_i;

    % 최종 Φ_i: x -> z_i,can
    Phi_final{i} = T_i * Phi_obs{i};   % li_i x n
    T_obs{i}     = T_i;
end


%% 정리

Phi1 = Phi_final{1};   F1 = F_can{1};   B1 = G_can{1};   H1 = H_can{1};
Phi2 = Phi_final{2};   F2 = F_can{2};   B2 = G_can{2};   H2 = H_can{2};
Phi3 = Phi_final{3};   F3 = F_can{3};   B3 = G_can{3};   H3 = H_can{3};
Phi4 = Phi_final{4};   F4 = F_can{4};   B4 = G_can{4};   H4 = H_can{4};
Phi5 = Phi_final{5};   F5 = F_can{5};   B5 = G_can{5};   H5 = H_can{5};


%% F1과 F3는 같고, F4와 F5도 같음 (observable subspace가 동일, 2-redundancy)

% F1: l1 x l1 행렬이라고 할 때
L1 = F1(:, end);   % 제일 오른쪽 열 (l1 x 1 벡터)
L2 = F2(:, end);
L3 = F3(:, end);
L4 = F4(:, end);
L5 = F5(:, end);

% Partial systems (8a)
F1_bar = F1 - L1*H1;    G1_bar = [B1  L1];
F2_bar = F2 - L2*H2;    G2_bar = [B2  L2];
F3_bar = F3 - L3*H3;    G3_bar = [B3  L3];
F4_bar = F4 - L4*H4;    G4_bar = [B4  L4];
F5_bar = F5 - L5*H5;    G5_bar = [B5  L5];



G_ = [[B1; B2; B3; B4; B5], blkdiag(L1, L2, L3, L4, L5)];

F_bar = blkdiag(F1_bar, F2_bar, F3_bar, F4_bar, F5_bar);


Phi = [
    Phi1;
    Phi2;
    Phi3;
    Phi4;
    Phi5
];

L_bar = blkdiag(L1, L2, L3, L4, L5);

Phi_pinv = pinv(Phi);




%% 센서 조합별 Phi

% Phi 리스트 및 각 센서 상태크기 li 사용
Phi_list = {Phi1, Phi2, Phi3, Phi4, Phi5};
li_vec   = [size(Phi1,1), size(Phi2,1), size(Phi3,1), size(Phi4,1), size(Phi5,1)];
% 보통 li_vec = [6 4 6 4 4] 가 나올 것

% 센서 인덱스 조합 (5개 중 3개씩 → 10개 조합)
idx_comb = nchoosek(1:5, 3);   % 10 x 3

Phi_bar_set  = cell(10,1);    % 각 k에 대한 [Phi_i1; Phi_i2; Phi_i3]
Phi_bar_pinv = cell(10,1);    % 각 k에 대한 pinv(Phi_bar_k)

for k = 1:size(idx_comb,1)
    idx = idx_comb(k,:);   % 예: [1 2 3]

    % Phi_bar_k = [Phi_i1; Phi_i2; Phi_i3]
    Phi_bar_k = [
        Phi_list{idx(1)};
        Phi_list{idx(2)};
        Phi_list{idx(3)}
    ];

    Phi_bar_set{k}  = Phi_bar_k;
    Phi_bar_pinv{k} = pinv(Phi_bar_k);

    % [r,c]   = size(Phi_bar_k);
    % [rp,cp] = size(Phi_bar_pinv{k});
    % 
    % fprintf(['k = %d, sensors = [%d %d %d], ', ...
    %          'Phi_bar size = [%d x %d], Phi_bar_pinv size = [%d x %d]\n'], ...
    %         k, idx(1), idx(2), idx(3), r, c, rp, cp);
end


%% ========= 6) 각 조합에 대해 z -> x_hat(k) 매핑 H_k 만들기 =========

% 전역 z 인덱스 (cum_li로 경계 계산)
cum_li = [0, cumsum(li_vec)];   % 예: [0 6 10 16 20 24]

H_k = cell(10,1);   % H_k{k} : 6 x 24,  z -> x_hat^(k)

for k = 1:size(idx_comb,1)
    idx_sensors = idx_comb(k,:);      % 예: [1 2 3]
    Pk = Phi_bar_pinv{k};             % 6 x (li_i1 + li_i2 + li_i3)

    Hk = zeros(6, 24);

    % Pk의 column은 [z_i1; z_i2; z_i3] 순서
    col_start = 0;
    for jj = 1:3
        s = idx_sensors(jj);          % 센서 번호 (1~5)
        lj = li_vec(s);               % 해당 센서 상태 크기

        cols_local  = col_start + (1:lj);   % Pk 내에서의 column 구간
        cols_global = (cum_li(s)+1) : cum_li(s+1);  % z_bar에서의 column 구간

        Hk(:, cols_global) = Pk(:, cols_local);

        col_start = col_start + lj;
    end

    H_k{k} = Hk;
end


%% ========= 7) H (60x24) 만들기 =========
% r = [r1; r2; ...; r10] = H * z

H = zeros(6*10, 24);

for k = 1:10
    % 각 조합에 대한 residual:
    % r_k = (H_k{k} - Phi_pinv) * z
    H_block = H_k{k} - Phi_pinv;     % 6 x 24

    rows = (6*(k-1)+1) : (6*k);      % 해당 r_k 위치
    H(rows, :) = H_block;
end



%% mat. 포맷으로 저장/ F_bar는 정수 , G_, H 는 실수
% save('FGH_data.mat','F_bar','G_','H');

save('FGHPhi_pinv.mat','F_bar','G_','H', 'Phi_pinv');



%% Simulation over Real #
iter = 100;

% plant initial state
xp0 = 1*ones(n,1);
z_hat_0 = zeros(24,1);
z_hat = z_hat_0;
xp = xp0;
u = [];
y = [];
r = [];


for i = 1:iter
    % plant & observer output

    if i > 50 && i<60
        attack = [0; 0; 1; 0; 0];
    else 
        attack = 0;
    end

    y = [y, C*xp(:,i)+ attack];
    r = [r, H*z_hat(:,i)];
    
    % feedback input
    u = K*xp;

    % state update
    xp = [xp, A*xp(:,i) + B*u(:,i)];
    z_hat = [z_hat, F_bar*z_hat(:,i) + G_*[u(:,i); y(:,i)]];
end


%% Plotting
t  = Ts*(0:iter-1);    % y, r, z_hat(:,1:iter)에 대응하는 시간축
t_x = Ts*(0:iter);     % x, x_hat (0..iter)에 대응하는 시간축

%% 1) 출력 y 플롯
figure;
plot(t, y);
grid on;
xlabel('time [s]');
ylabel('y');
title('Plant outputs y');
legend;   % 자동 legend

%% 2) x (plant state) vs x_hat (estimated state from z_hat)
% z_hat: 24 x (iter+1), Phi_pinv: 6 x 24 이라고 가정
x_hat = Phi_pinv * z_hat;   % 6 x (iter+1)

figure;
for i = 1:6
    subplot(6,1,i);
    plot(t_x, xp(i,1:iter+1), 'b', 'LineWidth', 1.2); hold on;
    plot(t_x, x_hat(i,1:iter+1), 'r--', 'LineWidth', 1.2);
    grid on;
    ylabel(sprintf('x_%d', i));
    if i == 1
        title('Plant states x_i vs estimated states \hat{x}_i');
    end
    if i == 6
        xlabel('time [s]');
    end
end

%% 3) residual r의 ∞-노름 플롯
% r: (n_r x iter) 라고 가정 (각 column이 k 시점 residual 벡터)
r_inf = vecnorm(r, Inf, 1);   % 각 k에서 max |r_i(k)|

figure;
% plot(t(7:end), r_inf(7:end));
plot(t, r_inf);
grid on;
xlabel('time [s]');
ylabel('||r||_\infty');
title('Residual norm ||r(k)||_\infty');
 
%% 3-1) Finding parameters and bound
s_ = 1e5;
r_ = 1e5;
z_tilde = norm(Phi * xp0 - z_hat_0,Inf);
l_max = max(li);
inf_norms = cellfun(@(M) norm(M, Inf), Phi_bar_pinv); 
kappa = max(norm(Phi_pinv,Inf),max(inf_norms))
G_norm = norm(G_,inf);

% v = [u;y]
vmax =  norm([K;C],Inf)* norm(xp0,Inf);
M = max(1,norm(H,Inf)) * ( norm(z_hat_0,Inf) + l_max*norm(G_,Inf)* vmax );
eini = 1/(2*r_*s_);
ez = (m+p)/(2*s_) * vmax + norm(G_,Inf)/(2*r_) + (m+p)/(4*r_*s_);
epsilonz = eini + l_max*ez;
epsilon = 2*(kappa*epsilonz + cum_li(end)/(2*s_)*(M+epsilonz))

transient = kappa * z_tilde

L = 2*(kappa*s_ + 12) * (1+6*(s_ * G_norm + 4))

%% 파이썬 포맷 프린트


% print_numpy('Phi_pinv_bar', Phi_pinv_bar, 0);

print_numpy('A', A, 6);
print_numpy('B', B, 6);
print_numpy('C', C, 0);

print_numpy('K', K, 6);



% print_numpy('F_bar', F_bar, 0);
% print_numpy('G_bar', G_bar, 0);
% print_numpy('H_bar', H_bar, 0);
% 
% 

function print_numpy(name, M, prec)
    if nargin < 3, prec = 4; end
    fmt = ['%0.' num2str(prec) 'f'];

    if isvector(M)
        % 1D 벡터는 한 줄로 출력 (예: H)
        fprintf('%s = np.array([ ', name);
        for k = 1:numel(M)
            fprintf(fmt, M(k));
            if k < numel(M), fprintf(', '); end
        end
        fprintf(' ], dtype=np.float64)  # shape (%d,)\n\n', numel(M));
    else
        % 2D 행렬은 행 단위로 출력 (예: F, G)
        fprintf('%s = np.array([\n', name);
        for i = 1:size(M,1)
            fprintf('    [ ');
            for j = 1:size(M,2)
                fprintf(fmt, M(i,j));
                if j < size(M,2), fprintf(', '); end
            end
            if i < size(M,1)
                fprintf(' ],\n');
            else
                fprintf(' ]\n');
            end
        end
        fprintf('], dtype=np.float64)\n\n');
    end
end