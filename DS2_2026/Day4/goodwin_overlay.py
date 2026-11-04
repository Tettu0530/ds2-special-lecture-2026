import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def goodwin_model(t, init, *param):
    v, u = init[0], init[1]

    [alpha, beta, gamma, rho, sigma] = param

    dvdt = ((1.0 / sigma - (alpha + beta)) - u / sigma) * v
    dudt = (-(alpha + gamma) + rho * v) * u

    return [dvdt, dudt]


init = [0.06, 0.92]
t_span = [0, 500]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

alpha = 0.02   # 技術進歩率
beta = 0.01    # 労働人口成長率
gamma = 0.01   # 賃金上昇の自然減衰項
rho = 0.5      # フィリップス曲線の感応度
sigma = 2.0    # 投資係数

param = [alpha, beta, gamma, rho, sigma]

# Trajectory (tight tolerance: the default drifts outward by numerical error)
solution = solve_ivp(goodwin_model, t_span, init, t_eval=t_eval, method='RK45', args=param, rtol=1e-10, atol=1e-12)

v, u = solution.y

# Nullclines
v_null = (alpha + gamma) / rho        # du/dt = 0 : v = (α + γ) / ρ
u_null = 1 - sigma * (alpha + beta)   # dv/dt = 0 : u = 1 - σ(α + β)

# Vector field
N = 15
vg = np.linspace(0.05, 0.07, N)
ug = np.linspace(0.91, 0.97, N)

[V, U] = np.meshgrid(vg, ug)

dVdt = ((1.0 / sigma - (alpha + beta)) - U / sigma) * V
dUdt = (-(alpha + gamma) + rho * V) * U

plt.figure(figsize=(11, 4.5))

for i, method in enumerate(['quiver', 'streamplot']):
    plt.subplot(1, 2, i + 1)
    plt.title(method)

    if method == 'quiver':
        plt.quiver(V, U, dVdt, dUdt)
    else:
        plt.streamplot(V, U, dVdt, dUdt)

    plt.axvline(v_null, color='b')   # du/dt = 0
    plt.axhline(u_null, color='r')   # dv/dt = 0
    plt.plot(v, u, 'k-')
    plt.plot(v_null, u_null, 'ko')   # fixed point

    plt.xlim(vg[0], vg[-1])
    plt.ylim(ug[0], ug[-1])
    plt.xlabel('v (employment rate)')
    plt.ylabel('u (wage share)')

plt.tight_layout()
plt.show()
