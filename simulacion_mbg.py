"""
Movimiento Browniano Geométrico: Euler-Maruyama (diffeqpy) vs. solución determinista
    Estocástica:  dX_t = mu * X_t dt + sigma * X_t dB_t
    Determinista: dx/dt = mu * x   ->   x(t) = X0 * exp(mu t)
"""
import numpy as np
import matplotlib.pyplot as plt
from diffeqpy import de

# Parámetros
mu, sigma = 0.5, 0.3     # deriva y volatilidad
X0 = 0.0                 # condición inicial (con X0 = 0 ambas soluciones son ≡ 0)
T, dt = 1.0, 0.01        # horizonte y paso de tiempo
M = 20                   # número de trayectorias
p = [mu, sigma]

# Coeficientes de la ecuación
def f(u, p, t):          # deriva (parte determinista)
    return p[0] * u

def g(u, p, t):          # difusión (parte estocástica)
    return p[1] * u

# 1) Solución determinista (sigma = 0)
prob_ode = de.ODEProblem(f, X0, (0.0, T), p)
sol_ode = de.solve(prob_ode, de.Tsit5(), saveat=dt)
t = np.array(sol_ode.t)
x_det = np.array(sol_ode.u)
x_exacta = X0 * np.exp(mu * t)

# 2) Solución estocástica con Euler-Maruyama
prob_sde = de.SDEProblem(f, g, X0, (0.0, T), p)
X = np.array([np.array(de.solve(prob_sde, de.EM(), dt=dt, saveat=dt, seed=k + 1).u)
              for k in range(M)])               # forma (M, n_pasos)
promedio = X.mean(axis=0)

# 3) Comparación
dif = np.abs(X - x_det)
print(f"X0 = {X0}")
print(f"Máx |EDO numérica - analítica|        = {np.max(np.abs(x_det - x_exacta)):.3e}")
print(f"Máx |Euler-Maruyama - determinista|   = {dif.max():.3e}")
print(f"Promedio EM en t=T = {promedio[-1]:.4f}   |   determinista x(T) = {x_det[-1]:.4f}")

# 4) Gráficas
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k in range(M):
    ax[0].plot(t, X[k], lw=0.8, alpha=0.6)
ax[0].plot(t, promedio, "r--", lw=2, label=f"Promedio de las {M} trayectorias")
ax[0].plot(t, x_det, "k", lw=2.5, label=r"Determinista $x(t)=X_0e^{\mu t}$")
ax[0].set_title(rf"MBG: Euler-Maruyama vs. determinista ($\mu$={mu}, $\sigma$={sigma}, $X_0$={X0})")
ax[0].set_xlabel("Tiempo (t)"); ax[0].set_ylabel(r"$X_t$")
ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].plot(t, dif.max(axis=0), "b", lw=2, label=r"máx$_k\,|X^{(k)}_t - x(t)|$")
ax[1].set_title("Diferencia estocástica − determinista")
ax[1].set_xlabel("Tiempo (t)"); ax[1].set_ylabel("Diferencia absoluta")
ax[1].grid(alpha=0.3); ax[1].legend()

plt.tight_layout()
plt.savefig("resultado_mbg.png", dpi=150)
plt.show()
