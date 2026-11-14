"""
MBG: varias trayectorias estocásticas vs. modelo determinista
    Estocástico:  dX_t = mu * X_t dt + sigma * X_t dB_t      (Euler-Maruyama, diffeqpy)
    Determinista: dx/dt = mu * x   ->   x(t) = X0 * exp(mu t)  (Tsit5, diffeqpy)

Panel izquierdo (escala lineal): trayectorias, modelo determinista y promedio muestral.
Panel derecho (escala log):     las mismas curvas + mediana teórica, para ver la dispersión.

Resultado teórico que se compara con la simulación:
    ln X_T ~ N( ln X0 + (mu - sigma^2/2) T , sigma^2 T )
    P( X_T < x(T) ) = Phi( sigma * sqrt(T) / 2 )
    -> la mayoría de trayectorias termina POR DEBAJO del modelo determinista.

Uso:
    python3 simulacion_mbg_vs_determinista.py
    python3 simulacion_mbg_vs_determinista.py --M 10 --mu 0.25 --sigma 1 --X0 1 --T 10
    python3 simulacion_mbg_vs_determinista.py --help
"""
import argparse
from math import erf, sqrt
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="MBG: trayectorias estocásticas vs. modelo determinista")
parser.add_argument("--mu", type=float, default=0.25, help="deriva (default 0.25)")
parser.add_argument("--sigma", type=float, default=1.0, help="volatilidad (default 1.0)")
parser.add_argument("--X0", type=float, default=1.0, help="condición inicial (default 1.0)")
parser.add_argument("--T", type=float, default=10.0, help="tiempo final (default 10)")
parser.add_argument("--dt", type=float, default=0.01, help="paso de tiempo (default 0.01)")
parser.add_argument("--M", type=int, default=10, help="número de trayectorias (default 10)")
parser.add_argument("--seed", type=int, default=1, help="semilla inicial (default 1)")
parser.add_argument("--salida", default="resultado_mbg_vs_determinista.png", help="nombre de la figura")
args = parser.parse_args()
mu, sigma, X0, T, dt, M = args.mu, args.sigma, args.X0, args.T, args.dt, args.M

from diffeqpy import de

# ---------------- Coeficientes ----------------
def f(u, p, t):          # deriva (parte determinista)
    return p[0] * u

def g(u, p, t):          # difusión (parte con ruido)
    return p[1] * u

p = (mu, sigma)

# ---------------- Modelo determinista (sigma = 0) ----------------
sol_det = de.solve(de.ODEProblem(f, X0, (0.0, T), p), de.Tsit5(), saveat=dt)
x_det = np.array(sol_det.u)

# ---------------- M trayectorias con Euler-Maruyama ----------------
prob = de.SDEProblem(f, g, X0, (0.0, T), p)
X = np.array([np.array(de.solve(prob, de.EM(), dt=dt, saveat=dt, seed=args.seed + k).u)
              for k in range(M)])                 # forma (M, n_pasos)
t = np.linspace(0.0, T, X.shape[1])

promedio = X.mean(axis=0)
mediana_teo = X0 * np.exp((mu - sigma**2 / 2) * t)

# ---------------- Comparación numérica ----------------
abajo = int(np.sum(X[:, -1] < x_det[-1]))
p_teo = 0.5 * (1 + erf((sigma * sqrt(T) / 2) / sqrt(2)))   # Phi(sigma sqrt(T) / 2)

print(f"mu = {mu}, sigma = {sigma}, X0 = {X0}, T = {T}, dt = {dt}, M = {M}")
print(f"Modelo determinista en t = T:      x(T) = {x_det[-1]:.4f}")
print(f"Promedio de las {M} trayectorias:    {promedio[-1]:.4f}")
print(f"Mediana teórica en t = T:          {mediana_teo[-1]:.4f}")
print(f"Trayectorias que terminan por debajo del determinista: {abajo} de {M} "
      f"({100*abajo/M:.0f} %)   |   probabilidad teórica: {100*p_teo:.1f} %")
print("\nValor final de cada trayectoria:")
for k in range(M):
    rel = "debajo" if X[k, -1] < x_det[-1] else "ENCIMA"
    print(f"  trayectoria {k+1:2d}: X_T = {X[k, -1]:10.4f}   ({rel} del determinista)")

# ---------------- Gráficas ----------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12})
fondo = "#fbf7ef"
colores = plt.cm.tab10(np.linspace(0, 1, 10))
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=fondo)

for i, ax in enumerate(axs):
    for k in range(M):
        ax.plot(t, X[k], color=colores[k % 10], lw=1, alpha=0.8,
                label=f"Trayectoria {k+1}" if i == 0 else None)
    ax.plot(t, x_det, color="black", lw=3, label=r"Determinista $x(t)=X_0e^{\mu t}$")
    ax.plot(t, promedio, color="red", lw=2, ls="--", label=f"Promedio de las {M} trayectorias")
    ax.set_xlim(0, T)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$X_t$")
    ax.grid(True, color="0.9")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

axs[0].set_title("Escala lineal", fontsize=13)
axs[0].legend(fontsize=8.5, ncol=2, loc="upper left", frameon=False)

axs[1].plot(t, mediana_teo, color="#2a63c6", lw=2.5, label=r"Mediana $X_0e^{(\mu-\sigma^2/2)t}$")
axs[1].set_yscale("log")
axs[1].set_title("Escala logarítmica", fontsize=13)
axs[1].legend(fontsize=9, loc="lower left", frameon=False)

fig.suptitle(rf"MBG: {M} trayectorias vs. modelo determinista   "
             rf"($\mu$={mu}, $\sigma$={sigma}, $X_0$={X0})",
             color="#1f3b6e", fontweight="bold", fontfamily="sans-serif", fontsize=15)
plt.tight_layout()
plt.savefig(args.salida, dpi=150, facecolor=fondo)
print(f"\nFigura guardada en {args.salida}")
plt.show()
