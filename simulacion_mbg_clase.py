"""
Movimiento Browniano Geométrico (MBG) con Euler-Maruyama usando diffeqpy
    dX_t = mu * X_t dt + sigma * X_t dB_t

Se grafican las trayectorias simuladas junto con:
    media   E[X_t]   = X0 * exp(mu t)                 (= solución determinista de dx/dt = mu x)
    mediana Med[X_t] = X0 * exp((mu - sigma^2/2) t)

Uso (los parámetros se cambian desde la terminal, sin editar el código):
    python3 simulacion_mbg_clase.py
    python3 simulacion_mbg_clase.py --mu 0.25 --sigma 1 --X0 1 --T 10 --M 20 --dt 0.01
    python3 simulacion_mbg_clase.py --help
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parámetros (modificables desde la terminal) ----------------
parser = argparse.ArgumentParser(description="Simulación del MBG con Euler-Maruyama (diffeqpy)")
parser.add_argument("--mu", type=float, default=0.25, help="deriva (default 0.25)")
parser.add_argument("--sigma", type=float, default=1.0, help="volatilidad (default 1.0)")
parser.add_argument("--X0", type=float, default=1.0, help="condición inicial (default 1.0)")
parser.add_argument("--T", type=float, default=10.0, help="tiempo final (default 10)")
parser.add_argument("--dt", type=float, default=0.01, help="paso de tiempo (default 0.01)")
parser.add_argument("--M", type=int, default=20, help="número de trayectorias (default 20)")
parser.add_argument("--seed", type=int, default=1, help="semilla inicial (default 1)")
parser.add_argument("--lineal", action="store_true", help="eje y lineal en vez de logarítmico")
parser.add_argument("--salida", default="resultado_mbg_clase.png", help="nombre de la figura")
args = parser.parse_args()

mu, sigma, X0, T, dt, M = args.mu, args.sigma, args.X0, args.T, args.dt, args.M

from diffeqpy import de   # se importa después de leer los argumentos (--help responde rápido)

# ---------------- Coeficientes de la EDE ----------------
def f(u, p, t):          # deriva (parte determinista)
    return p[0] * u

def g(u, p, t):          # difusión (parte estocástica)
    return p[1] * u

prob = de.SDEProblem(f, g, X0, (0.0, T), (mu, sigma))

# ---------------- Simulación con Euler-Maruyama ----------------
X = np.array([np.array(de.solve(prob, de.EM(), dt=dt, saveat=dt, seed=args.seed + k).u)
              for k in range(M)])                 # forma (M, n_pasos)
t = np.linspace(0.0, T, X.shape[1])

media = X0 * np.exp(mu * t)
mediana = X0 * np.exp((mu - sigma**2 / 2) * t)

print(f"mu = {mu}, sigma = {sigma}, X0 = {X0}, T = {T}, dt = {dt}, M = {M}")
print(f"mu - sigma^2/2 = {mu - sigma**2/2:.4f}  ->  "
      f"{'la mediana decrece' if mu - sigma**2/2 < 0 else 'la mediana crece'}")
print(f"t = T:  media teórica = {media[-1]:.4f}   mediana teórica = {mediana[-1]:.4f}")
print(f"t = T:  media muestral = {X[:, -1].mean():.4f}   mediana muestral = {np.median(X[:, -1]):.4f}")

# ---------------- Gráfica (estilo de la clase) ----------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
})
fondo = "#fbf7ef"
fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=fondo)
ax.set_facecolor("white")

for xi in X:
    ax.plot(t, xi, color="0.65", lw=0.8)
ax.plot(t, media, color="#a3262a", lw=2.5, label=r"media $e^{\mu t}$")
ax.plot(t, mediana, color="#2a63c6", lw=2.5, label=r"mediana $e^{(\mu-\sigma^2/2)t}$")

if not args.lineal:
    ax.set_yscale("log")
ax.set_xlim(0, T)
ax.set_xlabel(r"$t$")
ax.set_title(f"{M} trayectorias (Euler-Maruyama)", color="#1f3b6e", fontweight="bold",
             fontfamily="sans-serif", fontsize=16, pad=12)
ax.grid(True, which="major", color="0.9")
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.legend(loc="lower left", frameon=False)

plt.tight_layout()
plt.savefig(args.salida, dpi=150, facecolor=fondo)
print(f"Figura guardada en {args.salida}")
plt.show()
