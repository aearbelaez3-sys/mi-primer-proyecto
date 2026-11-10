import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema (Ejemplo 9.13)
r = 2.0          # Constante de amortiguamiento
sigma = 0.5      # Parámetro de volatilidad / ruido
T = 1.0          # Tiempo final de la simulación
N = 1000         # Número de pasos de tiempo
dt = T / N       # Tamaño del paso (Delta t)
M = 20_000       # Número de trayectorias (Monte Carlo)
w0 = 1.0         # Condición inicial w(0)
n_muestra = 5    # Trayectorias individuales que se dibujan en gris

rng = np.random.default_rng(42)   # Semilla para resultados reproducibles

# Tiempo y estado inicial
t = np.linspace(0, T, N + 1)
w = np.zeros((M, N + 1))   # Cada fila es una trayectoria
w[:, 0] = w0

# Euler-Maruyama (las M trayectorias a la vez)
for i in range(N):
    dB = np.sqrt(dt) * rng.normal(0, 1, size=M)              # Incremento browniano
    w[:, i + 1] = w[:, i] - r * w[:, i] * dt + sigma * dB     # Ecuación discreta 9.26

# ---- Estadísticas Monte Carlo en cada instante t ----
q05, q25, mediana, q75, q95 = np.percentile(w, [5, 25, 50, 75, 95], axis=0)
media_mc = w.mean(axis=0)

# ---- Solución exacta (proceso de Ornstein-Uhlenbeck) ----
# w(t) ~ Normal( w0 e^{-rt},  sigma^2/(2r) (1 - e^{-2rt}) )
media_exacta = w0 * np.exp(-r * t)                 # = solución de la EDO
desv_exacta = np.sqrt(sigma**2 / (2 * r) * (1 - np.exp(-2 * r * t)))
z95 = 1.6448536269514722                           # cuantil 95 % de la N(0,1)
q05_ex = media_exacta - z95 * desv_exacta
q95_ex = media_exacta + z95 * desv_exacta

# ---- Tabla en la terminal (como en la slide) ----
print(f"t = {T} {'':14s} exacto        Monte Carlo")
print(f"mediana            {media_exacta[-1]:8.3f}      {mediana[-1]:8.3f}")
print(f"media              {media_exacta[-1]:8.3f}      {media_mc[-1]:8.3f}")
print(f"cuantiles 5-95 %   {q05_ex[-1]:.3f} a {q95_ex[-1]:.3f}  {q05[-1]:.3f} a {q95[-1]:.3f}")

# ---- Gráfica tipo "abanico" ----
fig, ax = plt.subplots(figsize=(10, 5.5))

ax.fill_between(t, q05, q95, color='#dbe6f5', label='5–95 % (MC)')
ax.fill_between(t, q25, q75, color='#a9c1e6', label='25–75 % (MC)')

for k in range(n_muestra):
    ax.plot(t, w[k], color='gray', lw=0.7, alpha=0.8)

ax.plot(t, mediana, color='#1f4fbf', lw=2, label='mediana (MC)')
ax.plot(t, q05_ex, 'k--', lw=1.3, label='cuantiles exactos')
ax.plot(t, q95_ex, 'k--', lw=1.3)
ax.plot(t, media_exacta, color='#b22222', lw=2, linestyle=':',
        label='media exacta (= solución de la EDO)')
ax.axhline(0, color='gray', lw=0.8, alpha=0.6)

ax.set_title(f'Ecuación de Langevin: abanico de $2\\times10^4$ trayectorias',
             fontsize=14, color='#1f3864', fontweight='bold')
ax.set_xlabel('Tiempo (t)')
ax.set_ylabel('Posición w(t)')
ax.set_xlim(0, T)
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=9, frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig('resultado_langevin_abanico.png', dpi=150)
plt.show()
