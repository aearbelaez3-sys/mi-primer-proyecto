import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema (Ejemplo 9.13)
r = 2.0          # Constante de amortiguamiento
sigma = 0.5      # Parámetro de volatilidad / ruido espacial
T = 1.0          # Tiempo final de la simulación
N = 1000         # Número de pasos de tiempo
dt = T / N       # Tamaño del paso incremental (Delta t)
M = 5            # Número de trayectorias a simular
w0 = 1.0         # Condición inicial w(0)

rng = np.random.default_rng(42)   # Semilla para resultados reproducibles

# Configuración del arreglo de tiempo y el estado inicial
t = np.linspace(0, T, N+1)
w = np.zeros((M, N+1))   # Cada fila es una trayectoria
w[:, 0] = w0

# Simulación iterativa mediante el método de Euler-Maruyama (M trayectorias a la vez)
for i in range(N):
    Z = rng.normal(0, 1, size=M)          # Una normal estándar por trayectoria
    dB = np.sqrt(dt) * Z                  # Incremento browniano
    w[:, i+1] = w[:, i] - r * w[:, i] * dt + sigma * dB   # Ecuación discreta 9.26

# Promedio de las trayectorias simuladas
promedio = w.mean(axis=0)

# Solución de la EDO (sin ruido, sigma = 0): w(t) = w0 * exp(-r t)
w_edo = w0 * np.exp(-r * t)

# Graficar
plt.figure(figsize=(10, 5))
for k in range(M):
    plt.plot(t, w[k], lw=1, alpha=0.8, label=f'Trayectoria {k+1}')
plt.plot(t, promedio, color='red', lw=2, linestyle='--', label=f'Promedio de las {M} trayectorias')
plt.plot(t, w_edo, color='black', lw=2.5, label='Solución de la EDO')
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f'Simulacion de la Ecuacion de Langevin: {M} trayectorias y la solucion de la EDO')
plt.xlabel('Tiempo (t)')
plt.ylabel('Posicion w(t)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('resultado_langevin.png', dpi=150)
plt.show()
