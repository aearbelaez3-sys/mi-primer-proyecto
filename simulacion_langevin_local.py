import numpy as np
import matplotlib.pyplot as plt

# Parámetros exactos del Ejemplo 9.13
r = 2.0          # Constante de amortiguamiento
sigma = 0.5      # Parámetro de volatilidad / ruido aditivo
T = 1.0          # Tiempo final de la simulación
N = 1000         # Número de pasos de tiempo
dt = T / N       # Tamaño del paso incremental (Delta t)
M = 5            # Número de trayectorias independientes a simular

# Configuración del arreglo de tiempo y la matriz de estados
t = np.linspace(0, T, N+1)
w = np.zeros((N+1, M))
w[0, :] = 1.0    # Condición inicial y(0) = 1 para todas las trayectorias

# Generar todos los incrementos Brownianos en un solo bloque (N pasos x M trayectorias)
dB = np.random.normal(0, np.sqrt(dt), size=(N, M))

# Simulación iterativa mediante el método de Euler-Maruyama (Ecuación 9.26)
for i in range(N):
    w[i+1, :] = w[i, :] - r * w[i, :] * dt + sigma * dB[i, :]

# Calcular la recta promedio de las 5 trayectorias
w_promedio = np.mean(w, axis=1)

# Graficar los resultados
plt.figure(figsize=(10, 5))

# 1. Graficar las 5 trayectorias individuales
for j in range(M):
    plt.plot(t, w[:, j], alpha=0.5, lw=1.0, label=f'Trayectoria {j+1}')

# 2. Graficar la recta promedio de las trayectorias (Línea discontinua roja)
plt.plot(t, w_promedio, color='red', linestyle='--', lw=2.5, label='Promedio de Trayectorias')

# Estética de la gráfica
plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
plt.title('Simulación de la Ecuación de Langevin (Euler-Maruyama) - Ejemplo 9.13')
plt.xlabel('Tiempo (t)')
plt.ylabel('Posición (y)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Guardar la gráfica en una imagen
plt.savefig('resultado_langevin_5_trayectorias.png', dpi=300)
plt.show()

