import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema (Ejemplo 9.13)
r = 2.0          # Constante de amortiguamiento
sigma = 0.5      # Parámetro de volatilidad / ruido espacial
T = 1.0          # Tiempo final de la simulación
N = 1000         # Número de pasos de tiempo
dt = T / N       # Tamaño del paso incremental (Delta t)

# Configuración del arreglo de tiempo y el estado inicial
t = np.linspace(0, T, N+1)
w = np.zeros(N+1)
w[0] = 1.0       # Condición inicial de la posición w(0)

# Simulación iterativa mediante el método de Euler-Maruyama
for i in range(N):
    # Generar la variable normal estándar Z_i
    Z = np.random.normal(0, 1)
    
    # Incremento del movimiento Browniano: Delta B = sqrt(dt) * Z
    dB = np.sqrt(dt) * Z
    
    # Ecuación discreta 9.26 corregida
    w[i+1] = w[i] - r * w[i] * dt + sigma * dB

# Graficar la trayectoria resultante de la ecuación
plt.figure(figsize=(10, 5))
plt.plot(t, w, label='Trayectoria del Proceso', color='dodgerblue', lw=1.5)
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.title('Simulacion de la Ecuacion de Langevin (Euler-Maruyama)')
plt.xlabel('Tiempo (t)')
plt.ylabel('Posicion (y)')
plt.grid(True, alpha=0.3)
plt.legend()

# Guardar la gráfica en una imagen
plt.savefig('resultado_langevin.png', dpi=300)
plt.show()
