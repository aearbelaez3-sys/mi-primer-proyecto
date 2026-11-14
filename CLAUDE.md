# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto

Repositorio de un curso de modelación matemática. Todo el código, los comentarios, las gráficas y los mensajes de commit están en español; mantén ese idioma. No hay pruebas, linter ni sistema de build: cada archivo es un script independiente que simula una EDE, imprime estadísticas en la terminal y guarda una figura `resultado_*.png` en el directorio actual (sobrescribiéndola) antes de llamar a `plt.show()`.

## Entorno y comandos

- Entorno virtual de Python en `venv/` (ignorado por git): numpy, matplotlib, diffeqpy (vía juliacall) y JupyterLab.
- Julia está instalada con juliaup en `~/.juliaup/bin`. diffeqpy necesita que `julia` esté en el `PATH`; el cuaderno lo añade explícitamente con `os.environ["PATH"]` porque Jupyter puede abrirse sin ese PATH.
- La primera importación de `from diffeqpy import de` en una sesión es lenta (compilación de Julia); no es un cuelgue.

```bash
source venv/bin/activate
python3 simulacion_mbg_clase.py                       # parámetros de la clase (mu=0.25, sigma=1, X0=1, T=10, M=20)
python3 simulacion_mbg_clase.py --mu 0.25 --sigma 0.5 --M 50 --lineal --salida otra.png
python3 simulacion_mbg_clase.py --help                # responde rápido: diffeqpy se importa después de argparse
jupyter lab simulacion_mbg_clase.ipynb
```

Para ejecutar sin ventana gráfica (p. ej. al verificar desde la terminal): `MPLBACKEND=Agg python3 <script>.py`.

## Arquitectura

Dos familias de simulaciones, ambas con Euler-Maruyama:

**Movimiento browniano geométrico** `dX_t = mu X_t dt + sigma X_t dB_t` — usa **diffeqpy**, no un bucle manual:
- La EDE se define con `f(u,p,t) = p[0]*u` (deriva) y `g(u,p,t) = p[1]*u` (difusión), `de.SDEProblem(f, g, X0, (0,T), p)` y `de.solve(prob, de.EM(), dt=dt, saveat=dt, seed=...)`.
- Las M trayectorias se obtienen resolviendo M veces con semillas `seed + k` (reproducibles), y se apilan en un arreglo `(M, n_pasos)`.
- Referencias teóricas que se comparan: media `X0·e^{mu t}` (= solución determinista de `dx/dt = mu x`, resuelta también con `de.ODEProblem` + `de.Tsit5()`) y mediana `X0·e^{(mu - sigma²/2) t}`. Con `X0 = 0` todo es ≡ 0 porque el ruido es multiplicativo (hay que usar escala lineal).
- `simulacion_mbg_clase.py` (CLI con argparse) y `simulacion_mbg_clase.ipynb` son la versión principal, con el estilo gráfico "de la clase" (serif, fondo `#fbf7ef`, media en rojo `#a3262a`, mediana en azul `#2a63c6`, eje y logarítmico por defecto). Si cambias la lógica o el estilo en uno, mantén el otro consistente. `simulacion_mbg.py` es una versión anterior (comparación EM vs. determinista).
- `simulacion_mbg_vs_determinista.py` (CLI, mismas opciones que `simulacion_mbg_clase.py` salvo `--lineal`; M = 10 por defecto) compara las trayectorias con el modelo determinista en escala lineal y logarítmica, e imprime cuántas terminan por debajo de `x(T)` frente a la probabilidad teórica `Φ(sigma·√T/2)`.

**Ecuación de Langevin / Ornstein-Uhlenbeck** `dw = -r w dt + sigma dB` (Ejemplo 9.13, ecuación discreta 9.26 del texto del curso) — Euler-Maruyama vectorizado a mano con numpy (`simulacion_langevin*.py`). `simulacion_langevin_abanico.py` compara cuantiles Monte Carlo con la solución exacta `N(w0 e^{-rt}, sigma²/(2r)(1-e^{-2rt}))`.

## Notas del repositorio

- `mi-primer-proyecto/` es un clon anidado antiguo con su propio `.git`, sin seguimiento; no lo modifiques ni lo agregues al repositorio.
- Hay archivos locales sin seguimiento (`regresion_consumo_energetico.py`, `reporte_regresion_consumo.tex`, `prueba.*`) que no forman parte del historial; no los incluyas en commits salvo que se pida.
- Remoto: GitHub `aearbelaez3-sys/mi-primer-proyecto`; el trabajo de Claude se ha hecho en ramas `claude/*` integradas a `main` mediante PR.
