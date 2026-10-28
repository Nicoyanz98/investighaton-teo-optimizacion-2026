# Dirección A: plantillas de clusters (19/10/2026, con Claude)

Notebook: `notebooks/Implementacion_papers.ipynb`. Código en `src/`, scripts en `scripts/`, datos en `data/`.

## Contexto
- La idea del "stackeo" (`notes/idea.md`) lleva al marco de plantillas (x, φ, P).
- O'Keeffe (`papers/2610.21271v1.pdf`, 15/10/2026) usa exactamente ese marco y prueba
  **μ_c ≥ 0.6929077**, superando el 11/16 de la presentación.
- Sus scripts no son públicos, así que hicimos una reproducción independiente y exploramos
  variantes.

## Reproducido (todo con código propio)

| construcción | valor | cómo se chequeó |
|---|---|---|
| Blow-up G[K_m] | μ igual, trampa preservada | Hessiano; el cartesiano baja μ |
| C4 | μ = 3/4, L ≡ 0 | degenerado |
| Twisted como LP | **0.683876** (m = 19) | = Yoneda et al. 0.6838 |
| Canale | 11/16 | plantilla + k-spinnings reales con `clasificar` y simulación |
| S7 límite | q*, w*, μ0 = 0.69290770637 | 20 dígitos, polinomio (3) |
| S7 finito | μ = 0.6929050246494 | **certificado propio**: Krawczyk + Cholesky por intervalos (`scripts/certificar_s7.py`) |

Detalles:
- **Canale:** el paper tiene erratas en los conjuntos de aristas E_t. Usamos la geometría de la
  Fig. 2 y del Hessiano del Lema 1. La plantilla es estricta exactamente cuando c² > 2ad.
- **S7 finito:** con (t, h) = (10⁻⁵, 10⁻⁶), D = 2·10⁴². Reproduce la Proposición 1 de O'Keeffe a 13
  dígitos; el resto depende del redondeo racional.
- **Controles negativos:** con q > q* (t < 0) el Cholesky falla, y con Canale en d = 1/2 fallan el
  Krawczyk y el Cholesky.

## Exploración (evidencia numérica, no certificada)

**Patrones de partición de C4** (`scripts/explorar_particiones.py`, SDP interno + Nelder–Mead):
- Con arranques al azar no se reencuentra ni S7 (0.587 frente a 0.691). Las cuencas son angostas,
  como advierte O'Keeffe.
- Con arranques "estilo S7": partir el cluster ancla, o menos de tres clusters, empeora.
- (3,3,3,1) le gana a S7 con h fijo, pero la ventaja → 0 cuando h → 0:

  | h | ventaja de (3,3,3,1) |
  |---|---|
  | 0.02 | +3.2e-4 |
  | 0.01 | +1.7e-4 |
  | 0.005 | +2e-5 |

  Es consistente con O'Keeffe ("más piezas no mejoran μ0").
- Nuestros valores de S7 coinciden con su tabla: 0.6919 con h = 0.01 y 0.6924 con h = 0.005.

**Caso 0/1, blow-ups completos** (`scripts/explorar_01.py`, nauty): con k ≤ 7 el mejor μ es ≈ 0.616.
Hacen falta densidades intermedias.

## Pendientes posibles
- k = 8 en el caso 0/1 (unas 2–3 h de cómputo).
- Modelo límite h → 0 para otros patrones (estilo Apéndice A).
- El estado "satélite" (0.69155) de O'Keeffe.
- Análogos con ℓ = 5, 6 clusters (dirección C).

## Mapa del código (después del refactor del 19/10)

| dónde | qué |
|---|---|
| `src/grafos.py` | grafos (completo, ciclo, circulante), blow-ups, conectividad, grafos de nauty |
| `src/kuramoto.py` | dinámica, energía, Hessiano, simulación, clasificación, twisted states |
| `src/plantillas.py` | **única** implementación de d, F, c, L (float, mpmath o intervalos); `Plantilla`; `realizar`; C4 y Canale |
| `src/twisted_lp.py` | twisted como LP |
| `src/s7.py` | construcción de S7 |
| `src/certificado.py` | racionalizar, Newton, Krawczyk + Cholesky por intervalos |
| `src/busqueda.py` | SDP interno; `FamiliaParticiones` sobre cualquier `Base` (`base_c4`, `base_canale`); semillas; geometrías `una_red`, `dos_redes`, `red_aurea` |
| `src/aurea.py` | familia girasol: puntos y estado áureo, circulantes de Fibonacci, grafo geométrico áureo (`notes/familia_aurea.md`) |
| `src/experimentos.py` | paralelismo y JSON en `data/` |
| `scripts/` | `explorar_particiones.py`, `explorar_01.py`, `certificar_s7.py`, `barrer_semillas.py`, `explorar_aurea.py` |
| `tests/test_regresion.py` | fija todos los números de arriba: `.venv/bin/python -m pytest tests` |
| `tests/test_aurea.py` | verifica numéricamente los lemas y proposiciones de la familia girasol |
| `tests/test_notebook_sincronizado.py` | avisa si las funciones copiadas en `Calentamientos.ipynb` dejan de coincidir con `src/` |
| `notebooks/Calentamientos.ipynb` | autocontenido (funciones copiadas de `src/`): se abre directo en Colab |
| `notebooks/Implementacion_papers.ipynb` | reproducción de los papers, explicada para quien no los leyó; usa `src/` y `data/` |
| `notebooks/Familia_aurea.ipynb` | familia girasol (ángulo áureo): teoría, tres variantes y resultados; usa `src/` y `data/` |

Para probar la vía B (partir Canale) ya alcanza con:
`scripts/explorar_particiones.py --base canale --h 0.02 --patrones 21212121`.

**Colab.** `Implementacion_papers.ipynb` empieza con una celda que, solo en Colab, clona el repo (privado)
con un token de GitHub guardado en los Secrets como `GITHUB_TOKEN`. En local no hace nada.
Para lo pesado (scripts de exploración) conviene la máquina local: Colab gratuito tiene 2 núcleos y
nada del cálculo usa GPU (SDP con Clarabel, mpmath e intervalos son solo CPU).

**Notación.** **1** = (1, …, 1) es la rotación global, que nunca cambia la energía; **1**^⊥
("1 perpendicular") son los vectores que suman 0. "L ≻ 0 en **1**^⊥" = definida positiva salvo en la
rotación global, que es lo que se pide para un mínimo estricto.
