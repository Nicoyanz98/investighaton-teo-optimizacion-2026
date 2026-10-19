# Dirección A: plantillas de clusters (19/10/2026, con Claude)

Notebook: `notebooks/02_plantillas.ipynb`. Código en `src/`, scripts en `scripts/`, datos en `data/`.

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
| Canale | 11/16 | plantilla + k-spinnings reales con `classify` y simulación |
| S7 límite | q*, w*, μ0 = 0.69290770637 | 20 dígitos, polinomio (3) |
| S7 finito | μ = 0.6929050246494 | **certificado propio**: Krawczyk + Cholesky por intervalos (`scripts/certify_s7.py`) |

Detalles:
- **Canale:** el paper tiene erratas en los conjuntos de aristas E_t. Usamos la geometría de la
  Fig. 2 y del Hessiano del Lema 1. La plantilla es estricta exactamente cuando c² > 2ad.
- **S7 finito:** con (t, h) = (10⁻⁵, 10⁻⁶), D = 2·10⁴². Reproduce la Proposición 1 de O'Keeffe a 13
  dígitos; el resto depende del redondeo racional.
- **Controles negativos:** con q > q* (t < 0) el Cholesky falla, y con Canale en d = 1/2 fallan el
  Krawczyk y el Cholesky.

## Exploración (evidencia numérica, no certificada)

**Patrones de partición de C4** (`scripts/explore_splits.py`, SDP interno + Nelder–Mead):
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

**Caso 0/1, blow-ups completos** (`scripts/explore_01.py`, nauty): con k ≤ 7 el mejor μ es ≈ 0.616.
Hacen falta densidades intermedias.

## Pendientes posibles
- k = 8 en el caso 0/1 (unas 2–3 h de cómputo).
- Modelo límite h → 0 para otros patrones (estilo Apéndice A).
- El estado "satélite" (0.69155) de O'Keeffe.
- Análogos con ℓ = 5, 6 clusters (dirección C).
