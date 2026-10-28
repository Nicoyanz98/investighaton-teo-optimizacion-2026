# Grafo explícito tipo S7: de la plantilla de O'Keeffe a un grafo con n "chico"

**Estado:** Fases 0–3 hechas. Hay un grafo explícito, no pesado y **certificado por intervalos** con
n = 30 000 004 y μ = 10327019/15000002 ≈ 0.688468 > 11/16. La validación ciega en float64 (Fase 4)
coincide, incluido el Lanczos ciego en GPU. Primera versión del notebook en
`notebooks/Grafo_explicito_S7.ipynb`. Con el camino B (sub-piezas) hay uno certificado con
**n = 1 500 014** y μ = 1032699/1500014 ≈ 0.688463.

## Motivación
- O'Keeffe (arXiv 2610.21271) prueba μ_c ≥ 0.6929 con la familia S7, pero el grafo que certifica
  tiene n > 9·10⁸⁴ y nunca lo escribe. El tamaño viene de dos lados:
  - D ≈ 3·10⁴²: racionaliza un miembro que está a 6·10⁻¹⁹ de ser neutro;
  - n = D²ℓ: el blow-up aleatorio de su Lema 2.
- Canale da un grafo explícito (k-spinning, n = 8k) que con n = 10⁴ llega a 0.6874 < 11/16.
- **Objetivo:** un grafo no pesado, explícito y certificado, con μ > 11/16 y n lo más chico posible.

## Notación
- Clusters (Am, Ap, Bm, Bp, Cm, Cp, W): tres sectores de C4 partidos en mitades y W entero.
- **Plantilla:** (x, φ, P) con fases φ = fase del sector + h·s.
- **Datos enteros:**
  - tamaños n_a = k_a·g, con k = (k_A × 6, k_W);
  - grados d_ab = vecinos que tiene un vértice de a en b, con n_a d_ab = n_b d_ba y d_aa = n_a − 1.
  - La plantilla del grafo es x_a = n_a/n, P_ab = d_ab/n_b, y μ = (δ + 1)/n = min_a d_a.
- **Jacobiano** del equilibrio: J = ∂F/∂φ (φ_0 fijo), con valores singulares σ_1 ≥ … ≥ σ_6.

## Fase 0: frontera μ contra margen (`scripts/curva_s7.py` → `data/s7_curva.json`)
- **Tamaños racionales desde el principio:** x_a = k_a/K. El óptimo pide k_W/k_A ≈ 1 + q* ≈ 1.65:
  - (3, 5) y (20, 33) andan bien;
  - (5, 8), con cociente 1.6, pierde mucho (0.6826).
- **Barrido:** h ∈ {0.005, …, 0.05} y margen del SDP ε ∈ {1e-5, …, 1e-2}, donde ε es el margen de
  L/h² ⪰ ε.
- **Resultado:** μ baja al subir ε, y σ_min crece como ~ε·h². La frontera útil con μ > 11/16 es:

  | h | ε | μ plantilla | σ_min | n redondeo ingenuo |
  |---|---|---|---|---|
  | 0.01 | 1e-3 | 0.68961 | 2.6e-7 | ~10¹² |
  | 0.02 | 1e-3 | 0.68918 | 1.0e-6 | ~10¹⁰ |
  | 0.03 | 1e-3 | 0.68847 | 2.3e-6 | ~10⁹ |

- **El óptimo del SDP no está en equilibrio exacto:** F ≈ 1e-9, y eso solo ya mueve las fases
  ~1e-5. Por eso `pulir` corrige P con mínima norma (F es lineal en P) para que F(φ) = 0 exacto en
  alta precisión, y recién después se redondea.

## Por qué el redondeo es caro: el pliegue
- J tiene **tres** valores singulares chicos, los tres modos neutros heredados de C4. Con h = 0.01 y
  ε = 1e-3: 8·10⁻⁵, 6·10⁻⁷ y 3·10⁻⁷. Los otros son ≥ 0.13.
- Cerca de S7, el mínimo estricto está al lado de un pliegue (silla-nodo). En la dirección blanda k, el
  equilibrio es raíz de algo como σ_k y + (c/2) y² + ρ_k = 0, donde ρ_k es el residuo de esa ecuación.
  - Existe solo si 2cρ_k ≤ σ_k².
  - Su estabilidad pasa a ser √(σ_k² − 2cρ_k).
- **Redondeo ingenuo:** ρ ~ 1/n, así que hace falta n ≳ c/σ². Es lo que se mide en la tabla.
- **Lo vimos numéricamente:** con n chico, Newton se estanca en |F| ~ 1e-9 porque no hay equilibrio
  cerca. Los "autovalores negativos" que salían antes eran de puntos que no son equilibrios.

## Fase 1: redondeo por reticulado (`src/enteros.py`, `src/reticulado.py`)
- **Reducción de Lyapunov–Schmidt.** Con J = U S Vᵀ en la plantilla:
  - se resuelven exactamente las ecuaciones rígidas U_rᵀ F(φ* + V_r z) = 0 (bien condicionadas);
  - queda el residuo blando g = U_bᵀ F, de 3 componentes. g está bien definido aunque no haya
    equilibrio, y es casi lineal en los grados.
- **Newton entero.** En cada paso se elige u ∈ Z^m (un entero por par con densidad intermedia) con
  dg·u ≈ −g, midiendo cada componente en unidades de σ_k² (los pesos del código). Se usa LLL + Babai
  con el encaje de Kannan (`enteros_cercanos`).
- **Cuántos enteros libres (m).** Hay un entero por par a < b (d_ba queda fijado por n_a d_ab = n_b d_ba).
  De los 21 pares, 6 son opuestos (A–C, B–W) y los que el SDP deja en 0 o 1 quedan fijos (`pares_libres`,
  tolerancia 1e-7). Depende de la plantilla:
  - h = 0.03 (y 0.05): **m = 9**. Los 4 cruzados (Am–Bp, Ap–Bm, Bm–Cp, Bp–Cm) y 5 vecinos que bajan un
    poco de 1 (Ap–Bp, Bm–Cm, Am–W, Ap–W, Cm–W); los otros 6 vecinos quedan en 1.
  - h ≤ 0.02: **m = 15**. El SDP no deja ningún vecino en 1.
  (Una versión anterior de esta nota decía 13 para todas las plantillas.)
- **Granularidad.** Para que n_a d_ab = n_b d_ba con mitad y W, d_aW tiene que ser múltiplo de
  k_W/mcd(k_A, k_W), así que P_aW se mueve de a 1/mcd(n_A, n_W). Esto favorece tamaños con k chicos:
  (3, 5) da pasos de 23/n, y (20, 33) de 153/n.
- **Límite del método (heurístico).** El residuo blando tiene que quedar ≲ σ²/(2|c|) (el umbral del
  pliegue de la sección anterior), y cada corrección tiene que mover las densidades ≲ σ (si no, los
  términos cuadráticos del rígido lo arruinan). Con m enteros y 3 restricciones da n ≳ C·σ^{−(1+3/m)}:
  σ^{−1.33} con m = 9 (h = 0.03) y σ^{−1.2} con m = 15 (h ≤ 0.02), contra C·σ^{−2} del ingenuo. El
  factor 1/(2|c|) solo cambia la constante C, no el exponente. Con más enteros libres (partir clusters en
  sub-piezas de la misma fase) tendería a n ~ C/σ.

### Resultados (evaluación exacta en alta precisión, todavía sin intervalos)
`scripts/s7_entero.py` → `data/s7_enteros.json`.

| plantilla | n | redondeo ingenuo | reticulado | μ del grafo |
|---|---|---|---|---|
| h = 0.03, ε = 1e-3, (3, 5) | 1·10⁷ | no | no | — |
| h = 0.03, ε = 1e-3, (3, 5) | 3·10⁷ | no | **sí** (λ = 5.3e-6) | **0.688468** |
| h = 0.02, ε = 1e-3, (3, 5) | 3·10⁷ | no | no | — |
| h = 0.02, ε = 1e-3, (3, 5) | 1·10⁸ | no | **sí** (λ = 1.1e-6) | **0.689183** |
| h = 0.01, ε = 1e-3, (3, 5) | 1·10⁹ | no | no (λ = −8e-8, por poco) | — |

Comparación: Canale tiene 0.6874 con n = 10⁴, y O'Keeffe 0.6929 con n > 10⁸⁵, sin escribir el grafo.

## Fase 2: blow-up circulante (`src/blowup.py`)
- **Vértices:** el cluster a tiene k_a bloques, cada uno una copia de Z_g; el vértice es (a, i, v).
- **Dentro de un cluster:** clique (cada bloque es Z_g ∖ {0} y los bloques se unen completos).
- **Entre clusters a < b:** (a, i, v) ~ (b, j, v + s) con s ∈ {0, …, t_ij − 1}, un intervalo como en
  el k-spinning de Canale.
- **Tabla t (k_a × k_b), un problema de transporte:** filas d_ab, columnas d_ba, entradas en
  {base, base + 1}.
  - Los "+1" de la fila i van en las columnas cíclicas i·e, …, i·e + e − 1, con e = d_ab − k_b·base.
  - Así cada columna recibe los mismos y no hace falta que g sea múltiplo de nada.
- **Grados exactos:** cada vértice de a tiene exactamente d_ab vecinos en b, así que las fases de la
  plantilla entera son un equilibrio exacto del grafo.
- **Descripción publicada** (`descripcion_compacta`): k, g, d, las tablas y las fases. Son unos pocos
  KB por grafo.

## Fase 3: certificado del grafo completo (`blowup.certificar_blowup`, `scripts/certificar_grafo_s7.py`)
- **Descomposición de Fourier.** El Hessiano conmuta con v → v + 1, así que se parte en g bloques
  H_m de K × K (K = 23).
  - Lo verifica `tests/test_blowup.py`: el espectro denso es la unión de los espectros de H_m.
- **Modos de cluster** (los vectores constantes por cluster, dentro de H_0): dan exactamente n²·L(y).
  Los certifica `certificado.certificar` (Krawczyk + Cholesky por intervalos), que además devuelve la
  caja de fases.
- **Todos los demás modos:** cota de Schur + Gershgorin de 7 × 7, con los cos en intervalos sobre la
  caja. El margen del cluster a es Σ_b d_ab cos_ab + 1 − Σ_{b≠a} |cos_ab| √s_ab.
  - Para m ≠ 0: |N̂(m)| ≤ min(t, g − t), y entonces s_ab = min(d_ab, n_b − d_ab)·min(d_ba, n_a − d_ba).
    **No depende de qué conjuntos de desplazamientos se elijan.**
  - Para H_0 ortogonal a los modos de cluster: solo actúa E = t − base, con filas y columnas ≤ k.
  - Margen medido: ≥ 0.217·n en todos los casos. Es enorme porque las densidades intermedias están
    entre clusters casi ortogonales (cos = O(h)) o en pares con densidad casi 0 o casi 1.
- **Conclusión:** toda la dificultad está en el cociente de 7 × 7; el blow-up no cuesta nada.
- **Resultados** (`data/grafos_s7.json`, 1.6 s en total):
  - los 7 grafos del reticulado salen **certificados**;
  - controles negativos (redondeo ingenuo del mismo n): Krawczyk encuentra el equilibrio, pero el
    Cholesky de L falla (pivote < 0);
  - el único ingenuo que pasa es h = 0.03 con n = 10⁹, como en la Fase 1.

  | h | n | μ certificado | pivote mínimo de L | margen del resto |
  |---|---|---|---|---|
  | 0.03 | 3.0·10⁷ | 10327019/15000002 ≈ 0.688468 | 1.6e-6 | 0.218 n |
  | 0.02 | 1.0·10⁸ | ≈ 0.689183 | 2.9e-7 | 0.217 n |

## Fase 4: validación ciega (`src/validacion.py`, `scripts/validar_grafo.py`)
- **Sin Fourier ni Gershgorin.** Solo usa la regla de adyacencia de la descripción compacta.
  - W·y implícito: el clique de cada cluster más las ventanas cíclicas con sumas prefijas (cada bloque
    centrado por su media para no perder precisión). Cuesta O(n·k) y la memoria es O(n).
  - H·x = h∘x − cosθ∘W(cosθ∘x) − sinθ∘W(sinθ∘x), en float64, con numpy o torch (CPU/CUDA).
  - Tests contra el grafo denso, incluido un análogo con las densidades reales y g = 60.
- **Chequeos:** grados con W·1_b; velocidad; Q_ab = 1_aᵀH1_b/√(n_a n_b); Lanczos en el complemento de los
  modos de cluster; y, con `--completo`, Lanczos en 1^⊥ desde un vector al azar.
- **Detalle numérico de Lanczos:** el espectro del resto es muy degenerado y β cae a ~1e-10 en pocos
  pasos. Hay que reproyectar w antes de normalizar; si no, aparece un valor de Ritz espurio ≈ 0.
- **Resultados en CPU (n = 3·10⁷, 4.5 s por producto H·x):**
  - grados exactos; max |θ̇| = 1.5e-9 (≈ 5e-17·n);
  - |Q implícita − fórmula| = 2e-9; autovalores de Q: 0 (rotación), **157.5**, 210.5, 2.25e4 y tres
    ≈ 7.8e6;
  - resto: λ_min = 6.5372e6, contra la cota certificada de Gershgorin 6.5283e6. La cota es casi justa.
  - Por lo tanto λ_min(H en 1^⊥) = 157.5 > 0.
- **GPU** (server, GTX 1080 Ti; 0.73 s por producto H·x, 83 s en total, `data/validacion_s7_n30000004.json`):
  - repite los chequeos 1–4;
  - el Lanczos ciego en 1^⊥ desde un vector al azar da λ_min = 157.51123331 (residuo 7.5e-8), que
    coincide con Q con diferencia relativa 1e-10.
  - Convergió en 50 pasos y no en miles como estimé: el espectro son unos pocos autovalores aislados
    más una banda lejana [6.5e6, 7.95e6].
  - **Fase 4 completa:** la validación ciega confirma el certificado.

## Camino B: sub-piezas de la misma fase (`enteros.subdividir`, `scripts/s7_subpiezas.py`)
- **Idea.** Partir un cluster en piezas con la misma fase y las mismas densidades, unidas completas
  entre sí. Por ejemplo, W (k = 5) en piezas de k = 2 y 3.
  - Cada pieza ve lo mismo que el cluster original, así que F = 0 sigue exacto.
  - El modo relativo entre las piezas es rígido: no aparece ningún valor singular chico nuevo en J.
  - Cada par W–b se duplica, y el reticulado gana enteros libres para las mismas 3 ecuaciones blandas.
- **Certificado:** `certificar` ahora pide "no sync" (algún par de fases distinto), no fases distintas
  dos a dos; `distintas` queda como dato informativo. El blow-up y Gershgorin no cambian.
- **Sonda** (h = 0.03; un solo g por n):
  - sin partir: hace falta 3·10⁷;
  - W → (2, 3) o W → (1, 2, 2): funciona desde 2·10⁶;
  - mitades → (1, 2) (30 enteros libres): 10⁷;
  - todo partido: 3·10⁶.

  Partir las mitades no ayuda; con más dimensiones, quizás los parámetros del reticulado (radios,
  peso) quedan mal ajustados.
- **Barrido** (`data/s7_subpiezas.json`; 8 valores consecutivos de g por cada n, porque el éxito del
  reticulado no es monótono en n):
  - h = 0.03, W → (2, 3): el menor certificado es **n = 1 500 014, μ = 1032699/1500014 ≈ 0.688463**.
    Tasa de éxito: 3/8 con 1.5·10⁶, 7/8 con 2·10⁶, 2/8 con 3·10⁶, y 0/8 con 10⁶ o menos.
  - h = 0.02: nada hasta 3·10⁶. En la sonda, W → (2, 3) pasa con 3·10⁷ (sin partir hacía falta 10⁸).
  - Todos los equilibrios que el reticulado declaró estrictos (26) salieron certificados.
- Es una reducción de 20× sobre la versión sin partir.
- **Sonda de radios** (W → (2, 3), h = 0.03, 8 valores de g): con radios hasta 3000 en lugar de 100, el
  éxito con 3·10⁶ sube de 2/8 a 5/8. Con 7·10⁵ y 10⁶ sigue en 0/8. Los radios explican la falta de
  monotonía, pero no mueven el piso de ~1.5·10⁶.

## Pendiente
- **En revisión (23/10/2026):** el grafo actual es exacto y certificado, pero es una variante
  optimizada de S7 y no la construcción de O'Keeffe con menos nodos. Los objetivos para un grafo con
  forma reconocible (simetría espejo, densidades puras, familia G_ℓ) están en
  `notes/objetivos_grafo_elegante.md`, todavía sin decidir. Lo de abajo queda en pausa hasta entonces.
- **Camino B, siguiente:** por qué el éxito no es monótono en n (¿radios del reticulado?), particiones
  con más piezas, y h = 0.02 (más μ).
- **Notebook:** agregar el camino B (la Fase 4 completa ya está).
- **Otras mejoras:**
  - optimizar la plantilla pidiendo σ_min ≥ objetivo en lugar de margen del SDP;
  - mejor Nelder–Mead (la curva es ruidosa: (11, 18) a veces sale peor que (3, 5)).
