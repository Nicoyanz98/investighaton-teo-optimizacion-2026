# Notas: presentaciones, teoría y calentamientos 1–2

Resumen de lo charlado en la sesión del 17–18/10/2026 (con Claude). Complementa a `text.md`
(explicación larga de energía / sincronización / paisaje benigno) y al notebook
`notebooks/01_calentamiento_Kn.ipynb` (secciones 1 y 2).

---

## 0. Las presentaciones

**`charla_corta_kuramoto.pdf`**: motivación (luciérnagas, neuronas, metrónomos, láseres, redes), el
modelo, la energía y la pregunta central.

- Conectividad: μ(G) = (mín_v deg(v) + 1) / n.
- Conectividad crítica: μ_c = inf{μ : todo grafo con μ(G) ≥ μ sincroniza globalmente}.
- Estado del arte:
  - **Cotas superiores:** 1 (ejercicio) → 0.9395 (Taylor 2012) → ≈0.79 (Ling–Xu–Bandeira 2019, Lu–Steinerberger 2020) → 3/4 (Kassabov–Strogatz–Townsend 2021) → **3/4 − η** (Lepsveridze–Zhang, agosto 2026). Este último refuta la conjetura μ_c = 3/4.
  - **Cotas inferiores:** 0.6828 (Townsend–Stillman–Strogatz 2020) → 0.6838 (Yoneda–Tatsukawa–Teramae 2021) → **11/16 = 0.6875** (Canale 2022).
  - **Intervalo abierto:** 11/16 ≤ μ_c ≤ 3/4 − η.

**`charla_kuramoto_mc.pdf`** (34 slides), en orden:
1. El modelo y la energía (imagen de resortes); Kuramoto es un flujo gradiente.
2. Descenso de gradiente y paisajes benignos.
3. El grafo completo K_n sincroniza (parámetro de orden, "identidad mágica").
4. El problema abierto: twisted states en C_n y estado del arte. Lepsveridze–Zhang además prueban que, cerca de 3/4, el único obstáculo posible es un clustering en 4 grupos en {0, π/2, π, 3π/2}. Los grafos aleatorios y los expansores sincronizan con mucha menos conectividad; la dificultad está en el peor caso.
5. Matrix completion: la energía de Kuramoto **es** el objetivo de completar una matriz de rango bajo. Las matrices de diferencias de fase cumplen C + iS = zz*, que tiene rango 1. Aplicaciones: Netflix, Cryo-EM, relojes distribuidos, SLAM.
6. La competencia: 3 semanas.
   - **Calentamientos:** K_n; C_n; C_{n,k}.
   - **Direcciones:** A (buscar grafos con μ > 11/16 que no sincronicen), B (explicitar η), C (geometría cerca de 3/4), D (variantes), E (exposición creativa). Recomiendan empezar por A o E.
   - **Entrega:** informe de hasta 5 páginas con declaración **obligatoria** de uso de IA, repositorio con historiales de LLM, póster y charla de 15 minutos.

---

## 1. Teoría repasada

### 1.1 El modelo
- Fases θ_v ∈ (−π, π], una por vértice: una aguja de reloj, un punto del círculo. θ y θ + 2π son el mismo estado.
- Dinámica: θ̇_v = Σ_u w_uv sin(θ_u − θ_v). Cada vecino "tira" de v.
  - Desde la energía, la "fuerza" de un vecino es −∂/∂θ_v [1 − cos(θ_u − θ_v)] = sin(θ_u − θ_v).
  - Desde los resortes, es la componente tangente de la fuerza del resorte: Im(z̄_v (z_u − z_v)) = sin(θ_u − θ_v).
  - No hay inercia (la bolita en miel), así que la velocidad es directamente la suma de las fuerzas.
- Dos osciladores: φ = θ₂ − θ₁ cumple φ̇ = −2 sin φ. φ = 0 es estable y φ = π inestable, así que siempre sincronizan salvo que arranquen exactamente opuestos.

### 1.2 La energía
- 𝓔(θ) = Σ_aristas (1 − cos(θ_u − θ_v)) mide el **desacuerdo**: cada arista aporta 0 en fase y 2 en antifase.
- Es 1 − cos y no un cuadrado porque tiene que ser periódica; para diferencias chicas es ≈ x²/2.
- **Resortes:** |z_u − z_v|² = 2(1 − cos), así que 𝓔 = ½ Σ |z_u − z_v|² es la energía elástica de resortes que unen partículas sobre el círculo.
- **θ̇ = −∇𝓔** y d𝓔/dt = −‖∇𝓔‖² ≤ 0: la energía siempre baja. Los equilibrios son los puntos críticos.
- **Simetría de rotación:** sumar α a todas las fases no cambia nada. Por eso el mínimo es un círculo de configuraciones y el Hessiano siempre tiene un autovalor 0 con autovector (1, …, 1). Los certificados de mínimo piden Hessiano definido positivo **en el ortogonal** a esa dirección.
- **Hessiano:** es un Laplaciano con pesos w_uv cos(θ_u − θ_v).

### 1.3 Sincronizar
- **Estado sincronizado:** θ_u = θ_v en toda arista. Equivale a 𝓔 = 0, es decir, al mínimo global.
- **Grafo que sincroniza globalmente:** desde casi toda condición inicial termina sincronizado. Equivale a que los únicos mínimos locales sean los sincronizados.
- Dónde puede terminar una trayectoria:

  | final | ¿con probabilidad positiva? | ¿hace que G "no sincronice"? |
  |---|---|---|
  | mínimo global (sincronizado) | sí | — |
  | silla o máximo | no (medida cero) | no |
  | mínimo local no global | **sí** (cuenca de volumen positivo) | **sí** |

- "No sincroniza" no quiere decir caos: converge a un equilibrio estable con desfasajes fijos. Tampoco quiere decir "nunca sincroniza", sino que **no lo garantiza**.
- **ψ** es el ángulo del parámetro de orden r = (1/n) Σ e^{iθ_v} = |r| e^{iψ}, la dirección del centro de masa. En K_n, los puntos críticos con r ≠ 0 tienen cada fase en ψ o en ψ + π. También se usa ψ como "fase común cualquiera" del estado sincronizado.

### 1.4 Paisaje benigno
- El terreno es la energía sobre el toro de configuraciones, y el descenso de gradiente es soltar una bolita.
- **Benigno:** (i) todo mínimo local es global; (ii) toda silla es estricta (tiene un autovalor < 0).
- Con eso, el descenso de gradiente llega al óptimo desde casi cualquier inicio aunque la función no sea convexa.
- G sincroniza globalmente ⟺ el paisaje de 𝓔_G es benigno.
- μ_c es el umbral de grado mínimo a partir del cual **todo** grafo tiene paisaje benigno.

### 1.5 Twisted states en el ciclo C_n
- θ_v = 2πqv/n. Aquí **q es un parámetro**: el número de vueltas que se elige al construir el estado. Todas las aristas tienen diferencia d = 2πq/n.
- **Es equilibrio:** cada vértice tiene un vecino d adelante y otro d atrás, y sin d + sin(−d) = 0 porque el seno es impar.
- **Estabilidad con un oscilador:** moviendo v una cantidad ε, la velocidad queda sin(d − ε) − sin(d + ε) = −2 cos d · sin ε ≈ −2 cos d · ε. Vuelve si cos d > 0.
  - Intuición: el tirón sin(d) crece hasta 90°; pasando 90° el resorte "se rinde".
- **Estabilidad con todos a la vez:** con d_i = d + ε_i y Σε_i = 0 (las ε_i = δ_{i+1} − δ_i se cancelan de a pares),
  1 − cos(d + ε) ≈ (1 − cos d) + sin d · ε + (cos d/2) ε²,
  así que 𝓔 ≈ 𝓔* + (cos d/2) Σ ε_i². El término lineal se anula porque es un punto crítico.
- **Fórmula exacta:** H = cos d · L_{C_n}, con autovalores λ_k = cos(2πq/n)(2 − 2cos(2πk/n)).
  - |q| < n/4 → mínimo estricto (pozo trampa)
  - |q| > n/4 → máximo
  - |q| = n/4 → degenerado
- **Por qué n ≥ 5:** para q = 1, d = 360°/n < 90° ⟺ n > 4.
  - n = 3 (120°) es un máximo.
  - n = 4 (90°) es degenerado. A tercer orden, 𝓔 ≈ 𝓔* − (1/6) Σ ε_i³, y con ε = (−s, −s, −s, 3s) la energía baja: es una silla.
  - Que C_3 y C_4 sincronicen globalmente requiere además ver que no hay otras trampas: en un equilibrio todas las aristas tienen el mismo sin d_i, y los únicos estables son los twisted con |q| < n/4.

### 1.6 El winding number q como medición
- Hay dos usos de la letra q:
  - **(a)** el parámetro con el que se construye un twisted state;
  - **(b)** la medición de **cualquier** configuración:
    q(θ) = (1/2π) Σ_v wrap(θ_{v+1} − θ_v), con wrap llevando a (−π, π].
- **Qué representa:** es una propiedad de la *foto*, no de la *película*. Recorrés los vértices del ciclo una vez y pasás de cada uno al siguiente **por el arco más corto**; q es cuántas vueltas netas da la aguja en ese recorrido.
- **Por qué hace falta el wrap:** sin wrap, la suma de las diferencias da siempre 0. Con wrap, el resultado no depende de cómo se anoten los ángulos: sumarle 2π a un θ_v no cambia q.
- **Imagen del hilo:** un hilo pasa por las partículas en el orden del ciclo, cada tramo por el arco corto, y queda enrollado q veces alrededor del anillo. Para soltarlo, algún tramo tiene que medir media vuelta.
- **Por qué depende del tiempo:** θ(t) cambia, así que q(θ(t)) también. Es un entero que varía de forma continua salvo cuando alguna arista cruza ±π, así que es constante a tramos y **salta de a ±1** cuando dos vecinos quedan exactamente opuestos.
- **Qué nos aporta:**
  1. **La trampa es topológica.** Una arista en π aporta 2 a la energía, así que **si 𝓔 < 2, q queda congelado para siempre**, y si además q ≠ 0, seguro no sincroniza. Es una condición suficiente: en 2.7 la trampa q = −1 tiene 𝓔 = 2.34 > 2 y la garantiza el Hessiano.
  2. **Es una etiqueta barata.** En C_n el q final dice en qué pozo caíste, la energía final n(1 − cos(2πq/n)), y si sincronizó.
  3. **La dinámica no minimiza q.** Con energía alta, q salta libremente; pasar por q = 0 no es sincronizar, porque q = 0 es una región. q se congela recién cuando la energía es baja.
  4. **Conexión con μ_c (interpretación propia):** las construcciones de cotas inferiores son configuraciones "enroscadas" en grafos densos. Los 4 grupos en 0, π/2, π, 3π/2 de Lepsveridze–Zhang se parecen a un twisted q = 1 en 4 pasos de 90°, el caso borde n = 4.
- Para visualizarlo se armó un prompt para un artifact interactivo en Claude web: grafo, círculo de fases, "fase acumulada", el hilo, y escenas con q = 1, 0, 2, cambio de anotación y salto de q.

---

## 2. Calentamiento 1: K_n (notebook, sección 1)

- **Pipeline:**
  - `simulate`: descenso de gradiente con Euler y h = 1/(2·deg máx), lo que garantiza que la energía no suba. Para cuando ‖∇𝓔‖ < tol.
  - `classify`: autovalores del Hessiano → mínimo estricto, silla, máximo o degenerado.
- **Chequeos:** gradiente y Hessiano coinciden con diferencias finitas; la identidad 𝓔 = (n²/2)(1 − |r|²) y θ̇_v = n|r| sin(ψ − θ_v) se cumplen.
- **Una corrida (n = 10):** la energía baja de 34.8 a 0 en 37 pasos; |r| → 1; el final es un mínimo estricto.
- **Muchas corridas:** n ∈ {3, …, 100}, 200 cada uno → **100 % sincronizan**, todas en mínimo estricto. En el estado sincronizado el Hessiano es el Laplaciano de K_n (autovalores 0 y n).
- **Sillas:**
  - Con m osciladores en contra, 𝓔 = 2m(n − m) y hay m autovalores negativos.
  - m = n/2 (r = 0) es un máximo degenerado (H = −ssᵀ).
  - Arrancando exactamente en la silla, se queda; con ruido de 10⁻⁶, meseta y después escapa a sincronizar.

## 3. Calentamiento 2: C_n (notebook, sección 2)

- **Funciones nuevas:** `cycle_graph`, `twisted_state`, `winding_number`, `draw_cycle`.
- **Chequeos:** los twisted states son equilibrios (‖θ̇‖ ~ 10⁻¹⁵) y el espectro coincide con la fórmula cerrada.
- **Tabla n = 3…12:** la clasificación coincide con la regla |q| vs n/4 en **45/45** casos. Los twisted states nunca son sillas: los autovalores tienen el signo de cos d.
  - La tabla de 2.4 muestra además λ_min (autovalor no nulo más chico) junto al tipo: los pozos se vuelven más playos al crecer n (~4π²/n²) y al acercarse q a n/4.
- **Corrida atrapada (n = 20):** q pasa de 3 a 1; 𝓔 final = 0.979 = 20(1 − cos(2π/20)); mínimo estricto. La "fase acumulada" es una escalera recta.
- **Estadística (200 corridas por n):** la fracción que sincroniza baja de 93 % (n = 5) a 76 % (n = 8), 46 % (n = 20) y 34 % (n = 40).
  - Todos los finales son mínimos estrictos con |q| < n/4 y la energía de la fórmula.
  - q final = q inicial solo en un 30–60 % de las corridas.
  - La distribución del q final es acampanada alrededor de 0 y se ensancha con n.
- **Escape de un twisted inestable (n = 8, q = 3):** q pasa por 3 → 2 → 1 → 0 → −1 y queda trabado en q = −1 (𝓔 = 2.34).

## 4. Calentamiento 3: C_{n,k} (notebook, sección 3)

- **Grafo:** v conectado con v±1, …, v±k; grado 2k, conectividad μ = 2k/(n − 1). k = 1 es C_n; 2k = n − 1 es K_n.
- **Teoría:** los twisted states son equilibrios para todo k. El Hessiano es circulante, con autovalores λ_m = Σ_j cos(jd)(2 − 2cos(2πjm/n)). Las aristas largas con jd > 90° tienen peso negativo → aparecen sillas y las trampas se desestabilizan.
- **Funciones nuevas:** `circulant_graph`, `twisted_eigs` (fórmula cerrada), `is_trap`, `k_max`, `k_max_q1`.
- **Tabla n = 20:** q = 1 es mínimo hasta k = 6 (μ ≈ 0.63) y silla desde k = 7; los q grandes caen antes.
- **¿Hasta qué k?** (n = 5…200 con todos los q; hasta n = 5000 con q = 1):
  - los k con trampa forman siempre un intervalo 1…k_max, y el último en sobrevivir es siempre q = 1;
  - fórmula y `classify` coinciden para n ≤ 30;
  - k_max/n → 0.340 (n = 5000: k_max = 1701), o sea **μ → ≈ 0.681**: el resultado de Wiley–Strogatz–Girvan (2006), primera evidencia de μ_c ≥ 0.68. Queda por debajo de 11/16 → para mejorar hay que salir de C_{n,k} (dirección A).
- **Simulaciones (n = 30, k_max = 9, 200 corridas por k):** la fracción que sincroniza sube de 75/200 (k = 1) a 199/200 (k = 8, 9) y 200/200 desde k = 10. Todos los finales son mínimos estrictos y los no sincronizados son twisted states. Cerca de k_max la trampa existe pero su cuenca es diminuta: buscar trampas con μ alto pide análisis del Hessiano, no inicios al azar.

## 5. Lo que sigue

- **Elegir dirección** (A–E) para las 3 semanas; los calentamientos ya dejan el pipeline: simulador + test de estabilidad por el Hessiano + fórmula cerrada para familias circulantes.
- **Logística:**
  - Notebook en `notebooks/Calentamientos.ipynb`. Se ejecuta en Colab o con `.venv` local (~2 min).
  - Gráficos con nombres de variables en los ejes, sin títulos redundantes.
  - Para la declaración de uso de IA: estas notas, `text.md` y el historial de la sesión con Claude documentan cómo se usó.
