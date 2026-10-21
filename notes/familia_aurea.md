# Familia "girasol": osciladores separados por el ángulo áureo (21/10/2026)

Código: `src/aurea.py`, `red_aurea` en `src/busqueda.py`, `scripts/explorar_aurea.py`.
Notebook: `notebooks/Familia_aurea.ipynb`. Tests: `tests/test_aurea.py` (verifican cada afirmación de abajo).
Datos: `data/aurea_fibonacci.json`, `data/aurea_geometrico.json`, `data/aurea_plantillas.json`.

## Motivación

Los mentores sugirieron elegir **una familia** de grafos y analizarla a fondo antes de intentar subir
la cota. La propuesta: la distribución del girasol, donde la semilla v está girada v·γ respecto de la
anterior, con γ = 2π/φ² ≈ 137.5° (el **ángulo áureo**, φ = (1 + √5)/2).

La leemos sobre el círculo de fases: el oscilador v está en la posición x_v = v/φ mod 1 (usar 1/φ o
1/φ² da lo mismo salvo la reflexión θ → −θ, que es simetría de la energía), y se unen los que quedan
cerca. Es una familia 1D y densa, así que habla directo de μ_c.

**Resumen del resultado.** La familia es muy analizable, pero **no supera las cotas conocidas**, y
para casi todo lo podemos explicar por qué:

| variante | qué es | resultado | tipo |
|---|---|---|---|
| 1 | circulantes C_{F_k}(S), S ⊂ distancias de Fibonacci | μ ≤ 0.6838 siempre; en la práctica μ → 0 | demostrado + numérico |
| 2 | grafo geométrico sobre x_v = v/φ, radio r | n Fibonacci: es exactamente un anillo C_{n,ρ}; n no Fibonacci: peor que el anillo | demostrado (Fib) + numérico |
| 3 | plantillas con clusters en las fases áureas | ≤ 0.679 para m ≤ 40, por debajo de 0.6838 | numérico |

## Notación

- F_1 = F_2 = 1, F_{j+1} = F_j + F_{j−1}.
- Para S ⊂ Z_n con S = −S y 0 ∉ S, el **circulante** C_n(S) une u ~ v ⟺ v − u ∈ S (mod n). Tiene
  grado |S| y μ = (|S| + 1)/n. El anillo C_{n,k} es C_n({±1, …, ±k}).
- El **twisted state** de q vueltas es θ^{(q)}_v = 2π q v/n.
- **1^⊥**: los vectores perpendiculares a 1 = (1, …, 1), es decir, todas las direcciones salvo la
  rotación global. "Mínimo estricto" significa Hessiano definido positivo en 1^⊥.

## Teoría

### Lema 1 (twisted states en un circulante cualquiera)

θ^{(q)} es equilibrio de C_n(S) y los autovalores de su Hessiano son

  λ_m = Σ_{s∈S} cos(2π q s/n) · (1 − cos(2π m s/n)),   m = 0, …, n − 1,

con autovectores e^{2πi m v/n}. En particular, es mínimo estricto ⟺ λ_m > 0 para todo m ≠ 0.

*Prueba.*
- **Equilibrio.** La velocidad del vértice v es Σ_{s∈S} sin(θ_{v+s} − θ_v) = Σ_{s∈S} sin(2π q s/n),
  que vale 0 porque S = −S y el seno es impar.
- **Hessiano.** H_vv = Σ_{s∈S} cos(2π q s/n) y H_{v,v+s} = −cos(2π q s/n), que no depende de v. H es
  circulante, así que lo diagonalizan las ondas e^{2πi m v/n}, con autovalor
  λ_m = Σ_s cos(2π q s/n) − Σ_s cos(2π q s/n) e^{2πi m s/n}. La parte imaginaria se cancela
  agrupando s con −s y queda la fórmula.
- **Estrictez.** λ_0 = 0 es la rotación global. ∎

Con S = {±1, …, ±k} cada par ±s aporta 2(1 − cos), que es el factor (2 − 2cos) de
`autovalores_twisted`. Implementado en `autovalores_twisted_conjunto` (`src/kuramoto.py`).

### Lema 2 (multiplicadores: todo twisted state es un q = 1)

Si mcd(q, n) = 1, la biyección σ(v) = q v mod n es un isomorfismo C_n(S) → C_n(qS) que lleva θ^{(q)}
al twisted state θ^{(1)} de C_n(qS). Los dos estados tienen el mismo espectro del Hessiano, el mismo
tipo y la misma conectividad.

*Prueba.*
- σ es biyección porque q es invertible mod n.
- **Preserva aristas:** v − u ∈ S ⟺ q(v − u) ∈ qS ⟺ σ(v) − σ(u) ∈ qS.
- **Lleva un estado al otro:** θ^{(q)}_v = 2π q v/n ≡ 2π σ(v)/n (mod 2π).
- **Espectros:** energía y Hessiano dependen solo de aristas y diferencias de fase, así que
  H′ = Π H Πᵀ con Π la permutación de σ.
- **Conectividad:** |qS| = |S|. ∎

### Lema 3 (la congruencia de Fibonacci: las espirales del girasol)

Para 2 ≤ j < k:  F_{k−1} F_j ≡ (−1)^{j−1} F_{k−j}  (mod F_k).

*Prueba.* La identidad de d'Ocagne, F_a F_{b+1} − F_{a+1} F_b = (−1)^b F_{a−b}, con a = k − 1 y
b = j − 1, da F_{k−1} F_j − F_k F_{j−1} = (−1)^{j−1} F_{k−j}. Reducir mod F_k. ∎

*Chequeo con k = 6* (F_6 = 8, F_5 = 5): 5·2 = 10 ≡ 2 = F_3, 5·3 = 15 ≡ −1 = −F_2 y 5·5 = 25 ≡ 1 = F_1.
Verificado para k ≤ 40.

*Lectura geométrica.* Con n = F_k, el **estado áureo** es θ_v = 2π v F_{k−1}/F_k (la versión exacta,
racional, de v/φ). Los osciladores a distancia de índice F_j tienen diferencia de fase
2π F_{k−j}/F_k ≈ 2π φ^{−j}, que es chica. Es la razón por la que en el girasol los vecinos visuales
están a distancias de Fibonacci: son las espirales (parastiquias).

### Proposición A (los circulantes de Fibonacci son twisted states q = 1)

Sean n = F_k, S = {±F_j : j ∈ J} y el estado áureo θ_v = 2π v F_{k−1}/F_k. Entonces
(C_n(S), estado áureo) ≅ (C_n(S′), θ^{(1)}), con S′ = {±F_{k−j} : j ∈ J}.

*Prueba.* F_k y F_{k−1} son coprimos (dos Fibonacci consecutivos lo son), así que vale el Lema 2 con
q = F_{k−1}. Por el Lema 3, q·S = {±F_{k−j}}; los signos no importan porque S′ es simétrico. ∎

**Corolario.** Si el estado áureo es trampa estricta de C_{F_k}(S), entonces μ ≤ LP_twisted(F_k),
el LP de twisted states q = 1 (`src/twisted_lp.py`), que para todo m ≤ 120 da ≤ 0.683876.

*Prueba.* Por la Proposición A es un twisted state q = 1 en un circulante. La plantilla de m = F_k
clusters con densidades p_s ∈ {0, 1} (1 si s ∈ S′) es un punto factible del LP:
- sus autovalores de cluster son λ_m/n² > 0 (Lema 1);
- su splitting c = (1 + Σ p_s cos)/n es > H_vv/n > 0, porque H_vv = e_vᵀ H e_v > 0 (H ⪰ 0 con
  núcleo solo 1);
- su grado d = (1 + |S′|)/n coincide con μ.

El LP maximiza sobre un conjunto que contiene ese punto. ∎

**La variante 1 no puede superar el 0.6838 de Yoneda et al.**, y por lo tanto tampoco 11/16. Es un
resultado negativo demostrado.

### Proposición B (con n = F_k, el grafo geométrico áureo es exactamente un anillo)

Sean x_v = v/φ mod 1 (v = 0, …, n − 1, n = F_k, k ≥ 3) y el grafo que une u, v si su distancia
circular es ≤ r, con r = (ρ + ½)/n. Ese grafo es isomorfo a C_{n,ρ}, y el estado "fase = posición"
θ_v = 2π x_v está a menos de 2π·0.45/n del twisted state q = 1 de esa copia de C_{n,ρ}.

*Prueba.*
1. Por Binet, F_k/φ − F_{k−1} = (−1)^{k−1} φ^{−k}, así que ε := 1/φ − F_{k−1}/F_k cumple
   |ε| = φ^{−k}/F_k.
2. Entonces x_v = (v F_{k−1}/F_k + v ε) mod 1. El primer término es la "ranura" equiespaciada
   s_v/n, con s_v = v F_{k−1} mod F_k, que es una permutación de Z_n (Lema 2).
3. |v ε| < F_k |ε| = φ^{−k}, y también |(v − v′) ε| < φ^{−k}.
4. Como F_k = (φ^k − (−φ)^{−k})/√5, vale φ^{−k} < 1/(2F_k) = 1/(2n) para k ≥ 3 (k = 3: 0.236 < 0.25;
   k = 4: 0.146 < 0.167; la razón φ^{−k}·2F_k tiende a 2/√5 ≈ 0.894 < 1).
5. Entonces la distancia circular entre x_u y x_v difiere de |s_u − s_v|_circ/n en menos de 1/(2n).
   Con r = (ρ + ½)/n, u ~ v ⟺ |s_u − s_v|_circ ≤ ρ: en las etiquetas s es C_{n,ρ}.
6. θ_v = 2π s_v/n + 2π v ε, y como v ≤ n − 1, el segundo término cumple
   |2π v ε| ≤ 2π (n − 1) φ^{−k}/n. Por Binet, (n − 1) φ^{−k} = φ^{−k} F_k − φ^{−k}
   ≤ (1 + φ^{−2k})/√5 − φ^{−k}, que es < 0.45 para k ≥ 3 (vale 0.236 en k = 3, 0.36 en k = 5 y
   tiende a 1/√5 ≈ 0.447 desde abajo). Ojo: sin el −φ^{−k} no alcanza, porque φ^{−k} F_k pasa
   apenas de 0.45 en k impar (0.4508 en k = 5). ∎

**Consecuencia.** Si θ^{(1)} es trampa estricta de C_{n,ρ}, el flujo desde el estado áureo cae en esa
trampa (está a distancia O(1/n), dentro de la cuenca; lo verificamos numéricamente). Para n de
Fibonacci la variante 2 **es** la familia de anillos de Wiley–Strogatz–Girvan.

### Observación (Toeplitz)

Para cualquier n (Fibonacci o no), la distancia circular entre x_u y x_v es ‖(u − v)/φ‖ (distancia al
entero más cercano), que depende solo de u − v. Entonces la matriz de adyacencia del grafo geométrico
áureo es **Toeplitz** en el orden de índice v: es un "circulante sin la vuelta". Hay a lo sumo n − 1
radios distintos para barrer, y con n no Fibonacci lo que se rompe es justamente la vuelta (la
diferencia u − v no se reduce mod n). Por el teorema de las tres distancias (Steinhaus–Sós), los
huecos entre puntos consecutivos toman 2 o 3 valores, así que el grafo no es vértice-transitivo y el
estado "fase = posición" deja de ser un equilibrio exacto: hay que relajarlo.

## Resultados numéricos

### Variante 1: circulantes de Fibonacci (fuerza bruta sobre S, k ≤ 14)

| k | n = F_k | μ máx | distancias óptimas | LP twisted del mismo n |
|---|---|---|---|---|
| 5 | 5 | 0.6000 | {2} | 0.6584 |
| 6 | 8 | 0.6250 | {2, 3} | 0.6679 |
| 7 | 13 | 0.5385 | {2, 3, 5} | 0.6835 |
| 8 | 21 | 0.4286 | {2, 3, 5, 8} | 0.6762 |
| 10 | 55 | 0.2364 | {2, …, 21} | 0.6805 |
| 12 | 144 | 0.1181 | {2, …, 55} | 0.6838 |
| 14 | 377 | 0.0557 | {2, …, 144} | 0.6835 |

- **El óptimo es siempre "todas las distancias de Fibonacci salvo la 1"**: F_3, …, F_{k−2} (F_{k−1}
  ≡ −F_{k−2} da la misma distancia). Se entiende con la Proposición A: equivale al estado q = 1 en
  C_n({±1, ±2, ±3, ±5, …, ±F_{k−3}}). Ahí todas las
  diferencias de fase son ≤ F_{k−3}/F_k ≈ φ^{−3} ≈ 0.236 vuelta (< 90°), todos los acoples tienen
  cos > 0 y el Hessiano es un Laplaciano con pesos positivos: trivialmente estable. La distancia 1
  tiene diferencia de fase ≈ 1/φ vuelta (el propio ángulo áureo, 137.5°), con cos ≈ −0.74, y rompe
  la estabilidad.
- Es una familia **rala**: hay solo ~log_φ n distancias de Fibonacci; el óptimo tiene grado
  2(k − 4), así que μ = (2k − 7)/F_k → 0.

### Variante 2: grafo geométrico áureo (n = 8..200, todos los radios distintos)

Para cada n se barren todos los radios donde el grafo cambia, se relaja el estado áureo (L-BFGS y
Newton; ‖θ̇‖ < 10⁻¹² en todos los equilibrios), se clasifica, y se toma el mayor μ con mínimo estricto
y winding number ≠ 0. Se compara con el mejor anillo del mismo n, (2 k_max + 1)/n.

| n | μ áureo | μ mejor anillo |
|---|---|---|
| 21 (Fib) | 0.6190 | 0.6190 |
| 30 | 0.5000 | 0.6333 |
| 34 (Fib) | **0.6765** | 0.6765 |
| 50 | 0.6200 | 0.6600 |
| 55 (Fib) | 0.6727 | 0.6727 |
| 100 | 0.6300 | 0.6700 |
| 144 (Fib) | 0.6736 | 0.6736 |
| 150 | 0.6333 | 0.6733 |
| 199 (mejor no Fib) | 0.6734 | 0.6784 |
| 200 | 0.6350 | 0.6750 |

- **n Fibonacci (8, 13, …, 144): μ áureo = μ anillo exactamente**, como dice la Proposición B (y la
  trampa encontrada es la del anillo: el flujo desde el estado áureo cae en ella).
- **n no Fibonacci: nunca mejor que el anillo.** De 186 valores, 185 son peores y 1 empata (n = 68).
  La brecha media es 0.077 (0.045 para n ≥ 100). En n = 11 y 12 no hay ninguna trampa. Los huecos
  desiguales de las tres distancias **empeoran** la trampa.
- Mejor μ de la variante 2: 0.6765 (n = 34, Fibonacci); el mejor no Fibonacci es 0.6734 (n = 199).
- Detalle técnico: como μ no baja al crecer r, `mejor_radio` recorre los radios de mayor a menor y
  frena en la primera trampa (exacto y ~100 veces más rápido que barrer todos).

### Variante 3: plantillas en las fases áureas (m = 5..40)

m clusters en φ_a = 2π a/φ; con fases y tamaños fijos, el SDP (`sdp_interno`) da el mayor μ con
L ≻ 0 en 1^⊥ optimizando las densidades. Tres tipos de tamaños: iguales (1/m), "arcos" (cada cluster
pesa el arco del círculo más cercano a él que a los demás, como una celda de Voronoi) y geométricos
(∝ φ^{−a}).

| m | LP twisted (red equiespaciada) | áureo, iguales | áureo, arcos | áureo, geométricos |
|---|---|---|---|---|
| 8 (Fib) | 0.6679 | 0.6165 | 0.6480 | 0.3166 |
| 13 (Fib) | 0.6835 | 0.6551 | 0.6708 | 0.3187 |
| 14 | 0.6751 | 0.6308 | **0.6766** | 0.3186 |
| 15 | 0.6708 | 0.6121 | **0.6743** | 0.3185 |
| 21 (Fib) | 0.6762 | 0.6670 | 0.6746 | 0.3188 |
| 34 (Fib) | 0.6811 | 0.6771 | 0.6786 | 0.3188 |
| 40 | 0.6820 | 0.6577 | 0.6788 | (falla numérica) |

- El máximo en m ≤ 40 es **0.6788** (arcos, m = 40), por debajo del 0.6838 de twisted y lejos de
  11/16.
- Los pesos "arcos" son claramente los buenos: compensan los huecos desiguales (tres distancias), y
  con m creciente se acercan a una red equiespaciada.
- En m = 14 y 15 la red áurea con arcos le gana a la red equiespaciada **del mismo m**, porque esos m
  son malos para twisted, pero no al mejor twisted en general.
- Los pesos geométricos son malos (≈ 0.32): el primer cluster pesa ≈ 0.38 y los últimos casi nada
  (en m ≥ 39 pesan ~10⁻⁸ y el solver ya no converge).

## Conclusión

- La familia girasol **no da trampas más densas** que las conocidas. Las variantes 1 y 2 (con n
  Fibonacci) se reducen, por teoría de números, a twisted states q = 1 en circulantes, que ya están
  acotados por 0.6838. Lo que no se reduce (n no Fibonacci, plantillas) queda numéricamente por
  debajo.
- **Por qué no alcanza:** lo que hace especial al ángulo áureo es que reparte los puntos lo más
  parejo posible (es el número "más irracional"). Pero eso tiende a la distribución uniforme en el
  círculo, que es el mismo límite (grafón) de los anillos, μ → 0.681. Los récords (Canale, S7) hacen
  lo contrario: concentran los osciladores en **pocos clusters** muy desparejos.
- **Valor para la presentación:** es una familia con fórmulas cerradas y pruebas completas (Lemas 1–3,
  Proposiciones A y B), y responde con precisión "por qué no alcanza". La conexión entre las
  espirales del girasol (parastiquias) y el isomorfismo con q = 1 es linda para exponer.

## Abierto

- Variante 3 con **fases libres** partiendo de las áureas (optimizar también φ): probablemente
  converge a redes equiespaciadas o a geometrías conocidas.
- Estadística de cuencas: ¿qué fracción de arranques al azar cae en un estado no sincronizado en la
  variante 2 con n no Fibonacci, comparado con el anillo de μ parecido? Barato con `simular`.
- Usar la red áurea como **semilla** (vía E de `conclusiones_y_siguientes_pasos.md`): μ de borde con
  `semilla`.
