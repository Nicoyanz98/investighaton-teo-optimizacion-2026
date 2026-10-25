# De Canale y O'Keeffe a un grafo S7 explícito: la explicación completa

Este documento cuenta, desde el principio, cómo llegamos a un grafo **concreto, sin pesos y certificado**
con $\mu > 11/16$ que tiene un equilibrio estable no sincronizado. Arranca por el problema, sigue con las
dos construcciones de las que partimos (Canale 2022 y O'Keeffe 2026) y después recorre cada fase de
nuestra construcción, explicando qué se hace y **por qué**. Termina con la versión simétrica, que es la
más simple: un grafo de $n = 299\,989$ vértices descrito por cuatro enteros.

Notebooks: `notebooks/Grafo_explicito_S7.ipynb` (construcción general) y
`notebooks/Grafo_simetrico_S7.ipynb` (versión simétrica). Papers: `papers/2209.06362v3.pdf` (Canale) y
`papers/2610.21271v1.pdf` (O'Keeffe).

## Índice

1. [El problema](#1-el-problema)
2. [Estados de clusters y plantillas](#2-estados-de-clusters-y-plantillas)
3. [El punto de partida: C4](#3-el-punto-de-partida-c4)
4. [Canale: 11/16 con grafos explícitos](#4-canale-1116-con-grafos-explícitos)
5. [O'Keeffe: S7 y 0.6929, pero sin grafo](#5-okeeffe-s7-y-06929-pero-sin-grafo)
6. [Nuestro objetivo y el plan](#6-nuestro-objetivo-y-el-plan)
7. [Fase 0: elegir las densidades (el SDP)](#7-fase-0-elegir-las-densidades-el-sdp)
8. [Fase 1: pasar a enteros sin perder el equilibrio](#8-fase-1-pasar-a-enteros-sin-perder-el-equilibrio)
9. [Fases 2 y 3: el grafo y el certificado](#9-fases-2-y-3-el-grafo-y-el-certificado)
10. [Las densidades "casi 1" y la simetría de S7](#10-las-densidades-casi-1-y-la-simetría-de-s7)
11. [El grafo simétrico](#11-el-grafo-simétrico)
12. [Comparación final y qué queda abierto](#12-comparación-final-y-qué-queda-abierto)
13. [Glosario](#13-glosario)

---

## 1. El problema

### El modelo

En un grafo $G$ con $n$ vértices y matriz de adyacencia $W$ ($W_{uv} = 1$ si $u$ y $v$ están unidos), cada
vértice es un oscilador con una fase $\theta_u$. El **modelo de Kuramoto homogéneo** es

$$\dot\theta_u = \sum_v W_{uv}\,\sin(\theta_v - \theta_u).$$

Cada vecino "tira" de $\theta_u$ hacia su propia fase. Es el flujo gradiente de la energía

$$E(\theta) = \sum_{uv \in E} \big(1 - \cos(\theta_u - \theta_v)\big),$$

así que la dinámica siempre baja la energía y termina en un equilibrio.

- **Sincronización:** todas las fases iguales. Es el mínimo global ($E = 0$).
- **Equilibrio:** $\dot\theta = 0$.
- **Mínimo local estricto:** el Hessiano
  $$H_{uv} = -W_{uv}\cos(\theta_u - \theta_v)\ (u \ne v), \qquad H_{uu} = \sum_v W_{uv}\cos(\theta_u - \theta_v)$$
  es semidefinido positivo y su único cero es la rotación global (sumar la misma constante a todas las
  fases no cambia nada).

Un mínimo estricto que no es la sincronización es una **trampa**: si la dinámica arranca cerca, se queda
ahí y el grafo no sincroniza.

### La pregunta

Los grafos densos sincronizan: en el grafo completo, la sincronización es el único estado estable. ¿Qué
tan denso alcanza? La densidad se mide con la **conectividad**

$$\mu = \frac{\delta + 1}{n},$$

donde $\delta$ es el grado mínimo. El grafo completo tiene $\mu = 1$. (O'Keeffe usa $\delta/(n-1)$; para
$n$ grande es lo mismo. En el código usamos $(\delta+1)/n$, porque el "+1" representa al propio vértice y
hace que el blow-up de la sección 2 no cambie $\mu$.)

La **conectividad crítica** $\mu_c$ es el umbral: todo grafo con $\mu > \mu_c$ sincroniza siempre. Para
probar $\mu_c > \mu_0$ alcanza con **un** grafo con $\mu > \mu_0$ que tenga una trampa.

### Lo que se sabía

| cota inferior (grafos con trampa) | año | cota superior (pruebas de sincronización) | año |
|---|---|---|---|
| 0.6828 (twinning-plus-adding) | 2020 | 0.9395 (Taylor) | 2012 |
| 0.6838 (circulantes óptimos, Yoneda et al.) | 2021 | 0.7889 | 2020 |
| **11/16 = 0.6875 (Canale)** | 2022 | 3/4 (Kassabov, Strogatz, Townsend) | 2021 |
| **0.6929 (O'Keeffe)** | 2026 | $< 3/4$ (Lepsveridze, Zhang) | 2026 |

(Tabla I de O'Keeffe.) Este trabajo se ubica del lado de las cotas inferiores: construir grafos con
trampas lo más densos posible.

---

## 2. Estados de clusters y plantillas

### La idea

Los récords no usan estados "desordenados": los vértices se agrupan en pocos **clusters**, y todos los
vértices de un cluster tienen la misma fase. Esto viene de la idea del "stackeo" (`notes/idea.md`): si
tomamos un grafo chico $H$ con una trampa y reemplazamos cada vértice por un clique de muchas copias,
uniendo todas las copias de $u$ con todas las de $v$ cuando $uv$ es arista (el **blow-up** $H[K_m]$), la
trampa se conserva y $\mu$ no cambia.

Generalizando: los clusters pueden tener tamaños distintos, y entre dos clusters no hace falta unir
todo con todo, sino una **fracción** de los pares. Eso da el marco de las plantillas.

### Plantilla

Una **plantilla** con $m$ clusters es una terna $(x, \varphi, P)$:
- $x_a$: fracción de los vértices que hay en el cluster $a$ ($\sum_a x_a = 1$);
- $\varphi_a$: la fase del cluster $a$;
- $P_{ab} \in [0,1]$: la **densidad** entre los clusters $a$ y $b$. Cada vértice de $a$ está unido a
  $P_{ab}\, n x_b$ vértices de $b$. $P_{aa} = 1$: cada cluster es un clique.

Si **cada** vértice de $a$ tiene exactamente $P_{ab}\, n x_b$ vecinos en $b$ (la sección 9 muestra cómo
lograrlo), todas las cantidades del grafo de $n$ vértices se escriben con la plantilla, sin importar $n$:

| condición | fórmula | qué significa |
|---|---|---|
| grado | $d_a = \sum_b P_{ab} x_b$, y $\mu = \min_a d_a$ | un vértice de $a$ tiene $\approx n d_a$ vecinos |
| equilibrio | $F_a = \sum_b P_{ab} x_b \sin(\varphi_a - \varphi_b) = 0$ | la fuerza neta sobre cada cluster es 0 |
| estabilidad de un vértice suelto | $c_a = \sum_b P_{ab} x_b \cos(\varphi_a - \varphi_b) > 0$ | si un vértice se separa de su cluster, lo traen de vuelta |
| estabilidad de los modos de cluster | $L(y) = \sum_{a<b} x_a x_b P_{ab}\cos(\varphi_a - \varphi_b)(y_a - y_b)^2 > 0$ | mover clusters enteros (cada uno $y_a$) sube la energía |

La última condición se pide para todo $y$ que no sea constante (una rotación global). Una plantilla que
cumple las tres condiciones de estabilidad es **estricta**, y un grafo construido a partir de ella tiene
una trampa.

La ventaja es enorme: decidir si un grafo de 30 millones de vértices tiene una trampa se reduce a cuentas
con matrices de $m \times m$ (acá, $7 \times 7$).

---

## 3. El punto de partida: C4

Cuatro clusters iguales ($x_a = 1/4$) en las fases $0, \pi/2, \pi, 3\pi/2$. Cada uno está unido
completo con sus dos vecinos y sin links con el opuesto (a $\pi$ de distancia).

- **Grado:** cada cluster ve el suyo y los dos vecinos: $d = 3/4$, así que $\mu = 3/4$.
- **Equilibrio:** sí, por simetría.
- **Estabilidad:** los vecinos están a exactamente 90°, y $\cos(\pi/2) = 0$. Entonces $L \equiv 0$: los
  modos de cluster son **neutros**. C4 está "en el filo": ni estable ni inestable a primer orden.
  Kassabov, Strogatz y Townsend lo estudiaron, y Lepsveridze y Zhang probaron que es inestable
  (no lineal), lo que da $\mu_c < 3/4$.

C4 es la referencia de todo lo que sigue: tanto Canale como O'Keeffe lo **deforman** para que deje de ser
neutro y pase a ser estable, pagando con algo de $\mu$.

---

## 4. Canale: 11/16 con grafos explícitos

Canale (arXiv 2209.06362, 2022) dio los primeros grafos explícitos con trampa y densidad cercana a 11/16.
Su construcción tiene dos partes, y las dos las reutilizamos.

### 4.1 El grafo pesado $G_p$

Ocho vértices ($\mathbb Z_8$), pensados como cuatro pares: cada vértice de C4 se parte en dos, "$-$" y
"$+$", corridos $\mp\beta/2$ respecto de la fase del sector:

$$\theta_{2i+s} = i\,\frac{\pi}{2} + s\,\frac{\beta}{2}, \qquad i \in \{0,1,2,3\},\ s = \pm 1.$$

Las aristas tienen cuatro tipos, con pesos $p = (a, b, c, d)$ según el ángulo que separa a sus extremos
(geometría de su Figura 2 y del Lema 1; los conjuntos $E_t$ escritos en el paper tienen erratas, ver
`src/plantillas.py: plantilla_canale`):
- $c$: entre las dos mitades del mismo sector (ángulo $\beta$);
- $a$: de la mitad "$+$" de un sector a la "$-$" del siguiente (ángulo $\alpha = \pi/2 - \beta$, las que
  quedan **más cerca** de 90° por dentro);
- $b$: entre mitades del mismo signo de sectores vecinos (ángulo exacto $\pi/2$, no aportan al
  equilibrio ni a la estabilidad);
- $d$: de la "$-$" de un sector a la "$+$" del siguiente (ángulo $\alpha + 2\beta$, **más** de 90°).

**Equilibrio:** las aristas $b$ se compensan, y queda $a\sin\alpha = c\sin\beta + d\sin(\alpha+2\beta)$,
que con $\alpha + \beta = \pi/2$ es $(a - d)\tan\alpha = c$.

**Lema 1 (estabilidad):** el equilibrio es linealmente estable si $c^2 > 2ad$.

### 4.2 El k-spinning: de pesos a un grafo simple

Un grafo pesado no sirve directamente: necesitamos un grafo simple. El **k-spinning** $S_k(G)$ reemplaza
cada vértice $u$ por $k$ copias $(u, 0), \dots, (u, k-1)$, formando un clique, y une

$$(u, i) \sim (v, j) \quad\text{si}\quad j \in \{i, i+1, \dots, i + w_{uv} - 1\} \pmod k.$$

Cada copia de $u$ se une a un **intervalo** de $w_{uv}$ copias consecutivas de $v$. Entonces cada
vértice de $u$ tiene exactamente $w_{uv}$ vecinos en $v$: el peso se convierte en un grado exacto. Las
Proposiciones 1 y 2 de Canale muestran que el equilibrio y su estabilidad pasan del grafo pesado al
k-spinning.

Esto es un **blow-up circulante**: cada cluster es una copia de $\mathbb Z_k$ y los links entre clusters son
"corrimientos" en un intervalo. Es exactamente la herramienta que usamos en la Fase 2.

### 4.3 Por qué 11/16

Con $n = 8k$ vértices, el grafo es regular con grado $\delta = a + 2b + c + d + (k-1)$. Canale elige

$$a = b = c = k, \qquad d = k/2 - 1,$$

que cumple $c^2 = k^2 > 2k(k/2 - 1) = 2ad$. Entonces

$$\mu = \frac{\delta + 1}{8k} = \frac{5.5k - 1}{8k} \;\longrightarrow\; \frac{11}{16} \text{ por debajo}.$$

Cada cluster ve 5.5 de los 8 clusters. Con $k = 34$ ($n = 272$) ya supera 0.6838, y con $k = 1250$
($n = 10^4$) llega a 0.6874.

**El límite:** la condición de estabilidad $c^2 > 2ad$ obliga a $d < k/2$, y eso frena la densidad en
11/16. Además, todos los links son 0 o 1 "en bloque" (completos entre partes de los clusters). O'Keeffe
cuenta que sus búsquedas numéricas con margen de estabilidad y links 0/1 vuelven siempre al estado de
Canale, lo que hizo pensar que 11/16 podía ser el valor exacto de $\mu_c$.

---

## 5. O'Keeffe: S7 y 0.6929, pero sin grafo

O'Keeffe (arXiv 2610.21271, octubre de 2026) usa el marco de plantillas con **densidades intermedias** y
encuentra estados que superan 11/16.

### 5.1 La familia S7

Se deforma C4 así:
- los sectores **A** (fase 0), **B** ($\pi/2$) y **C** ($\pi$) tienen tamaño $2a$ cada uno y se parten en
  dos mitades, $m$ ("menos") y $p$ ("más"), corridas $h s$ respecto de la fase del sector;
- el sector **W** ($3\pi/2$) tiene tamaño $1 - 6a$ y queda entero;
- son 7 clusters, en el orden $(A_m, A_p, B_m, B_p, C_m, C_p, W)$, con desplazamientos
  $s = (-1, 1, -w, w, -1, 1, 0)$.

$h$ es la **apertura**: cuánto se separan las mitades.

### 5.2 El patrón límite (Tabla II)

Cuando $h \to 0$, las densidades son:

| | Am | Ap | Bm | Bp | Cm | Cp | W |
|---|---|---|---|---|---|---|---|
| **Am** | 1 | 1 | 1 | $q$ | 0 | 0 | 1 |
| **Ap** | 1 | 1 | $q$ | 1 | 0 | 0 | 1 |
| **Bm** | 1 | $q$ | 1 | 1 | 1 | $q$ | 0 |
| **Bp** | $q$ | 1 | 1 | 1 | $q$ | 1 | 0 |
| **Cm** | 0 | 0 | 1 | $q$ | 1 | 1 | 1 |
| **Cp** | 0 | 0 | $q$ | 1 | 1 | 1 | 1 |
| **W** | 1 | 1 | 0 | 0 | 1 | 1 | 1 |

Todo es 1 (completo) o 0 (vacío), salvo **cuatro pares cruzados** con densidad $q$ (8 entradas de la
matriz, porque es simétrica): Am–Bp, Ap–Bm, Bm–Cp y Bp–Cm. A–C y B–W, los sectores opuestos, no tienen
links.

**Grados límite.** Las mitades tienen tamaño $a$ y W tiene $1 - 6a$:
- un vértice de Am ve su sector ($2a$), Bm completo y Bp con densidad $q$ ($(1+q)a$), nada de C, y W
  completo: $d_{A} = 1 - (3 - q)a$;
- un vértice de Bm ve su sector, y en A y en C una mitad completa y otra con $q$; nada de W:
  $d_B = (4 + 2q)a$;
- W ve A, C y a sí mismo: $d_W = 1 - 2a$.

Se elige $a$ para que A y B tengan el mismo grado: $a = 1/(7+q)$. Entonces

$$\mu_0(q) = \frac{4 + 2q}{7 + q},$$

que crece con $q$. Cuanto más grande $q$, mejor, pero hay un límite.

### 5.3 La esquina

Para $h \to 0$, los modos de cluster de S7 tienen autovalores de orden $h^2$. Hay un punto $(q^*, w^*)$,
"la esquina", en el que dos de esos modos (los "dipolos") son exactamente neutros. Es la raíz de las dos
condiciones del Apéndice A:

$$f = -2q^2w^2 - 4q^2 w - 3q^2 - 2q + 4w + 1 = 0,\qquad e = -q^2w^2 + q^2 - 2qw^2 + 6q - w^2 + 1 = 0,$$

con $q^* = 0.650569\ldots$, $w^* = 1.398278\ldots$ y

$$\mu_0 = \mu_0(q^*) = 0.6929077063\ldots$$

Para $q$ un poco menor que $q^*$ la plantilla es estricta. Ese es el valor de O'Keeffe.

### 5.4 A $h$ finito: la corrección $p'$

Con $h > 0$, las densidades del patrón límite ya no dan un equilibrio. O'Keeffe toma $q = q^* - t$ y

$$P(h) = p + h\,p' + \Delta(h),$$

donde $p$ es el patrón límite, $p'$ una corrección explícita de orden $h$ y $\Delta = O(h^2)$ un ajuste
que hace el equilibrio exacto. Lo importante de $p'$ (Apéndice A):

| par | $p$ | $p'$ |
|---|---|---|
| Am–Bp, Bm–Cp | $q$ | $-\kappa = -1.9$ |
| Ap–Bm, Bp–Cm | $q$ | $A_2 - \kappa \approx -0.82$ |
| Ap–Bp, Bm–Cm | 1 | $-1.9$ |
| Ap–W, Cm–W | 1 | $\approx -2.86$ |
| Am–Bm, Bp–Cp | 1 | $A_1 - \kappa \approx -0.18$ |
| Am–W, Cp–W | 1 | $\approx -0.05$ |

O'Keeffe elige $\kappa = 19/10$ "para que $p' < 0$ en todos los pares vecinos con $p = 1$, así esas
densidades quedan por debajo de 1". O sea: **a $h$ finito, todos los pares vecinos bajan de 1**, no solo
los cuatro cruzados. Con su $h = 10^{-6}$ esos pares valen 0.999998, y a simple vista parecen 1.

Hay dos parámetros chicos:
- $t$ = distancia a la esquina. Alejarse vuelve estrictamente estables los dipolos, con un gap
  proporcional a $t$, pero baja $\mu$.
- $h$ = apertura. El modo más débil queda a distancia $\sim h^2(t + h)$ de la neutralidad.

### 5.5 Los resultados y el problema del tamaño

- **Proposición 1:** con $t = 10^{-5}$ y $h = 10^{-6}$, un miembro de la familia certificado con
  aritmética de intervalos: $\mu = 0.6929050\ldots$, con el autovalor más chico de los modos de cluster
  en $6.2\cdot10^{-19}$.
- **Teorema 1:** haciendo $t \to 0$, $\mu_c \ge \mu_0 = 0.6929077\ldots$

Pero el grafo **nunca se escribe**:
1. Para tener datos exactos, racionaliza la plantilla con denominador $D \approx 3\cdot10^{42}$. Hace
   falta tanta precisión porque el margen es $6\cdot10^{-19}$: un redondeo más grueso lo destruye.
2. El blow-up (Lema 2) usa conjuntos de corrimientos **aleatorios** y necesita $n = D^2\ell$ vértices.

Resultado: $n > 9\cdot10^{84}$. Es una prueba de existencia.

---

## 6. Nuestro objetivo y el plan

**Objetivo:** un grafo **escrito, sin pesos**, con $\mu > 11/16$ y una trampa **certificada** (no solo
numérica), con $n$ razonable.

El $n$ de O'Keeffe tiene dos causas, y atacamos cada una:

| causa | solución |
|---|---|
| blow-up aleatorio, $n = D^2\ell$ | blow-up **circulante** como el k-spinning de Canale, con grados exactos y $n = 23g$ para cualquier $g$ |
| margen diminuto ($6\cdot10^{-19}$), que obliga a $D$ enorme | resignar un poco de $\mu$ a cambio de margen (Fase 0) y redondear con cuidado (Fase 1) |

El camino, en fases:

```
Plantilla (números reales)  →  Datos enteros (grados d_ab)  →  Grafo  →  Certificado
x, φ, P con 7 clusters          vecinos de cada vértice        blow-up    intervalos
                                de a en b                      circulante
      FASE 0                          FASE 1                   FASE 2     FASE 3
  "¿qué P elegir?"           "¿cómo redondear sin
                               perder el equilibrio?"
```

---

## 7. Fase 0: elegir las densidades (el SDP)

### 7.1 Qué se busca

No usamos la fórmula $p + hp'$ de O'Keeffe: a $h$ finito pierde mucho $\mu$ (con $t = 10^{-3}$ y
$h = 0.01$ da 0.6830, por debajo de 11/16). En cambio, fijamos $h$ y buscamos **las mejores densidades
posibles** para un margen de estabilidad dado.

Con los tamaños $x$ y las fases $\varphi$ fijos, se busca la matriz $P$ que maximiza $\mu = \min_a d_a$ y
cumple:
1. densidad válida: $0 \le P \le 1$, simétrica, $P_{aa} = 1$, ceros en A–C y B–W;
2. equilibrio: $F(P) = 0$;
3. vértice suelto: $c(P) \ge 10^{-3}$;
4. modos de cluster con margen: $Q^\top L(P)\,Q / h^2 \succeq \varepsilon I$, donde $Q$ es una base de los
   vectores que suman 0 (para descartar la rotación global) y "$\succeq$" significa "semidefinida
   positiva". Se divide por $h^2$ porque esos modos son de orden $h^2$.

### 7.2 Por qué es un SDP, y por qué eso importa

Con $x$ y $\varphi$ fijos, los senos y cosenos son **números**, así que $F$, $c$, los grados y la matriz
$L$ son **lineales** en $P$. Un problema con objetivo lineal, restricciones lineales y una restricción
"esta matriz, que depende linealmente de las variables, es semidefinida positiva" se llama **programa
semidefinido (SDP)**.

Lo importante es que es **convexo**: no tiene óptimos locales falsos, y un solver (CLARABEL, vía
`cvxpy`) encuentra el óptimo global en milisegundos. La condición de estabilidad, que a primera vista es
"$L(y) > 0$ para infinitos $y$", queda como una sola restricción sobre una matriz.

### 7.3 El lazo de afuera

Las fases dependen de los desplazamientos $s$, y con $s$ como variable el problema deja de ser convexo.
Entonces:
- **adentro**, para cada $s$, el SDP da el mejor $P$ y su $\mu$;
- **afuera**, Nelder–Mead (un optimizador sin derivadas) prueba distintos $s$ buscando el mayor $\mu$,
  arrancando del ansatz de O'Keeffe.

Los tamaños se fijan racionales de entrada, $x = (3, 3, 3, 3, 3, 3, 5)/23$, para no tener que
redondearlos después. La razón $5/3 \approx 1.67$ está cerca del óptimo $(1 - 6a)/a = 1 + q^* \approx 1.65$.

### 7.4 El margen $\varepsilon$ en lugar de $t$

En O'Keeffe la estabilidad se controla indirectamente con $t$. Acá se pide directamente
$\lambda_{\min}(L)/h^2 \ge \varepsilon$ y el SDP elige **todas** las densidades. Por eso no hay un único $q$
ni un $t$. Con la escala de O'Keeffe (modo más débil $\sim h^2(t + h)$), si $h \gg \varepsilon$ la propia
apertura aporta el margen y no hace falta alejarse de la esquina.

`scripts/curva_s7.py` mide cuánto $\mu$ cuesta cada $\varepsilon$. La plantilla elegida tiene $h = 0.03$,
$\varepsilon = 10^{-3}$ y $\mu = 0.6885$: entre $11/16 = 0.6875$ y $\mu_0 = 0.6929$.

---

## 8. Fase 1: pasar a enteros sin perder el equilibrio

### 8.1 Datos enteros

Un grafo concreto se describe con:
- tamaños $n_a = k_a g$ (seis mitades de $3g$ vértices y W de $5g$, así que $n = 23g$);
- grados $d_{ab}$: cuántos vecinos tiene **cada** vértice de $a$ en $b$.

Restricciones: contar los links entre $a$ y $b$ desde los dos lados da $n_a d_{ab} = n_b d_{ba}$; y cada
cluster es un clique, $d_{aa} = n_a - 1$. Hay **un entero por par** $a < b$. De los 21 pares, 6 son
opuestos (vacíos) y los que valen 0 o 1 en la plantilla quedan fijos. En la plantilla $h = 0.03$ quedan
**9 pares libres**; en las plantillas con $h \le 0.02$ el SDP no deja ningún vecino en 1 y son 15.

**Redondeo ingenuo:** cada $d_{ab}$ es el entero permitido más cercano a $P_{ab} n_b$. Cada densidad
cambia en $\sim 1/n$.

### 8.2 Por qué el redondeo ingenuo no alcanza: el pliegue

El Jacobiano $J$ de las ecuaciones de equilibrio tiene **tres valores singulares diminutos**
($\sigma \approx 6\cdot10^{-4}$, $5\cdot10^{-6}$, $2\cdot10^{-6}$): tres direcciones "blandas" en las que el
equilibrio casi no está sostenido. Son la herencia de los modos neutros de C4.

Imagen: el equilibrio es una bolita en un pozo muy poco profundo. Si inclinás un poco la mesa (el error de
redondeo), el pozo desaparece. En fórmulas, en cada dirección blanda el equilibrio es raíz de

$$\sigma\,y + \tfrac{c}{2}\,y^2 + \rho = 0,$$

donde $y$ es cuánto se mueven las fases en esa dirección y $\rho$ es el error del redondeo. Es la forma
normal de un **pliegue** (silla-nodo):
- hay solución solo si $\rho \le \sigma^2/(2|c|)$;
- la estabilidad de esa solución es $\sqrt{\sigma^2 - 2c\rho}$: baja a 0 al llegar al pliegue;
- si el redondeo pasa el pliegue, el equilibrio **desaparece** (Newton no converge o las fases saltan a
  otro estado).

Con el redondeo ingenuo $\rho \sim 1/n$, así que hace falta $n \gtrsim 2|c|/\sigma^2$. Medido para esta
plantilla: el ingenuo necesita $n \approx 10^9$.

### 8.3 Qué errores absorben las fases y cuáles no

Las fases pueden moverse para compensar el error. Separando el error en las direcciones de la SVD de $J$:
- en las direcciones **rígidas** ($\sigma \gtrsim 0.1$) la tolerancia es enorme frente a $1/n$: las
  fases absorben el error sin problemas;
- en las **blandas** la tolerancia es diminuta: las fases no alcanzan, y el error ahí hay que eliminarlo
  eligiendo bien los grados.

### 8.4 Lyapunov–Schmidt: medir solo lo que importa

Para elegir los grados hace falta medir "cuánto error queda en las direcciones blandas". La reducción de
**Lyapunov–Schmidt** es separar lo fácil de lo difícil, con $J = U S V^\top$:
1. se deja que las fases se muevan **solo en las direcciones rígidas** ($\varphi = \varphi^* + V_r z$) hasta que
   las ecuaciones rígidas ($U_r^\top F = 0$) se cumplen exactamente: Newton común, que ahí converge;
2. lo que queda en las direcciones blandas es el **residuo blando** $g = U_b^\top F$: 3 números.

$g$ depende solo de los grados, está bien definido aunque no haya equilibrio, y es casi lineal en los
grados. El objetivo queda: **elegir los grados enteros para que cada $|g_k|$ quede bien por debajo de su
tolerancia**.

### 8.5 Newton entero

Cerca del redondeo ingenuo, $g(d + u) \approx g(d) + G\,u$, con $G = \partial g/\partial d$ (3 filas, una
columna por grado libre). Newton común pediría $G\,u = -g$: con más incógnitas que ecuaciones hay
infinitas soluciones reales, y la de menor norma es $u_0$.

Pero $u$ tiene que ser **entero**, y redondear $u_0$ coordenada por coordenada no sirve: cada coordenada
se mueve hasta $\pm 1/2$ y el error vuelve a ser del tamaño del redondeo ingenuo.

**La idea clave: elegir los enteros en conjunto.** Ejemplo: buscar enteros con $u_1 + \sqrt2\,u_2 \approx
-0.3$. Con $|u_i| \le 1$ lo mejor es $(1, -1)$, con error 0.11. Con $|u_i| \le 10$ aparece $(11, -8)$:
$11 - 11.314 = -0.314$, error 0.014. Moviendo los enteros más, hay muchas más combinaciones y alguna cae
muy cerca. Hay un compromiso: $u$ grande cancela mejor, pero rompe la linealización. Por eso se prueban
varios radios (3, 10, 30, 100), se repite el paso unas pocas veces y la decisión final la toma la
evaluación exacta.

Detalles del paso:
- cada componente de $g$ se mide en unidades de su tolerancia (pesos $1/\sigma_k^2$), porque las tres
  tolerancias difieren en varios órdenes de magnitud;
- la derivada $G$ se calcula por diferencias finitas en alta precisión (60 dígitos), porque adentro de
  $g$ hay un Newton.

### 8.6 El reticulado: encontrar el entero bueno

Encontrar "el $u$ entero cerca de $u_0$ con $G(u - u_0) \approx 0$" es un problema de **vector más cercano
en un reticulado**. Un **reticulado** es el conjunto de combinaciones con coeficientes enteros de unos
vectores base, como $\mathbb Z^2$ pero posiblemente inclinado y estirado.

**Encaje de Kannan: juntar las dos condiciones en una distancia.** A cada entero $u$ se le asigna un vector
más largo,

$$v(u) = \big(\,u/r,\ \ \text{peso}\cdot G u\,\big),$$

y al objetivo $t = v(u_0)$. Entonces

$$|v(u) - t|^2 = \underbrace{|u - u_0|^2/r^2}_{\text{no alejarse}} + \underbrace{\text{peso}^2\,|G(u - u_0)|^2}_{\text{cancelar el residuo}}.$$

Un $u$ con esa distancia chica cumple las dos cosas. Los $v(u)$ forman un reticulado (sus vectores base son
las filas de la matriz `B` del código), así que el problema es "el punto del reticulado más cercano a
$t$". (En la literatura, "encaje de Kannan" designa estrictamente una variante que agrega $t$ a la base y
busca el vector más corto. Lo que usamos es la idea de fondo: un reticulado aumentado con pesos.)

**LLL: una buena base.** Un mismo reticulado tiene muchas bases, y con una mala base redondear da
cualquier cosa. Por ejemplo, $(1,0)$ y $(100,1)$ generan $\mathbb Z^2$. En esa base, el punto $(0.4, 0.4)$
es $-39.6\,b_1 + 0.4\,b_2$; al redondear los coeficientes queda $(-40, 0)$, a distancia 40. Con la base
$(1,0), (0,1)$ el redondeo da $(0,0)$. **LLL** (Lenstra–Lenstra–Lovász) transforma una base en otra del
mismo reticulado con vectores cortos y casi perpendiculares, restando múltiplos enteros e intercambiando
vectores.

**Babai: redondear bien.** Con la base reducida, el **plano más cercano de Babai** ortogonaliza
(Gram–Schmidt) y redondea el coeficiente de cada dirección, de la última a la primera, restando lo ya
elegido. No garantiza el óptimo exacto, pero con una base LLL queda cerca.

**Resultado:** el reticulado baja el $n$ necesario de $10^9$ a $3\cdot10^7$, cambiando unos pocos grados en
unas pocas unidades (de un total de $\sim10^7$).

---

## 9. Fases 2 y 3: el grafo y el certificado

### 9.1 Fase 2: blow-up circulante

Falta un grafo en el que **cada** vértice de $a$ tenga **exactamente** $d_{ab}$ vecinos en $b$. Es el
k-spinning de Canale generalizado a clusters de tamaños distintos:
- el cluster $a$ tiene $k_a$ bloques, y cada bloque es una copia de $\mathbb Z_g$: vértices $(a, i, v)$;
- dentro de un cluster: clique;
- entre clusters: $(a, i, v) \sim (b, j, v + s \bmod g)$ para $s \in \{0, \dots, t_{ij} - 1\}$, un intervalo
  de $t_{ij}$ vértices consecutivos;
- la tabla $t$ ($k_a \times k_b$) tiene filas que suman $d_{ab}$ y columnas que suman $d_{ba}$ (un problema de
  transporte), con entradas $\text{base}$ o $\text{base} + 1$ repartidas en forma cíclica.

Como todos los vértices de un cluster ven exactamente lo mismo, las fases de los clusters son un
**equilibrio exacto** del grafo. A diferencia del blow-up aleatorio de O'Keeffe, $g$ no tiene que ser
múltiplo de nada: el grafo tiene $n = 23g$ vértices con $g$ libre. Lo que se publica ($k$, $g$, grados,
tablas, fases) ocupa unos pocos KB.

### 9.2 Fase 3: el certificado

El Hessiano del grafo conmuta con el corrimiento $v \to v + 1$ en todos los bloques, así que se parte en
$g$ bloques chicos, uno por frecuencia de Fourier. Con eso:
1. **Modos de cluster (frecuencia 0, constantes por cluster):** se reducen a la plantilla de $7\times7$ con
   los datos enteros exactos ($x_a = n_a/n$, $P_{ab} = d_{ab}/n_b$, como fracciones). Ahí:
   - **Krawczyk** (aritmética de intervalos) prueba que existe un único equilibrio dentro de una caja de
     fases de radio $10^{-40}$;
   - en toda la caja, $c > 0$ y la matriz de $L$ admite **Cholesky por intervalos** con pivotes
     positivos, así que $L > 0$ salvo la rotación.
2. **El resto de los modos:** márgenes de **Gershgorin** positivos, evaluados en intervalos sobre la caja.
3. **$\mu = (\delta+1)/n > 11/16$** en aritmética racional exacta.

### 9.3 Resultados de la construcción general

| grafo | $n$ | $\mu$ |
|---|---|---|
| reticulado, $h = 0.03$ | 30 000 004 | 10327019/15000002 ≈ 0.688468 |
| camino B (sub-piezas, `notes/blowup_explicito.md`) | 1 500 014 | 1032699/1500014 ≈ 0.688463 |

Además, una validación "ciega" (el Hessiano del grafo completo en float64, sin usar la teoría del
certificado) y simulaciones de Kuramoto lo confirman. Las simulaciones muestran también que la cuenca de
la trampa es diminuta (del orden de $10^{-4}$ rad en las direcciones lentas): desde fases al azar el grafo
sincroniza.

---

## 10. Las densidades "casi 1" y la simetría de S7

### 10.1 ¿Por qué hay pares cerca de 1 que no son 1?

En la plantilla del SDP, además de los cuatro cruzados, hay pares que en el patrón límite valen 1 y acá
valen 0.97–0.99. **No es un error: O'Keeffe hace lo mismo** (sección 5.4: $p' < 0$ en todos los pares
vecinos). A $h$ finito, el equilibrio necesita corregir esas densidades a primer orden, y un par en 1 solo
puede bajar.

Y hay una coincidencia linda: los pares que el SDP deja en 1 (o casi) son exactamente los que O'Keeffe
adelgaza menos ($p' = -0.18$ y $-0.05$), y los que el SDP baja son los que él baja más ($-1.9$, $-2.86$).
Su $\kappa = 1.9$ es una elección cómoda, no óptima; el SDP adelgaza solo lo necesario, porque adelgazar
cuesta $\mu$.

### 10.2 La simetría de reflejo

La simetría de S7 es la reflexión $\varphi \mapsto \pi - \varphi$, un espejo que pasa por B y por W:

$$A_m \leftrightarrow C_p, \qquad A_p \leftrightarrow C_m, \qquad B_m \leftrightarrow B_p, \qquad W \leftrightarrow W.$$

(Am está en $-h$ y va a $\pi + h$, que es donde está Cp.) Para que el estado sea simétrico, los
desplazamientos cumplen $s_{Cp} = -s_{Am}$, $s_{Cm} = -s_{Ap}$, $s_{Bp} = -s_{Bm}$, y las densidades
$P_{ab} = P_{R(a)R(b)}$.

### 10.3 Por qué los cuatro pares cruzados no pueden ser iguales

En el límite valen todos $q$, pero con $h > 0$ no son geométricamente iguales. Bp está en
$\pi/2 + wh$, corrido hacia C:
- Am queda a $\pi/2 + (1+w)h$ de Bp, **más de 90°**: un par **lejano**, con $\cos < 0$, cuyos links
  **desestabilizan**;
- Cm queda a $\pi/2 - (1+w)h$, **menos de 90°**: un par **cercano**, con $\cos > 0$, cuyos links
  **estabilizan**.

Con los $s$ de O'Keeffe y $h = 0.03$: los lejanos están a 94.12° ($\cos = -0.072$) y los cercanos a 85.88°
($\cos = +0.072$). Conviene que los lejanos sean más finos que los cercanos. El reflejo empareja lejano con
lejano (Am–Bp con Cp–Bm) y cercano con cercano (Ap–Bm con Cm–Bp): la simetría permite **dos** valores de
$q$, no uno. O'Keeffe también los separa: $q - 1.9h$ contra $q - 0.82h$.

### 10.4 ¿La simetría empeora la solución?

Medido con el SDP ($h = 0.03$, $\varepsilon = 10^{-3}$), agregando restricciones lineales:

| restricción | $\mu$ | valores intermedios distintos |
|---|---|---|
| sin restricciones | 0.687923 | 6 |
| **reflejo** | **0.688579** | **4** |
| reflejo + un solo $q$ en los cuatro cruzados | 0.671711 (**< 11/16**) | 4 |
| pares vecinos en 1 (Tabla II pura) | infactible | — |

- **El reflejo no cuesta nada.** Si las fases son simétricas, el problema es invariante: si $P$ es
  factible, su reflejo también, y por convexidad el promedio de los dos es factible con el mismo $\mu$.
  Siempre hay un óptimo simétrico. Encima gana un poco, porque Nelder–Mead busca en 3 parámetros en
  lugar de 6.
- **Un solo $q$ pierde el resultado:** forzar lejano = cercano cuesta del orden de $h$ en $\mu$, que es
  justo la ventaja de S7 sobre 11/16. En un barrido de $h \in [0.005, 0.03]$ y dos márgenes, nunca pasó de
  0.6872.
- **Vecinos en 1 es infactible:** algunos "1" tienen que bajar.

---

## 11. El grafo simétrico

(Notebook: `notebooks/Grafo_simetrico_S7.ipynb`.)

### 11.1 La plantilla

Con el reflejo, la plantilla ($h = 0.03$, $\varepsilon = 10^{-3}$, tamaños $(3,5)$, $\mu = 0.688579$,
desplazamientos $s = (-0.9367,\ 0.8967,\ -1.2782,\ 1.2782,\ -0.8967,\ 0.9367)$) tiene esta matriz de
densidades completa. Las intermedias están en negrita, y la letra indica a qué pareja espejo pertenecen:

| $P$ | Am | Ap | Bm | Bp | Cm | Cp | W |
|---|---|---|---|---|---|---|---|
| **Am** | 1 | 1 | 1 | **0.6124** ᵃ | 0 | 0 | 1 |
| **Ap** | 1 | 1 | **0.6783** ᵇ | **0.9892** ᶜ | 0 | 0 | **0.9670** ᵈ |
| **Bm** | 1 | **0.6783** ᵇ | 1 | 1 | **0.9892** ᶜ | **0.6124** ᵃ | 0 |
| **Bp** | **0.6124** ᵃ | **0.9892** ᶜ | 1 | 1 | **0.6783** ᵇ | 1 | 0 |
| **Cm** | 0 | 0 | **0.9892** ᶜ | **0.6783** ᵇ | 1 | 1 | **0.9670** ᵈ |
| **Cp** | 0 | 0 | **0.6124** ᵃ | 1 | 1 | 1 | 1 |
| **W** | 1 | **0.9670** ᵈ | 0 | 0 | **0.9670** ᵈ | 1 | 1 |

- ᵃ Am–Bp y Bm–Cp: cruzados **lejanos** (0.6124);
- ᵇ Ap–Bm y Bp–Cm: cruzados **cercanos** (0.6783);
- ᶜ Ap–Bp y Bm–Cm (0.9892);
- ᵈ Ap–W y Cm–W (0.9670).

Son solo **cuatro valores** intermedios, cada uno en un par y su espejo. Todo lo demás es 0 (sectores
opuestos) o 1. La matriz respeta el espejo: la fila de Am leída al revés es la de Cp, la de Ap es la de
Cm, la de Bm es la de Bp, y la de W es simétrica en sí misma.

Comparada con el patrón límite de la sección 5.2: los cuatro $q$ se separaron en dos valores (lejano y
cercano), y cuatro "1" bajaron un poco. Esos cuatro son los pares que tocan a Ap o a Cm, las mitades que
miran hacia B.

### 11.2 Por qué la simetría ayuda al redondeo

Un movimiento de fases se refleja como $\delta\varphi_{R(a)} \mapsto -\delta\varphi_a$. Cada dirección del
Jacobiano es **simétrica** o **antisimétrica**. Las tres blandas de la plantilla simétrica:

| $\sigma$ | paridad |
|---|---|
| $5.4\cdot10^{-6}$ | antisimétrica |
| $6.9\cdot10^{-6}$ | **simétrica** |
| $8.1\cdot10^{-4}$ | antisimétrica |

**Si los grados enteros también son simétricos**, el error de redondeo es simétrico y no puede empujar en
las direcciones antisimétricas: esas ecuaciones se cumplen **solas**. Queda **una sola ecuación blanda**.

En las direcciones antisimétricas ya no hay pliegue. Lo único que hace falta es que el cambio de
densidades ($\sim 1/n$) sea menor que el margen de estabilidad ($\sim 5\cdot10^{-6}$): una condición lineal
en $1/n$, no cuadrática.

### 11.3 El sistema reducido

Un estado simétrico queda determinado por $\psi = (\varphi_{Am}, \varphi_{Ap}, \varphi_{Bm})$: las demás fases
son $\pi$ menos la de su espejo, y W queda en $3\pi/2$. Las ecuaciones también se reducen: en un estado
simétrico $F_{R(a)} = -F_a$ y $F_W = 0$. Quedan **3 ecuaciones con 3 incógnitas**, con 2 direcciones rígidas
y 1 blanda. (Ya no hay rotación global: rotar un estado simétrico lo deja de hacer simétrico.)

### 11.4 Cuatro enteros, una ecuación

- **Datos:** los 8 pares intermedios forman 4 parejas espejo, y cada pareja usa el mismo grado. Son **4
  enteros** $j_1, \dots, j_4$.
- **Pulido simétrico:** el SDP deja $F \approx 10^{-9}$; se corrige $P$ con el cambio de menor norma que hace
  $F = 0$ exacto, con una variable por pareja (3 ecuaciones, 4 incógnitas), para no romper la simetría.
- **Residuo blando:** Lyapunov–Schmidt en el sistema reducido: Newton en las 2 rígidas y $g$ = lo que queda
  en la blanda, **un solo número**.
- **El umbral:** la forma normal $\sigma y + \tfrac c2 y^2$ medida por diferencias finitas da
  $\sigma = 6.9\cdot10^{-6}$ y $c = 9\cdot10^{-4}$. El umbral del pliegue es $\sigma^2/(2|c|) \approx
  2.7\cdot10^{-8}$, unas 500 veces más que $\sigma^2$, porque $c$ también es chico.
- **Newton entero por búsqueda exhaustiva:** $G_c = g(j + e_c) - g(j)$ (cuánto cambia $g$ al sumarle 1 a
  un entero). Con 4 enteros y radio 20 hay $41^4 \approx 2.8$ millones de candidatos, y $g + G\cdot u$ se
  evalúa en todos con una multiplicación de matrices. Se elige el mejor y se repite una vez con radio 3.
  **No hace falta el reticulado.**

### 11.5 Resultados

| $n$ | ingenuo | $\lvert g\rvert$ ingenuo | $\lvert g\rvert$ Newton | Newton estricto | certificado |
|---|---|---|---|---|---|
| 100 004 | no | $1.6\cdot10^{-5}$ | $1.4\cdot10^{-8}$ | no | — |
| **299 989** | no | $1.5\cdot10^{-6}$ | $4.2\cdot10^{-10}$ | **sí** | **sí** |
| 999 994 | no | $1.4\cdot10^{-6}$ | $5.3\cdot10^{-11}$ | sí | sí |
| 3 000 005 | no | $2.2\cdot10^{-7}$ | $6.0\cdot10^{-11}$ | sí | sí |
| 10 000 009 | no | $6.1\cdot10^{-8}$ | $1.9\cdot10^{-11}$ | sí | sí |
| 30 000 004 | no | $6.0\cdot10^{-8}$ | $9.9\cdot10^{-13}$ | sí | sí |

- El ingenuo queda siempre por encima del umbral ($2.7\cdot10^{-8}$): nunca funciona.
- El Newton entero baja $|g|$ entre 3 y 5 órdenes de magnitud, moviendo cada entero a lo sumo 20 unidades.
- Con $n = 10^5$ el residuo ya está debajo del umbral, pero falla la otra condición: el redondeo cambia las
  densidades en $\sim 10^{-5}$, más que el margen $\sim 5\cdot10^{-6}$. **El límite ahora no es el
  pliegue, sino el margen de estabilidad.**

### 11.6 El grafo

$g = 13\,043$, $k = (3, 3, 3, 3, 3, 3, 5)$: seis mitades de 39 129 vértices y W de 65 215, en total
$n = 299\,989$.

| entero | pares | vecinos de cada vértice del 1.º en el 2.º | densidad |
|---|---|---|---|
| $j_1$ | Am–Bp y Bm–Cp | 23 984 | 0.6129 |
| $j_2$ | Ap–Bm y Bp–Cm | 26 540 | 0.6783 |
| $j_3$ | Ap–Bp y Bm–Cm | 38 697 | 0.9890 |
| $j_4$ | Ap–W y Cm–W | 63 070 (y 37 842 de W hacia la mitad) | 0.9671 |

- **Completos:** Am–Ap, Am–Bm, Am–W, Bm–Bp, Bp–Cp, Cm–Cp, Cp–W.
- **Vacíos:** A–C y B–W.
- **Certificado:** Krawczyk, Cholesky de $L$ (pivote mínimo $1.15\cdot10^{-6}$), Gershgorin y
  $\mu = 206565/299989 \approx 0.688575 > 11/16$.

Datos completos en `data/grafo_simetrico_s7.json` y figura en
`data/figuras/grafo_simetrico_n299989_estructura.png`.

---

## 12. Comparación final y qué queda abierto

| construcción | explícito | pesos | $n$ | $\mu$ | descripción |
|---|---|---|---|---|---|
| Canale (2022) | sí | no | $8k$ ($10^4$) | $\to 11/16$ por debajo (0.6874) | 4 pesos $(a,b,c,d)$ |
| O'Keeffe, Proposición 1 | no (existencia) | no | $> 9\cdot10^{84}$ | 0.6929050 | datos racionales con $D \approx 3\cdot10^{42}$ |
| nuestro, construcción general | sí | no | 30 000 004 | 0.688468 | 9 enteros libres |
| nuestro, camino B | sí | no | 1 500 014 | 0.688463 | sub-piezas |
| **nuestro, simétrico** | **sí** | **no** | **299 989** | **0.688575** | **4 enteros** |

### Lo que logramos

- Un grafo **escrito**, **sin pesos** y **certificado** con $\mu > 11/16$: supera a Canale con un grafo
  concreto, con el método de plantillas de O'Keeffe y el blow-up circulante de Canale.
- La versión simétrica es la más simple y la más chica: 4 enteros, una sola ecuación blanda, sin
  reticulados, y 100 veces menos vértices que la construcción general.

### Lo que no logramos

- **No mejoramos la cota de $\mu_c$:** O'Keeffe ya probó $\mu_c \ge 0.6929$. Nuestro aporte es el grafo
  explícito y chico, no un número mejor.
- **La cuenca es diminuta:** la trampa existe, pero desde fases al azar el grafo sincroniza.
- **El resultado simétrico depende del óptimo de Nelder–Mead:** con otra opción de Nelder–Mead la plantilla
  salió apenas distinta y el mínimo $n$ certificado fue $10^6$ en lugar de $3\cdot10^5$.
- **Para el grafo simétrico no repetimos** la validación ciega ni las simulaciones del notebook general.

### Abierto (de `notes/objetivos_grafo_elegante.md`)

- **O1, simetría espejo exacta:** hecho (este grafo).
- **O2, densidades puras en $\{0, q, 1\}$:** con fases en $\pm h$ es infactible (sección 10.4). Queda
  abierto si existe con fases libres.
- **O3, una familia infinita como la de Canale:** datos racionales fijos con denominador chico, que
  funcionen para todo $g$ múltiplo de él, con un único certificado.

---

## 13. Glosario

- **Blow-up:** reemplazar cada vértice (o cluster) por muchos vértices con la misma fase.
- **Blow-up circulante / k-spinning:** cada cluster es una copia de $\mathbb Z_g$ y cada vértice se une a un
  intervalo de vértices consecutivos del otro cluster; da grados exactos.
- **C4:** cuatro clusters a 90°; $\mu = 3/4$ pero neutro.
- **Certificado:** prueba por computadora con aritmética de intervalos (redondeo hacia afuera), que vale
  como demostración.
- **Cholesky por intervalos:** factorización que, si termina con pivotes positivos, prueba que una matriz
  es definida positiva para todos los valores de una caja.
- **Conectividad $\mu$:** $(\delta+1)/n$, con $\delta$ el grado mínimo.
- **Dirección blanda / rígida:** vector singular de $J$ con valor singular chico / grande.
- **Encaje de Kannan:** convertir "dos condiciones sobre enteros" en "el punto más cercano de un
  reticulado", juntándolas como bloques de un mismo vector.
- **Krawczyk:** test de intervalos que prueba que un sistema de ecuaciones tiene una única solución dentro
  de una caja.
- **LLL:** algoritmo que da una base corta y casi ortogonal de un reticulado.
- **Lyapunov–Schmidt:** resolver las ecuaciones bien condicionadas y quedarse con el residuo de las mal
  condicionadas.
- **Pliegue (silla-nodo):** punto donde un equilibrio estable y uno inestable se juntan y desaparecen;
  forma normal $\sigma y + \tfrac c2 y^2 + \rho = 0$.
- **Plano más cercano de Babai:** algoritmo que aproxima el punto de un reticulado más cercano a un vector.
- **Plantilla $(x, \varphi, P)$:** descripción de un estado de clusters por tamaños, fases y densidades.
- **Reflejo:** la simetría $\varphi \mapsto \pi - \varphi$ de S7 (Am↔Cp, Ap↔Cm, Bm↔Bp, W fijo).
- **SDP (programa semidefinido):** optimización convexa con restricciones lineales y de "matriz semidefinida
  positiva".
- **Trampa:** mínimo local estricto de la energía que no es la sincronización.
