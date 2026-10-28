Te lo explico de punta a punta, empezando por el mapa general, porque cada término técnico resuelve un problema puntual dentro de esa cadena.

---

El mapa: de una idea a un grafo concreto

Buscamos un grafo con una trampa (un mínimo local estricto que no es la sincronización) y con $\mu > 11/16$. Escribir directamente un grafo de millones de vértices es imposible, así que se avanza en tres pasos:

Plantilla (números reales)     →  Datos enteros (grados d_ab)  →  Grafo (lista de aristas)
x, φ, P  con 7 clusters           cuántos vecinos tiene cada       blow-up circulante
                                  vértice de a en b
     FASE 0                            FASE 1                         FASE 2
  "¿qué P elegir?"              "¿cómo redondear P sin           "¿cómo armar un grafo
   (SDP)                          romper el equilibrio?"           con esos grados exactos?"
                                  (reticulado)

La plantilla describe el grafo "en promedio": 7 clusters ($A_m, A_p, B_m, B_p, C_m, C_p, W$), cada uno es un clique donde todos sus vértices tienen la misma fase $\varphi_a$. El cluster $a$ ocupa una fracción $x_a$ de los vértices, y $P_{ab}$ es la fracción del cluster $b$ a la que está conectado cada vértice de $a$. Si $P_{AmBp}=0.614$, cada vértice de $A_m$ está unido al 61.4% de $B_p$.

Lo útil es que, si cada vértice de $a$ tiene exactamente $P_{ab},n_b$ vecinos en $b$, todo lo que importa del grafo (si es equilibrio, si es estable, su $\mu$) se calcula con esos 7 números $x$, 7 fases y la matriz $P$ de 7×7. El grafo de 30 millones de vértices queda resumido en un problema de 7×7.

---

Fase 0: elegir $P$ (el SDP)

Qué se quiere

Con los tamaños $x$ y las fases $\varphi$ fijos, se busca la matriz $P$ que haga $\mu = \min_a d_a$ lo más grande posible y que a la vez cumpla:
1. Equilibrio: $F_a = \sum_b P_{ab} x_b \sin(\varphi_a-\varphi_b) = 0$.
2. Estabilidad de un vértice suelto: $c_a > 0$.
3. Estabilidad de los modos de cluster: $L(y) > 0$ para todo movimiento $y$ que no sea una rotación global, y con un margen: $\lambda_{\min}(L)/h^2 \ge \varepsilon$.
4. Que $P$ sea una densidad válida: $0 \le P \le 1$, simétrica, ceros en los pares opuestos (A–C, B–W).

Por qué es un SDP

Como $\varphi$ y $x$ están fijos, los senos y cosenos son números. Entonces:
- $F$, $c$ y los grados $d_a$ son lineales en $P$ (sumas de $P_{ab}$ por constantes);
- la matriz $L$ también es lineal en $P$.

La condición 3 dice "la matriz $L$ es semidefinida positiva" (todos sus autovalores son ≥ 0, que equivale a $y^\top L y \ge 0$ para todo $y$). Un problema con:
- objetivo lineal,
- restricciones lineales,
- y una restricción "esta matriz, que depende linealmente de las variables, es semidefinida positiva",

se llama SDP (programa semidefinido). Lo importante es que es convexo: no tiene mínimos locales falsos, y un solver (CLARABEL, vía cvxpy) encuentra el óptimo global en milisegundos. En sdp_interno, la línea (Q.T @ L @ Q) / escala >> margen * I es esa restricción; el >> de cvxpy significa "semidefinida positiva".

(La $Q$ proyecta sobre los vectores perpendiculares a $(1,\dots,1)$ para descartar la rotación global, que siempre tiene autovalor 0.)

El lazo de afuera

Las fases dependen de los desplazamientos $s$ de las 6 mitades. Con $s$ dentro de las variables el problema deja de ser convexo, así que se separa en dos niveles:
- adentro, para cada $s$, el SDP da el mejor $P$ y su $\mu$;
- afuera, Nelder–Mead (un optimizador que no usa derivadas) prueba distintos $s$ buscando el mayor $\mu$.

El resultado de la Fase 0: una curva

Cuanto más margen $\varepsilon$ se pide, menos $\mu$ queda. Ese costo se mide para varios $h$ y $\varepsilon$, y la plantilla elegida es $h=0.03$, $\varepsilon=10^{-3}$, con $\mu = 0.6885$. ¿Para qué sirve el margen? Es lo que va a permitir redondear a enteros en la Fase 1 sin romper el equilibrio. Sin margen, el redondeo lo destruye.

---

Fase 1: pasar a enteros

El problema

Un grafo real necesita enteros: $d_{ab}$ = cuántos vecinos tiene cada vértice de $a$ en $b$. La plantilla da $P_{ab},n_b$, que es un número real, y hay que redondearlo (respetando además $n_a d_{ab} = n_b d_{ba}$, que es contar las mismas aristas desde los dos lados).

El redondeo ingenuo redondea cada $d_{ab}$ por separado al entero permitido más cercano. Cada densidad cambia en $\sim 1/n$. Parece poco, pero no alcanza.

Por qué no alcanza: el pliegue

El Jacobiano $J$ de las ecuaciones de equilibrio tiene tres valores singulares diminutos ($\sigma \approx 6\cdot10^{-4},\ 5\cdot10^{-6},\ 2\cdot10^{-6}$). Son tres direcciones "blandas" en las que el equilibrio casi no está sostenido.

Una imagen: el equilibrio es una bolita en un pozo muy poco profundo. Si inclinás un poco la mesa (el error del redondeo), el pozo desaparece y la bolita se va. En fórmulas, en cada dirección blanda el equilibrio es raíz de

$$\sigma, y + \tfrac{c}{2}y^2 + \rho = 0,$$

donde $\rho$ es el error que mete el redondeo. Esta ecuación cuadrática tiene raíz solo si $\rho \lesssim \sigma^2$. Si $\rho$ es más grande, no hay equilibrio ("se pasó el pliegue"). Ese es el pliegue de tipo silla-nodo.

Con el redondeo ingenuo $\rho \sim 1/n$, así que hace falta $1/n \lesssim \sigma^2$, o sea $n \gtrsim 1/\sigma^2$. Con estos $\sigma$ eso da $n\approx 10^9$. Con $n = 3\cdot10^7$, el notebook muestra que el ingenuo da un equilibrio inestable.

La idea clave: redondear todo junto

Hay 13 grados libres (los pares con $0<P_{ab}<1$) y solo 3 ecuaciones delicadas (las blandas). Cada redondeo comete un error, pero los errores se pueden combinar para que se cancelen en esas 3 direcciones.

Un ejemplo de juguete: buscás enteros chicos $u_1, u_2$ con $u_1 + \sqrt2,u_2 \approx 0$. Redondear cada uno por separado no ayuda, pero $(u_1,u_2)=(-7,5)$ da $-7+7.071 = 0.07$, y $(17,-12)$ da $0.03$. Elegidos en conjunto, los enteros anulan una forma lineal mucho mejor que $1/2$. Eso mismo se hace con 13 enteros y 3 formas lineales.

Reducción de Lyapunov–Schmidt

Hay 6 ecuaciones de equilibrio (7 fases menos la rotación). Lyapunov–Schmidt es separar las fáciles de las difíciles usando la SVD $J = U S V^\top$:
- 3 direcciones rígidas (valores singulares grandes): ahí el pozo es profundo y cualquier error chico se absorbe. Se resuelven exactamente con Newton común.
- 3 direcciones blandas: después de resolver las rígidas, queda un residuo $g = U_b^\top F$ de 3 números.

Todo el problema se reduce a esto: elegir los 13 enteros para que esos 3 números sean $\ll \sigma^2$. $g$ es casi lineal en los grados, lo que habilita el paso siguiente.

Newton entero

Es Newton común, pero el paso tiene que ser un vector de enteros:
- se linealiza: $g(d+u) \approx g + \mathrm{d}g\cdot u$;
- se busca $u \in \mathbb{Z}^{13}$ con $\mathrm{d}g\cdot u \approx -g$, con cada componente medida en unidades de $\sigma_k^2$ (la tolerancia de esa dirección);
- $u$ tiene que ser chico (radio 3 a 100) para no estropear las direcciones rígidas;
- se repite unas pocas veces.

Encontrar ese $u$ entero es el problema del vector más cercano en un reticulado, y para eso están las tres herramientas que siguen.

Encaje de Kannan: armar el reticulado

Un reticulado es el conjunto de todas las combinaciones con coeficientes enteros de unos vectores base (como $\mathbb{Z}^2$ en el plano, pero posiblemente inclinado y estirado).

Se quieren dos cosas a la vez: $u$ cerca del objetivo real $u_0$, y $A(u-u_0)\approx 0$. El truco de Kannan es pegar las dos en un solo vector:

$$u ;\longmapsto; \big(,u/r,;; \text{peso}\cdot A u,\big) \in \mathbb{R}^{13+3}.$$

Los vectores de este reticulado (uno por cada $u$ entero) son las filas de B = [I/r | peso·Aᵀ]. El vector objetivo es $(u_0/r,\ \text{peso}\cdot A u_0)$. Si un vector del reticulado está cerca del objetivo, entonces:
- las primeras 13 coordenadas están cerca, así que $u \approx u_0$ (a escala del radio $r$);
- las últimas 3 están cerca, y como el peso es grande, $Au \approx Au_0$ con mucha precisión.

Así, "dos objetivos" pasa a ser "el punto del reticulado más cercano a este vector".

LLL: conseguir una buena base

Un mismo reticulado tiene muchas bases, y con una base mala redondear da cualquier cosa. Por ejemplo, $b_1=(1,0)$ y $b_2=(100,1)$ generan exactamente $\mathbb{Z}^2$. Para el punto $(0.4,,0.4)$:
- en esa base, $(0.4,0.4) = -39.6,b_1 + 0.4,b_2$; al redondear los coeficientes queda $-40,b_1 + 0,b_2 = (-40, 0)$, a distancia 40;
- con la base buena $(1,0),(0,1)$, el redondeo da $(0,0)$, que es el correcto.

LLL (Lenstra–Lenstra–Lovász) toma una base mala y devuelve una base del mismo reticulado con vectores cortos y casi perpendiculares. Lo hace con dos operaciones repetidas: restarle a un vector múltiplos enteros de los anteriores (lo acorta) e intercambiar dos vectores consecutivos cuando el orden es malo (la "condición de Lovász" en el código). En el ejemplo, convertiría $(100,1)$ en $(0,1)$ restándole $100,b_1$.

Plano más cercano de Babai: redondear bien

Con la base ya reducida, Babai aproxima el punto más cercano:
- ortogonaliza la base (Gram–Schmidt);
- del último vector al primero: mira cuánto del objetivo "cae" en esa dirección ortogonal, lo redondea a un entero, resta ese múltiplo del vector y sigue con el anterior.

Geométricamente, el reticulado se ve como una pila de "planos" paralelos: elige el plano más cercano, baja una dimensión y repite. No garantiza el óptimo exacto, pero con una base LLL queda cerca, y acá alcanza.

Qué gana todo esto

- En el juguete de la celda 30, redondear coordenada a coordenada deja $|A(u-u_0)|\sim 1$, y el reticulado lo baja varios órdenes de magnitud.
- En el problema real, con $n = 3\cdot10^7$, el ingenuo es inestable y el reticulado da un equilibrio estricto, cambiando solo unos pocos grados en unas pocas unidades (de un total de $\sim10^7$).
- El $n$ necesario baja de $10^9$ a $3\cdot10^7$.

---

Y de ahí al grafo (Fase 2)

Al salir de la Fase 1 hay una tabla de enteros $d_{ab}$ con las propiedades buscadas. Falta un grafo en el que cada vértice de $a$ tenga exactamente $d_{ab}$ vecinos en $b$:
- cada cluster $a$ se parte en $k_a$ bloques de $g$ vértices, numerados $0,\dots,g-1$ en ronda;
- el vértice $v$ de un bloque se une con $v, v+1, \dots, v+t-1 \pmod g$ del otro bloque (un intervalo, como el k-spinning de Canale);
- las longitudes $t$ se reparten entre bloques para que las sumas den $d_{ab}$ y $d_{ba}$ (la "tabla de transporte").

Como todos los vértices de un cluster ven exactamente lo mismo, las fases de la plantilla son un equilibrio exacto del grafo, y la estabilidad se hereda de la de la plantilla de enteros (eso lo certifica la Fase 3).

En una línea: el SDP elige las densidades ideales con margen de estabilidad (Fase 0); el reticulado las convierte en enteros con errores que se cancelan justo en las 3 direcciones frágiles (Fase 1); y el blow-up circulante convierte esos enteros en aristas (Fase 2).

Si querés, puedo agregar al notebook una celda con el ejemplo 2D de LLL/Babai y el de $u_1+\sqrt2,u_2$, para ver esas ideas antes del caso de 13 dimensiones.