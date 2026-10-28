Caso especifico de minimo local podemos multiplicarlo por infinitos nodos
"Stackearlos" uno encima de otro (los nodos en la misma "linea"/fase se conectan entre sí)
La idea es que los theta ya estan en la misma posicion entonces no puede ser mas chico. Por lo tanto la energia no deberia cambiar
El grado de los nodos solo crece en 1 por cada stackeo entre los que se conectan
Como los resortes se conectan entre puntos que estan en la misma fase, estos no se desestabilizan y por lo tanto ahora deberia ser mas estables

---

## Análisis (19/10/2026, con Claude)

### Dos formas de "stackear"

Sea θ* un mínimo local estricto no sincronizado de G (n vértices, grado mínimo δ). Se hacen m copias
de cada vértice, todas en la misma fase.

**(a) Cartesiano, G □ K_m (la idea tal como está escrita).** Hay m copias de G, y las copias de un
mismo vértice forman un clique. θ* sigue siendo mínimo estricto. El Hessiano es H_G ⊗ I + I ⊗ L_{K_m}
y sus autovalores son λ_i(H_G) + λ_j(K_m), todos > 0 salvo el 0 de la rotación. **Pero μ baja:**

  μ = (δ + (m − 1) + 1) / (mn) = (δ + m)/(mn) → 1/n.

El grado crece en m − 1 y n crece en un factor m. Ejemplo: en C_{20,6} con q = 1, μ pasa de 0.65 a
0.35 (m = 2) y a 0.25 (m = 3).

**(b) Lexicográfico, G[K_m] (blow-up).** Cada copia de v se une además con **todas** las copias de
cada vecino u de v.
- grado = m·deg(v) + (m − 1) y n' = mn, así que **μ(G[K_m]) = (δ + 1)/n = μ(G) exactamente**. Para
  esto sirve el "+1" de la definición de μ: es el self-loop que representa al clique.
- Sigue siendo equilibrio: la velocidad de cada copia es m veces la del vértice original.
- Sigue siendo mínimo estricto. Toda perturbación se descompone en dos subespacios ortogonales que el
  Hessiano no mezcla:
  1. **constantes por cluster** (todas las copias de v se mueven igual): ahí el Hessiano actúa como
     m·H_G, que es > 0 en 1^⊥ (todas las direcciones salvo la rotación global 1 = (1, …, 1));
  2. **de media cero dentro de cada cluster**: los términos entre clusters se cancelan y queda el
     autovalor m·(1 + Σ_{u~v} cos(θ_u − θ_v)), que es > 0 porque la diagonal de H_G es positiva.

Conclusión: el blow-up con m iguales **preserva** la trampa y μ, pero no puede superar 11/16 si G
no lo supera.

### Lo aprovechable: clusters de tamaños distintos

Cada vértice v de una plantilla chica H pasa a ser un clique de m_v = N·p_v vértices, con
Σ p_v = 1. Entonces:

- μ = min_v p(N[v]) = min_v (p_v + Σ_{u~v} p_u);
- equilibrio: Σ_{u~v} p_u sin(θ_u − θ_v) = 0 para todo v;
- estabilidad entre clusters: el Laplaciano con pesos p_u p_v cos(θ_u − θ_v) es > 0 en 1^⊥;
- estabilidad dentro del cluster v: p_v + Σ_{u~v} p_u cos(θ_u − θ_v) > 0.

Los pesos permiten "engordar" a los vecinos del vértice que limita el grado mínimo. Con eso, la
búsqueda sobre grafos grandes se reduce a un problema continuo sobre (H, p, θ) con H chico.

### Ojo: ya está hecho (O'Keeffe, arXiv 2610.21271, 15/10/2026)

`papers/2610.21271v1.pdf` usa exactamente este marco: "templates" (x, φ, P) con tamaños de cluster,
fases y **densidades** P_ab ∈ [0, 1] entre clusters, que después pasa a grafos por blow-up. Nuestras
condiciones coinciden con las suyas: d_a (grado), F_a = 0 (equilibrio), c_a > 0 ("stability against
splitting") y L(y) > 0 ("cluster modes"). Con eso prueba **μ_c ≥ 0.6929077**, que supera el 11/16 de
Canale. Su estado S7 parte tres clusters vecinos del estado de 4 clusters C4 y adelgaza algunos links
a densidad q* ≈ 0.65.

Según el paper, las búsquedas con margen de estabilidad y links 0/1 vuelven siempre al estado de
Canale (11/16). Nuestra versión, con clusters completos y links 0/1, es un caso particular de la
suya, así que difícilmente supere 0.6929.
