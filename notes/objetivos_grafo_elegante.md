# Objetivos: de "un grafo que funciona" a un grafo S7 con forma reconocible

**Estado (23/10/2026):** discusión abierta, **sin decidir**. Falta elegir la prioridad (ver la última
sección) antes de programar.

## Punto de partida: qué es y qué no es el grafo actual
El grafo de n = 30 000 004 (y el del camino B, n = 1 500 014) está en `notes/blowup_explicito.md`.

**Lo que es exacto, sin vendas:**
- es un grafo concreto: grados enteros y aristas por intervalos circulantes;
- el certificado por intervalos prueba que tiene un equilibrio estable no sincronizado con μ > 11/16;
- la validación ciega lo confirma;
- el blow-up ya es como el de Canale (circulante, determinístico), no el aleatorio de O'Keeffe, que
  obliga a n = D²ℓ.

**Lo que no es: la construcción de O'Keeffe con menos nodos.** Comparte con S7 la estructura (7
clusters, C4 partido, W entero, mismos pares vacíos, mismo patrón límite), pero sus números salen de
una optimización:

| | O'Keeffe | grafo actual |
|---|---|---|
| Densidades | fórmula P = p(q*−t) + h·p′ + Δ | SDP (optimización libre) |
| Desplazamientos s | (−1, 1, −w*, w*, −1, 1), fijos | Nelder–Mead: (−0.873, 0.833, −1.247, 1.242, −0.912, 0.948) |
| Simetría espejo A ↔ C | exacta | rota levemente (0.08 en s; densidades espejo difieren hasta 0.2%) |
| Tamaños | a = 1/(7+q) | a = 3/23, fijados a mano |
| h | → 0 (10⁻⁶ en su certificado; t = 10⁻⁵) | 0.03 |
| Parámetro t | distancia a la esquina, q = q* − t | no existe; su papel lo cumple el margen ε del SDP |

Nada se ajustó a mano para que pase el certificado, pero el resultado es "una selección que funciona",
lo que lo hace **poco vistoso para la competencia**: no se puede decir "tomamos S7 y lo construimos con
menos nodos", y no tiene una descripción corta ni propiedades limpias.

Otros hechos que entran en la discusión:
- **Los links W–A no son completos** (96.7%–100%). En O'Keeffe tampoco a h finito: su p′ da
  Am–W = 1 − 0.049h y Ap–W = 1 − 2.86h, así que no es una desviación de S7, pero sí muestra que la
  fórmula a h finito no es "limpia".
- **Costo de la fidelidad estricta:** la fórmula de O'Keeffe pierde mucho μ a h finito (con t = 10⁻³,
  h = 0.01 da μ = 0.6830 < 11/16). Su propia cota es μ ≥ μ₀(q) − 0.9956h, así que superar 11/16 pide
  h ≲ 0.005, con margen ~h²(t+h) ≲ 10⁻⁷ y un n mucho mayor (estimación sin medir: 10⁹–10¹¹).
- **La cuenca de la trampa es diminuta** (~10⁻⁴ rad en las direcciones lentas; sección 10 del
  notebook): una consecuencia del margen chico que cualquier versión va a heredar en algún grado.

## Qué haría "agradable" al grafo
1. **Simetría exacta.** Automorfismo espejo A ↔ C (Am ↔ Cp, Ap ↔ Cm, Bm ↔ Bp, W fijo), además de la
   rotación de Z_g del blow-up. Se impone en el SDP y en el redondeo.
   - **Hipótesis:** si alguno de los 3 modos blandos es antisimétrico, la simetría lo anula y el
     redondeo tiene menos ecuaciones que cuidar, así que n podría bajar.
2. **Densidades puras en {0, q, 1}.** El grafo se define en una línea: "7 grupos, uniones completas o
   vacías salvo 4 pares con densidad q".
   - **Pregunta abierta:** O'Keeffe necesita p′ porque fija las fases en ±h. Con las fases libres
     (resueltas por Newton) quizá exista un equilibrio partido y estricto con el patrón puro, o quizá
     el único sea C4, que es degenerado.
3. **Una familia infinita, como Canale.** Una plantilla con datos racionales exactos y denominador D
   moderado funciona para todo g múltiplo de D, sin redondear:
   - familia G_ℓ con n = c·ℓ y el mismo μ exacto para todos;
   - un enunciado de tipo teorema: "para todo ℓ ≥ 1, G_ℓ tiene un equilibrio estable no sincronizado";
   - un único certificado, porque las cotas de Gershgorin escalan linealmente con g.
   - **Dificultad:** el pliegue. Hoy hace falta D ~ 10⁶; el objetivo es un miembro con D ~ 10³–10⁴,
     aceptando menos μ, pero siempre por encima de 11/16.

Lo ideal es combinar las tres: una familia simétrica, con densidades simples y certificada para todo ℓ.

## Objetivos propuestos

| # | Objetivo | Criterio de éxito | Riesgo |
|---|---|---|---|
| O1 | Versión con simetría espejo exacta | certificado con μ > 11/16; medir si n baja | bajo |
| O2 | Patrón puro {0, q, 1} | existe un equilibrio estricto con fases partidas y μ > 11/16 | alto: puede no existir |
| O3 | Familia G_ℓ con datos racionales fijos | un certificado válido para todo ℓ; menor miembro con n ≲ 10⁶ | medio-alto |
| — | Grafo actual | queda como "variante optimizada" y punto de comparación | — |

**Orden sugerido:** O1, que es barato y dice si la simetría alivia el pliegue, y en paralelo una
sonda de pocas horas para O2. O3 depende de lo que salga de las dos.

**Alternativa considerada: fidelidad estricta a O'Keeffe.** Densidades por su fórmula P(t, h),
fases por su ansatz más Newton, espejo exacto, W entero, a = 1/(7+q) con q racional (por ejemplo,
q = 16/25 da a = 25/191 exacto), y como única libertad el redondeo de los grados respetando el espejo.
El resultado sería "S7 de O'Keeffe con n ~ 10⁹–10¹¹" (sin medir), fiel pero probablemente grande y
no más vistoso que O1–O3.

## Decisiones pendientes
1. **Prioridad si las propiedades chocan.** Puede que la versión más simple tenga μ apenas sobre
   11/16 o necesite un n mayor. Propuesta: priorizar la familia (O3) aunque baje μ, porque da un
   enunciado de tipo teorema como el de Canale. Falta la opinión del grupo.
2. **El camino B** (partir W) rompe "W entero": ¿queda fuera, o solo como variante?
3. **Lugar del grafo actual en el relato:** resultado aparte ("variante optimizada") o solo
   comparación.
