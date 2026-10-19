# Dirección A: qué logramos y por dónde seguir (19/10/2026)

Detalle técnico y números: `notes/direccion_A.md`. Notebook: `notebooks/02_plantillas.ipynb`.

## Qué logramos

**1. Analizamos la idea del "stackeo"** (`notes/idea.md`).
- En la versión literal (copias de G unidas solo entre sí, producto cartesiano) μ **baja**.
- La versión correcta es el blow-up lexicográfico G[K_m]: deja μ **igual** y conserva la trampa.
  Por sí solo no mejora la cota.
- La generalización con clusters de tamaños distintos y densidades entre clusters es exactamente
  el marco de O'Keeffe (arXiv 2610.21271, 15/10/2026), que con él sube la cota a **μ_c ≥ 0.6929**.
  La idea era buena, pero ya estaba publicada.

**2. Reprodujimos de forma independiente todo lo conocido**, con código propio y abierto:
- twisted states: 0.6838 (Yoneda et al.);
- Canale: 11/16, con grafos explícitos verificados (Hessiano + simulación);
- S7 de O'Keeffe: 0.6929.

En la rúbrica de la presentación, esto cubre el nivel **"básico"** ("pipeline reproducible que
reproduce lo conocido").

**3. Hicimos un certificado riguroso propio** del miembro de S7 de la Proposición 1 de O'Keeffe:
Krawczyk + Cholesky por intervalos, con controles negativos que fallan donde deben. Los scripts del
paper no son públicos ("on request"), así que una verificación independiente y abierta tiene valor
por sí misma.

**4. Exploramos sistemáticamente y obtuvimos resultados negativos documentados** (nivel
**"esperado"**):
- con arranques al azar no se reencuentra ni S7;
- partir el cluster "ancla", o menos de tres clusters, empeora;
- partir en 3 piezas en vez de 2 no mejora μ0 en el límite h → 0;
- los blow-ups completos (solo densidades 0/1) de grafos base de hasta 7 vértices no pasan de
  μ ≈ 0.616.

**No mejoramos la cota.** Lo de la exploración es evidencia numérica, no está certificado.

## Por dónde seguir para subir la cota

### A. Modelo límite h → 0 general (recomendado)

El cuello de botella, nuestro y de O'Keeffe, es numérico: la brecha de estabilidad escala como
h²(t + h), así que hay que trabajar con h finito, y ahí los márgenes y las cuencas angostas arruinan
la búsqueda. O'Keeffe resolvió el límite h → 0 solo para el patrón simétrico de S7, y dice
explícitamente que **no probó** que μ0 sea óptimo ni siquiera dentro de ese patrón.

Propuesta: hacer el desarrollo en h de forma genérica, con fases jπ/2 + h·s y densidades p + h·p′.
Con x y s fijos, todas las condiciones son **lineales en (p, p′)**:
- equilibrio a orden 1 y a orden h;
- condiciones de corte;
- forma de segundo orden G₀ ⪰ 0.

Entonces el límite es otra vez un **SDP, pero sin h y bien condicionado**. Eso permite recorrer
patrones arbitrarios: mitades desiguales, asimetrías, más piezas. Si algún punto supera μ0, el
camino a una prueba ya está armado:
1. elegir un miembro finito (t, h);
2. certificarlo con `src/certify.py`;
3. convertirlo en grafos con el Lema 2 de O'Keeffe, que se cita.

Primer paso: derivar e implementar ese SDP y validarlo recuperando q*, w* y μ0 desde el patrón de S7.

### B. Partir el estado de Canale en su borde

En d = 1/2 Canale también es degenerado (c² = 2ad). Es el mismo mecanismo que C4 en 3/4: un estado
al borde de la estabilidad que se rescata partiendo clusters. O'Keeffe encontró S7 "partiendo y
fusionando clusters de estados tipo C4 y tipo Canale", pero no reporta un análisis límite alrededor
de Canale. Con el código de A sale casi gratis.

### C. Combinar el estado "satélite" (0.69155) con S7

Son dos deformaciones distintas de C4 y el paper no prueba combinarlas. Es más especulativo.

### D. Mínimos no estrictos (estabilidad de orden superior)

Hay estados con Hessiano degenerado que igual son mínimos locales por términos de orden 4, y la
definición de μ_c pide mínimo local, no Hessiano definido positivo. Es la vía más arriesgada:
certificarlos es mucho más difícil y la presentación pide certificados de Hessiano.

### E. Buscador de semillas: otras familias más allá de C4 y Canale

**Idea.** S7 y Canale siguen el mismo esquema:
1. una **semilla**: un estado de clusters con μ alto justo en el borde de la estabilidad (L ⪰ 0 con
   modos neutros);
2. una **deformación** (partir, fusionar, agregar satélites) que la vuelve estricta, con algún
   costo en μ.

Buscar "otras familias" es buscar **otras semillas**.

**Barrido rápido (19/10).** SDP con L ⪰ 0 sin brecha sobre dos familias de geometrías: una red de
ℓ clusters equiespaciados, y dos redes de ℓ puntos rotadas un ángulo β. Se probó ℓ = 3–12 para una
red y ℓ ∈ {3, 4, 5, 6, 8} con pesos r ∈ {0.3, 0.4, 0.5} para dos redes.

| geometría | mejor μ de borde | modos neutros |
|---|---|---|
| una red, ℓ = 4 (C4) | **0.75** | 3 |
| una red, ℓ = 10 | 0.682 | 2 |
| dos redes, ℓ = 4, β ≈ 26° (Canale) | 0.6869 ≈ 11/16 | 2 |
| dos redes, ℓ = 8 | 0.6839 | 2 |
| dos redes, ℓ = 5 | 0.6831 | 2 |

Conclusiones:
- **Todos los óptimos son semillas.** El óptimo del SDP siempre queda en el borde, con 2 modos
  neutros (C4 tiene 3). El barrido redescubre Canale sin buscarlo.
- **Hay dos tipos de semilla:**
  - **arriba del récord** (μ > 0.6929): la deformación las estabiliza **perdiendo** μ, como C4
    (0.75 → 0.6929). En estas familias, **C4 es la única**.
  - **abajo del récord** (todas las demás, incluida Canale): la deformación tendría que **ganar**
    μ. Para Canale serían ~0.005 con correcciones en solo dos direcciones neutras; parece
    improbable.
- **Las redes regulares están agotadas:** no dan semillas nuevas por encima de 0.6929.

**Dónde buscar.** Por Lepsveridze–Zhang, cerca de 3/4 el único obstáculo es el de 4 clusters, así
que las semillas nuevas, si existen, viven en la **franja intermedia, aproximadamente entre 0.693 y
0.74**, donde no se sabe nada. Hay que salir de las redes regulares:
- tres o más redes con pesos y rotaciones libres;
- C4 con clusters desiguales o fases no equiespaciadas (cruza con la dirección C);
- estados intermedios entre C4 y Canale;
- fusiones de clusters de dos semillas distintas.

**Herramienta.** Maximizar μ sobre (x, φ, P) con L ⪰ 0 y quedarse con los óptimos degenerados con
μ > 0.6929, **excluyendo C4 y sus parientes**; si no, el optimizador cae siempre en 0.75. Por
ejemplo, se pueden penalizar las configuraciones con todas las fases cerca de cuartos de vuelta.
Cada semilla nueva pasa después al análisis de deformación (modelo límite de la vía A).

### Sobre usar redes neuronales / RL

- **RL sobre grafos directamente (estilo Wagner 2021):** no conviene.
  - Los récords son límites de grafos enormes: Canale con 160 vértices da 0.679, y el grafo de S7
    tiene más de 10⁸⁴ vértices.
  - La recompensa (encontrar un mínimo no sincronizado) es cara y casi siempre da "nada", aunque el
    estado exista.
- **RL sobre la estructura de plantillas:** sí, como complemento, después de tener el modelo límite
  (A) o el buscador de semillas (E).
  - Un agente con el método de entropía cruzada de Wagner propone la estructura discreta: cuántos
    clusters, en qué sectores de fase, cuáles se parten, qué pares se prohíben.
  - La recompensa es el SDP (~10 ms).
  - Donde más sentido tiene es en E, que tiene una parte combinatoria grande.
- **Vías B, C y D:** no hace falta.
  - B tiene unos cientos de patrones salvo simetría y se enumera completo; eso permite decir
    "probamos todos".
  - C tiene pocas variables.
  - D es un problema de prueba, no de búsqueda.
- **Siempre con línea base:** el RL solo aporta si encuentra algo que la enumeración no alcanza; hay
  que reportar las dos cosas lado a lado. Aunque no encuentre nada, sirve para la dirección "RL" de
  la slide 27 y para la sección de uso de IA del informe.

## Expectativa realista

Superar 0.6929 en lo que queda de la competencia es poco probable. Pero A es la herramienta
adecuada, y un resultado de A, positivo o negativo, sería un aporte genuino y verificable. Un
ejemplo de resultado negativo: "μ0 es óptimo localmente entre todos los patrones de partición de C4
con hasta k piezas".
