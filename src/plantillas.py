"""Plantillas de clusters (x, φ, P) para estados de Kuramoto no sincronizados.

Notación de O'Keeffe (arXiv 2610.21271, sec. III):
  x_a    fracción de osciladores en el cluster a (Σ x_a = 1)
  φ_a    fase común del cluster a
  P_ab   densidad de links entre los clusters a y b (P_aa: dentro del cluster)

Condiciones en el límite de red grande:
  d_a = Σ_b P_ab x_b                         grado / n;    μ(T) = min_a d_a
  F_a = Σ_b P_ab x_b sin(φ_a − φ_b) = 0      equilibrio
  c_a = Σ_b P_ab x_b cos(φ_a − φ_b) > 0      estabilidad contra splitting
  L(y) = Σ_{a<b} x_a x_b P_ab cos(φ_a − φ_b)(y_a − y_b)² > 0 fuera de y ∝ 1   (modos de cluster)

Una plantilla es *estricta* si cumple las tres condiciones. Su blow-up da grafos con un mínimo local
estricto no sincronizado (Lema 2 de O'Keeffe; Prop. 2 de Canale para los k-spinnings).

Las funciones genéricas (grados, fuerzas, splitting, matriz_clusters, jacobiano_fuerzas) reciben
`lib` = numpy, mpmath o mpmath.iv, y aceptan x como secuencia y P como algo indexable P[a][b]
(lista de listas o array de numpy). Así hay una sola implementación para float, alta precisión e
intervalos.
"""
from dataclasses import dataclass

import mpmath as mp
import numpy as np


# ---------------------------------------------------------------- cantidades genéricas

def grados(x, P):
    """d_a = Σ_b P_ab x_b (grado normalizado de un vértice del cluster a)."""
    m = len(x)
    d = []
    for a in range(m):
        d.append(sum(P[a][b] * x[b] for b in range(m)))
    return d


def fuerzas(x, phi, P, lib=np):
    """F_a = Σ_{b≠a} P_ab x_b sin(φ_a − φ_b); el equilibrio es F = 0."""
    m = len(x)
    F = []
    for a in range(m):
        total = 0
        for b in range(m):
            if b != a:
                total = total + P[a][b] * x[b] * lib.sin(phi[a] - phi[b])
        F.append(total)
    return F


def splitting(x, phi, P, lib=np):
    """c_a = Σ_b P_ab x_b cos(φ_a − φ_b): cuánto tira el resto hacia el cluster a si un oscilador se separa."""
    m = len(x)
    c = []
    for a in range(m):
        total = P[a][a] * x[a]
        for b in range(m):
            if b != a:
                total = total + P[a][b] * x[b] * lib.cos(phi[a] - phi[b])
        c.append(total)
    return c


def matriz_clusters(x, phi, P, lib=np):
    """Matriz de L(y): Laplaciano con pesos x_a x_b P_ab cos(φ_a − φ_b), como lista de listas."""
    m = len(x)
    L = []
    for a in range(m):
        fila = [0] * m
        for b in range(m):
            if b != a:
                fila[b] = -x[a] * x[b] * P[a][b] * lib.cos(phi[a] - phi[b])
        diagonal = 0
        for b in range(m):
            if b != a:
                diagonal = diagonal - fila[b]
        fila[a] = diagonal
        L.append(fila)
    return L


def jacobiano_fuerzas(x, phi, P, lib=np):
    """∂F_a/∂φ_b para a, b = 1..m−1 (la fase φ_0 queda fija para sacar la rotación global)."""
    m = len(x)
    J = []
    for a in range(1, m):
        fila = []
        for b in range(1, m):
            if a == b:
                diagonal = 0
                for k in range(m):
                    if k != a:
                        diagonal = diagonal + P[a][k] * x[k] * lib.cos(phi[a] - phi[k])
                fila.append(diagonal)
            else:
                fila.append(-P[a][b] * x[b] * lib.cos(phi[a] - phi[b]))
        J.append(fila)
    return J


def base_perpendicular(m):
    """Base ortonormal (m × (m−1)) del ortogonal de (1, …, 1)."""
    Q, _ = np.linalg.qr(np.eye(m) - 1.0 / m)
    return Q[:, : m - 1]


def analizar(x, phi, P, lib=np, dps=60):
    """μ, max|F|, min c y autovalores de L fuera de la rotación global, en float (np) o alta precisión (mp).

    Con mp se trabaja con dps dígitos y se usa M = X^{-1/2} L X^{-1/2}, cuyo autovalor 0 corresponde
    a la rotación y se descarta.
    """
    if lib is mp:
        with mp.workdps(dps):
            return _analizar(x, phi, P, lib)
    return _analizar(x, phi, P, lib)


def _analizar(x, phi, P, lib):
    """Cuerpo de analizar, con la precisión ya fijada."""
    d = grados(x, P)
    F = fuerzas(x, phi, P, lib)
    c = splitting(x, phi, P, lib)
    L = matriz_clusters(x, phi, P, lib)
    if lib is np:
        Q = base_perpendicular(len(x))
        autovalores = list(np.linalg.eigvalsh(Q.T @ np.array(L, dtype=float) @ Q))
    else:
        m = len(x)
        raices = [mp.sqrt(xa) for xa in x]
        M = mp.matrix(m, m)
        for a in range(m):
            for b in range(m):
                M[a, b] = L[a][b] / (raices[a] * raices[b])
        autovalores = sorted(mp.eigsy(M)[0])
        indice_rotacion = min(range(m), key=lambda i: abs(autovalores[i]))
        autovalores.pop(indice_rotacion)
    return {
        "mu": min(d),
        "max|F|": max(abs(f) for f in F),
        "min c": min(c),
        "autovalores L": autovalores,
    }


# ---------------------------------------------------------------- plantilla numérica

@dataclass
class Plantilla:
    """Plantilla (x, φ, P) en float, con sus cantidades y el test de estrictez."""

    x: np.ndarray
    phi: np.ndarray
    P: np.ndarray
    nombre: str = ""

    def __post_init__(self):
        """Convierte a arrays de numpy y valida simetría de P y Σ x = 1."""
        self.x = np.asarray(self.x, dtype=float)
        self.phi = np.asarray(self.phi, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        assert np.allclose(self.P, self.P.T), "P tiene que ser simétrica"
        assert abs(self.x.sum() - 1) < 1e-12, "Σ x_a tiene que ser 1"

    @property
    def m(self):
        """Cantidad de clusters."""
        return len(self.x)

    def grados(self):
        """d_a para cada cluster."""
        return np.array(grados(self.x, self.P))

    def mu(self):
        """μ(T) = min_a d_a."""
        return self.grados().min()

    def fuerzas(self):
        """F_a para cada cluster (cero en equilibrio)."""
        return np.array(fuerzas(self.x, self.phi, self.P))

    def splitting(self):
        """c_a para cada cluster."""
        return np.array(splitting(self.x, self.phi, self.P))

    def matriz_clusters(self):
        """Matriz de L(y) como array."""
        return np.array(matriz_clusters(self.x, self.phi, self.P))

    def autovalores_clusters(self):
        """Autovalores de L en el ortogonal de 1 (sin el 0 de la rotación global)."""
        Q = base_perpendicular(self.m)
        return np.linalg.eigvalsh(Q.T @ self.matriz_clusters() @ Q)

    def reporte(self):
        """μ, max|F|, min c y el menor autovalor de L fuera de la rotación."""
        resultado = analizar(self.x, self.phi, self.P)
        resultado["min autovalor L"] = min(resultado.pop("autovalores L"))
        return resultado

    def es_estricta(self, tol=1e-12):
        """¿Equilibrio, c > 0 y L ≻ 0 fuera de la rotación, con tolerancia tol?"""
        r = self.reporte()
        return r["max|F|"] < tol and r["min c"] > tol and r["min autovalor L"] > tol


# ---------------------------------------------------------------- realización como grafo

def realizar(T, g, rng=None, tipo="aleatorio"):
    """Blow-up de una plantilla con clusters de igual tamaño en anillos Z_g (Fig. 4 de O'Keeffe).

    El vértice (a, i) se une con (b, j) si j − i ∈ S_ab, con |S_ab| = P_ab·g (tiene que ser entero),
    S_ba = −S_ab y S_aa simétrico sin el 0. Cada vértice de a tiene exactamente P_ab·g vecinos en b,
    así que las fases de la plantilla son un equilibrio exacto. Con tipo="intervalo",
    S_ab = {0, …, k−1}, como el k-spinning de Canale. Devuelve (W, theta).
    """
    assert np.allclose(T.x, 1 / T.m), "realizar solo soporta clusters de igual tamaño"
    rng = np.random.default_rng(rng)
    m = T.m
    W = np.zeros((m * g, m * g))
    for a in range(m):
        for b in range(a, m):
            k = T.P[a, b] * g
            assert abs(k - round(k)) < 1e-9, f"P[{a},{b}]·g = {k} no es entero"
            k = int(round(k))
            if a == b:
                k = min(k, g - 1)  # P_aa = 1: clique, g − 1 vecinos (sin self-loop)
                S = _conjunto_simetrico(g, k, rng, tipo)
            elif tipo == "intervalo":
                S = list(range(k))
            else:
                S = list(rng.choice(g, size=k, replace=False))
            for i in range(g):
                for s in S:
                    origen = a * g + i
                    destino = b * g + (i + s) % g
                    W[origen, destino] = 1
                    W[destino, origen] = 1
    theta = np.repeat(T.phi, g)
    return W, theta


def _conjunto_simetrico(g, k, rng, tipo):
    """Subconjunto simétrico de Z_g sin el 0, de tamaño k (pares {s, −s} más g/2 si k es impar)."""
    assert k <= g - 1
    pares = [(s, g - s) for s in range(1, (g + 1) // 2)]
    if k % 2 == 1:
        assert g % 2 == 0, "con g impar el tamaño de S_aa tiene que ser par"
    cantidad_pares = k // 2
    if tipo == "intervalo":
        elegidos = pares[:cantidad_pares]
    else:
        indices = rng.choice(len(pares), cantidad_pares, replace=False)
        elegidos = [pares[i] for i in indices]
    S = []
    for par in elegidos:
        S.extend(par)
    if k % 2 == 1:
        S.append(g // 2)
    return S


# ---------------------------------------------------------------- plantillas conocidas

def plantilla_c4():
    """Estado de 4 clusters de Kassabov–Strogatz–Townsend: μ = 3/4, L ≡ 0 (degenerado)."""
    P = np.array([
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
    ], dtype=float)
    return Plantilla(np.full(4, 0.25), np.arange(4) * np.pi / 2, P, "C4")


def plantilla_canale(a=1.0, b=1.0, c=1.0, d=0.5, interna=1.0):
    """Plantilla de Canale (arXiv 2209.06362, sec. V): 4 pares de clusters (i, ±) en πi/2 ± β/2.

    Los conjuntos E_t del paper tienen erratas; usamos la geometría de su Fig. 2 y del Lema 1:
      c: (i,+)–(i,−)       diferencia β          a: (i,+)–(i+1,−)   diferencia α
      b: (i,s)–(i+1,s)     diferencia π/2        d: (i,−)–(i+1,+)   diferencia α + 2β
    con α + β = π/2 y equilibrio (a − d) tan α = c. Estable si c² > 2ad.
    Con a = b = c = 1 y d → 1/2⁻, μ → 11/16. Índice del cluster (i, s): 2i + (s == +1).
    """
    def indice(i, s):
        """Índice del cluster (i, s) en la plantilla."""
        return 2 * (i % 4) + (s == 1)

    P = np.zeros((8, 8))
    for i in range(4):
        P[indice(i, 1), indice(i, -1)] = c
        P[indice(i, 1), indice(i + 1, -1)] = a
        P[indice(i, 1), indice(i + 1, 1)] = b
        P[indice(i, -1), indice(i + 1, -1)] = b
        P[indice(i, -1), indice(i + 1, 1)] = d
    P = P + P.T
    np.fill_diagonal(P, interna)
    alpha = np.arctan2(c, a - d)
    beta = np.pi / 2 - alpha
    phi = np.zeros(8)
    for k in range(8):
        signo = 1 if k % 2 else -1
        phi[k] = np.pi * (k // 2) / 2 + signo * beta / 2
    return Plantilla(np.full(8, 1 / 8), phi, P, f"Canale (a,b,c,d)=({a},{b},{c},{d})")
