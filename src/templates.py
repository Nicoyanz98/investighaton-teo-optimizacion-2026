"""Plantillas de clusters (x, φ, P) para estados de Kuramoto no sincronizados.

Notación de O'Keeffe (arXiv 2610.21271, sec. III):
  x_a    fracción de osciladores en el cluster a (Σ x_a = 1)
  φ_a    fase común del cluster a
  P_ab   densidad de links entre los clusters a y b (P_aa: dentro del cluster)

Condiciones (límite de red grande):
  d_a = Σ_b P_ab x_b                            grado / n;     μ(T) = min_a d_a
  F_a = Σ_b P_ab x_b sin(φ_a − φ_b) = 0         equilibrio
  c_a = Σ_b P_ab x_b cos(φ_a − φ_b) > 0         estabilidad contra splitting
  L(y) = Σ_{a<b} x_a x_b P_ab cos(φ_a − φ_b) (y_a − y_b)²  > 0 fuera de y ∝ 1   (modos de cluster)

Una plantilla es *estricta* si cumple las tres. El blow-up de una plantilla estricta da grafos con
un mínimo local estricto no sincronizado (Lema 2 de O'Keeffe; Prop. 2 de Canale para los k-spinnings).
"""
from dataclasses import dataclass

import mpmath as mp
import numpy as np


@dataclass
class Template:
    x: np.ndarray
    phi: np.ndarray
    P: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.phi = np.asarray(self.phi, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        assert np.allclose(self.P, self.P.T), "P tiene que ser simétrica"
        assert abs(self.x.sum() - 1) < 1e-12, "Σ x_a tiene que ser 1"

    @property
    def m(self):
        return len(self.x)

    def degrees(self):
        return self.P @ self.x

    def mu(self):
        return self.degrees().min()

    def forces(self):
        D = self.phi[:, None] - self.phi[None, :]
        return (self.P * np.sin(D)) @ self.x

    def splitting(self):
        D = self.phi[:, None] - self.phi[None, :]
        return (self.P * np.cos(D)) @ self.x

    def cluster_matrix(self):
        # matriz M con y·My = L(y): Laplaciano con pesos x_a x_b P_ab cos(φ_a − φ_b)
        K = np.outer(self.x, self.x) * self.P * np.cos(self.phi[:, None] - self.phi[None, :])
        np.fill_diagonal(K, 0)
        return np.diag(K.sum(axis=1)) - K

    def cluster_eigs(self):
        # autovalores de L restringida al ortogonal de 1 (sin el 0 de la rotación global)
        return np.linalg.eigvalsh(_perp_basis(self.m).T @ self.cluster_matrix() @ _perp_basis(self.m))

    def report(self):
        return {
            "mu": self.mu(),
            "max|F|": np.abs(self.forces()).max(),
            "min c": self.splitting().min(),
            "min eig L": self.cluster_eigs().min(),
        }

    def is_strict(self, tol=1e-12):
        r = self.report()
        return r["max|F|"] < tol and r["min c"] > tol and r["min eig L"] > tol


def _perp_basis(m):
    # base ortonormal del ortogonal de (1, …, 1)
    Q, _ = np.linalg.qr(np.eye(m) - 1.0 / m)
    return Q[:, : m - 1]


def report_mp(x, phi, P, dps=60):
    """Mismas cantidades que Template.report pero en precisión mpmath (x, φ, P como listas de mpf)."""
    with mp.workdps(dps):
        m = len(x)
        F = [mp.fsum(P[a][b] * x[b] * mp.sin(phi[a] - phi[b]) for b in range(m)) for a in range(m)]
        c = [mp.fsum(P[a][b] * x[b] * mp.cos(phi[a] - phi[b]) for b in range(m)) for a in range(m)]
        d = [mp.fsum(P[a][b] * x[b] for b in range(m)) for a in range(m)]
        M = mp.matrix(m, m)
        for a in range(m):
            for b in range(m):
                if a != b:
                    M[a, b] = -x[a] * x[b] * P[a][b] * mp.cos(phi[a] - phi[b])
            M[a, a] = -mp.fsum(M[a, b] for b in range(m) if b != a)
        # restringir a ⊥1 fijando y_0 = 0 (mismo signo de definición que en ⊥1)
        Mr = M[1:, 1:]
        eig = sorted(mp.eigsy(Mr)[0])
        return {"mu": min(d), "max|F|": max(abs(f) for f in F), "min c": min(c), "min eig L(y0=0)": eig[0]}


# ---------------------------------------------------------------- realización como grafo

def realize(T, g, rng=None, kind="random"):
    """Blow-up de una plantilla con clusters de igual tamaño (x_a = 1/m) en anillos Z_g.

    El vértice (a, i) se une con (b, j) si j − i ∈ S_ab, con |S_ab| = P_ab·g (tiene que ser entero).
    S_ba = −S_ab y S_aa es simétrico sin el 0, así que cada vértice de a tiene exactamente P_ab·g
    vecinos en b y las fases de la plantilla son un equilibrio exacto (Fig. 4 de O'Keeffe).
    kind = "interval" usa S_ab = {0, …, k−1} como el k-spinning de Canale.
    Devuelve (W, theta).
    """
    assert np.allclose(T.x, 1 / T.m), "realize solo soporta clusters de igual tamaño"
    rng = np.random.default_rng(rng)
    m = T.m
    W = np.zeros((m * g, m * g))
    idx = lambda a, i: a * g + (i % g)
    for a in range(m):
        for b in range(a, m):
            k = T.P[a, b] * g
            assert abs(k - round(k)) < 1e-9, f"P[{a},{b}]·g = {k} no es entero"
            k = int(round(k))
            if a == b:
                k = min(k, g - 1)  # P_aa = 1: clique, g − 1 vecinos (sin self-loop)
                S = _symmetric_set(g, k, rng, kind)
            elif kind == "interval":
                S = list(range(k))
            else:
                S = list(rng.choice(g, size=k, replace=False))
            for i in range(g):
                for s in S:
                    W[idx(a, i), idx(b, i + s)] = 1
                    W[idx(b, i + s), idx(a, i)] = 1
    theta = np.repeat(T.phi, g)
    return W, theta


def _symmetric_set(g, k, rng, kind):
    # subconjunto simétrico de Z_g \ {0} de tamaño k
    assert k <= g - 1
    pairs = [(s, g - s) for s in range(1, (g + 1) // 2)]
    half = [g // 2] if g % 2 == 0 else []
    if k % 2 == 1:
        assert half, "con g impar el tamaño de S_aa tiene que ser par"
    npairs = k // 2
    chosen = pairs[:npairs] if kind == "interval" else [pairs[i] for i in rng.choice(len(pairs), npairs, replace=False)]
    S = [s for p in chosen for s in p]
    if k % 2 == 1:
        S += half
    return S


# ---------------------------------------------------------------- plantillas conocidas

def c4_template():
    # estado de 4 clusters de Kassabov–Strogatz–Townsend: μ = 3/4, L ≡ 0
    P = np.array([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]], dtype=float)
    return Template(np.full(4, 0.25), np.arange(4) * np.pi / 2, P, "C4")


def canale_template(a=1.0, b=1.0, c=1.0, d=0.5, internal=1.0):
    """Plantilla de Canale (arXiv 2209.06362, sec. V): G_p con 8 vértices y pesos (a, b, c, d).

    Hay 4 pares de clusters, uno por cuarto de vuelta: el cluster (i, s), i ∈ Z_4, s = ±1, está en
    la fase πi/2 + sβ/2. Los conjuntos de índices E_t del paper tienen erratas; usamos la geometría
    de la Fig. 2 y del Hessiano del Lema 1, que fija la diferencia de fase de cada tipo de arista:
      c: (i,+)–(i,−)                  diferencia β        (dentro del par)
      a: (i,+)–(i+1,−)                diferencia α        (miembros cercanos de pares vecinos)
      b: (i,s)–(i+1,s)                diferencia π/2      (se compensan entre sí)
      d: (i,−)–(i+1,+)                diferencia α + 2β   (miembros lejanos)
    con α + β = π/2 y equilibrio (a − d) tan α = c. Estable si c² > 2ad (Lema 1).
    Con a = b = c = 1 y d → 1/2⁻, μ → (a + 2b + c + d + 1)/8 = 11/16.
    Índice del cluster (i, s): 2i + (s == +1).
    """
    v = lambda i, s: 2 * (i % 4) + (s == 1)
    P = np.zeros((8, 8))
    for i in range(4):
        P[v(i, 1), v(i, -1)] = c
        P[v(i, 1), v(i + 1, -1)] = a
        P[v(i, 1), v(i + 1, 1)] = b
        P[v(i, -1), v(i + 1, -1)] = b
        P[v(i, -1), v(i + 1, 1)] = d
    P = P + P.T
    np.fill_diagonal(P, internal)
    alpha = np.arctan2(c, a - d)
    beta = np.pi / 2 - alpha
    phi = np.array([np.pi * (k // 2) / 2 + (beta / 2 if k % 2 else -beta / 2) for k in range(8)])
    return Template(np.full(8, 1 / 8), phi, P, f"Canale (a,b,c,d)=({a},{b},{c},{d})")
