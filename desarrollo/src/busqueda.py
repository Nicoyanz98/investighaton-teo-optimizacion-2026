"""Búsqueda de plantillas densas: SDP en las densidades y familias de particiones de una base.

Problema interno (convexo, O'Keeffe sec. III): con tamaños x y fases φ fijos,
    max t   s.t.  P simétrica, 0 ≤ P ≤ 1, P_aa = 1, P_ab = 0 en los pares prohibidos
                  d_a ≥ t,   F_a = 0,   c_a ≥ margen_c,   Qᵀ L(P) Q / escala ⪰ margen · I
es un SDP. Cerca de un estado degenerado (como C4) los modos de cluster son de orden h², así que se
reescala por escala = h²; si no, el margen mata los estados (O'Keeffe sec. VI).

Problema externo: una *base* es un estado de sectores (fases y tamaños), y un *patrón* dice en
cuántas piezas se parte cada sector. Las piezas del sector j quedan en φ = fase_j + h·s y se
optimizan sus tamaños y desplazamientos s con un método sin derivadas.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from .aurea import PHI, puntos_aureos
from .plantillas import Plantilla, base_perpendicular, plantilla_c4, plantilla_canale


# ---------------------------------------------------------------- problema interno

def sdp_interno(x, phi, escala=1.0, margen=1e-6, margen_c=1e-3, prohibidos=None, solver="CLARABEL"):
    """Mayor μ sobre las densidades P con x y φ fijos. Devuelve (μ, P) o (None, None) si es infactible."""
    m = len(x)
    diferencias = phi[:, None] - phi[None, :]
    senos = np.sin(diferencias)
    cosenos = np.cos(diferencias)
    pesos_L = np.outer(x, x) * cosenos
    np.fill_diagonal(pesos_L, 0)

    P = cp.Variable((m, m), symmetric=True)
    t = cp.Variable()
    K = cp.multiply(P, pesos_L)
    L = cp.diag(cp.sum(K, axis=1)) - K
    Q = base_perpendicular(m)
    restricciones = [
        P >= 0,
        P <= 1,
        cp.diag(P) == 1,
        P @ x >= t,
        cp.multiply(P, senos) @ x == 0,
        cp.multiply(P, cosenos) @ x >= margen_c,
        (Q.T @ L @ Q) / escala >> margen * np.eye(m - 1),
    ]
    if prohibidos is not None:
        restricciones.append(cp.multiply(P, prohibidos.astype(float)) == 0)
    problema = cp.Problem(cp.Maximize(t), restricciones)
    try:
        problema.solve(solver=solver)
    except cp.error.SolverError:
        return None, None
    if problema.status not in ("optimal", "optimal_inaccurate"):
        return None, None
    return float(t.value), np.array(P.value)


# ---------------------------------------------------------------- bases

@dataclass
class Base:
    """Estado de sectores sobre el que se parten clusters.

    fases: fase de cada sector. tamanos: tamaño total de cada sector (para los arranques).
    prohibidos: pares de sectores sin links. radios: amplitud inicial de los desplazamientos por sector.
    """

    nombre: str
    fases: np.ndarray
    tamanos: np.ndarray
    prohibidos: np.ndarray
    radios: np.ndarray


def base_c4(q=0.6505691669447988, w=1.3982781340204612):
    """C4 con los tamaños de S7 (2a, 2a, 2a, 1 − 6a) y radios (1, w*, 1, 1); opuestos sin links."""
    a = 1 / (7 + q)
    T = plantilla_c4()
    return Base(
        nombre="C4",
        fases=T.phi,
        tamanos=np.array([2 * a, 2 * a, 2 * a, 1 - 6 * a]),
        prohibidos=(T.P == 0),
        radios=np.array([1.0, w, 1.0, 1.0]),
    )


def base_canale():
    """Canale en el borde d = 1/2 (8 sectores iguales); sin links donde Canale no los tiene."""
    T = plantilla_canale(d=0.5)
    return Base(
        nombre="Canale",
        fases=T.phi,
        tamanos=T.x,
        prohibidos=(T.P == 0),
        radios=np.ones(8),
    )


# ---------------------------------------------------------------- familias de particiones

class FamiliaParticiones:
    """Plantillas que parten cada sector j de una base en piezas[j] piezas, separadas por h·s."""

    def __init__(self, base, piezas, h):
        """Arma la familia: a qué sector pertenece cada pieza y qué pares quedan prohibidos."""
        assert len(piezas) == len(base.fases), "un número de piezas por sector"
        self.base = base
        self.piezas = list(piezas)
        self.h = h
        self.sector = np.repeat(np.arange(len(piezas)), self.piezas)
        self.m = len(self.sector)
        self.partida = np.array([self.piezas[j] > 1 for j in self.sector])
        self.prohibidos = base.prohibidos[np.ix_(self.sector, self.sector)]

    def cantidad_parametros(self):
        """Largo de z: un logit por pieza más un desplazamiento por pieza partida."""
        return self.m + int(self.partida.sum())

    def decodificar(self, z):
        """z = (logits de tamaños, desplazamientos de las piezas partidas) → (x, φ)."""
        logits = z[: self.m]
        desplazamientos_libres = z[self.m:]
        x = np.exp(logits - logits.max())
        x = x / x.sum()
        desplazamientos = np.zeros(self.m)
        desplazamientos[self.partida] = desplazamientos_libres
        phi = self.base.fases[self.sector] + self.h * desplazamientos
        return x, phi

    def desplazamientos(self, phi):
        """Inversa parcial de decodificar: s = (φ − fase del sector)/h."""
        return (phi - self.base.fases[self.sector]) / self.h

    def mu(self, z, margen=1e-6):
        """μ del SDP interno para los parámetros z (−1 si es infactible o hay piezas vacías)."""
        x, phi = self.decodificar(z)
        if np.min(x) < 1e-3:
            return -1.0
        mu, _ = sdp_interno(x, phi, escala=self.h**2, margen=margen, prohibidos=self.prohibidos)
        return -1.0 if mu is None else mu


def arranque_estructurado(familia, rng=None, ruido=0.0):
    """Tamaños de la base repartidos en partes iguales y desplazamientos equiespaciados en [−r_j, r_j].

    Con la base C4 y el patrón (2,2,2,1) es el ansatz de S7. Con ruido > 0 sirve para reinicios.
    """
    rng = np.random.default_rng(rng)
    tamanos = np.repeat(familia.base.tamanos / np.array(familia.piezas), familia.piezas)
    desplazamientos = []
    for j, k in enumerate(familia.piezas):
        if k > 1:
            r = familia.base.radios[j]
            desplazamientos.extend(np.linspace(-r, r, k))
    z = np.concatenate([np.log(tamanos), np.array(desplazamientos)])
    return z + ruido * rng.standard_normal(len(z))


def arranque_aleatorio(familia, rng):
    """Tamaños casi iguales por sector y desplazamientos al azar en [−1.5, 1.5]."""
    sectores = len(familia.piezas)
    tamanos = np.repeat(1 / sectores / np.array(familia.piezas), familia.piezas)
    logits = np.log(tamanos) + 0.1 * rng.standard_normal(familia.m)
    desplazamientos = []
    for k in familia.piezas:
        if k > 1:
            desplazamientos.extend(np.sort(rng.uniform(-1.5, 1.5, k)))
    return np.concatenate([logits, np.array(desplazamientos)])


def optimizar_familia(familia, z0, margen=1e-6, max_iteraciones=400):
    """Nelder–Mead sobre z maximizando μ desde z0. Devuelve (mejor μ, mejor z)."""
    mejor = (familia.mu(z0, margen), z0)
    opciones = {"maxiter": max_iteraciones, "xatol": 1e-4, "fatol": 1e-7, "adaptive": True}
    resultado = minimize(lambda z: -familia.mu(z, margen), z0, method="Nelder-Mead", options=opciones)
    if -resultado.fun > mejor[0]:
        mejor = (-resultado.fun, resultado.x)
    return mejor


# ---------------------------------------------------------------- semillas (vía E)

def semilla(x, phi):
    """μ de borde (L ⪰ 0, sin brecha) y cantidad de modos de cluster neutros en el óptimo."""
    mu, P = sdp_interno(x, phi, margen=0.0, margen_c=1e-6)
    if mu is None:
        return None, None
    T = Plantilla(x, phi, (P + P.T) / 2)
    neutros = int((T.autovalores_clusters() < 1e-7).sum())
    return mu, neutros


def una_red(l):
    """ℓ clusters iguales equiespaciados."""
    return np.full(l, 1 / l), 2 * np.pi * np.arange(l) / l


def dos_redes(l, r, beta):
    """Dos redes de ℓ puntos rotadas β entre sí, con pesos totales r y 1 − r."""
    red = 2 * np.pi * np.arange(l) / l
    x = np.concatenate([np.full(l, r / l), np.full(l, (1 - r) / l)])
    phi = np.concatenate([red, red + beta])
    return x, phi


def red_aurea(m, pesos="iguales"):
    """m clusters en las fases áureas 2π a/φ (variante 3 de la familia girasol).

    pesos: "iguales" (1/m), "arcos" (cada cluster pesa el arco del círculo más cercano a él que a los
    demás) o "geometricos" (proporcionales a φ^{−a}).
    """
    x_posiciones = puntos_aureos(m)
    phi = 2 * np.pi * x_posiciones
    if pesos == "iguales":
        x = np.full(m, 1 / m)
    elif pesos == "arcos":
        orden = np.argsort(x_posiciones)
        ordenadas = x_posiciones[orden]
        huecos = np.diff(np.concatenate([ordenadas, [ordenadas[0] + 1]]))
        arcos_ordenados = (huecos + np.roll(huecos, 1)) / 2
        x = np.zeros(m)
        x[orden] = arcos_ordenados
    elif pesos == "geometricos":
        x = PHI ** -np.arange(m, dtype=float)
        x = x / x.sum()
    else:
        raise ValueError(f"pesos desconocidos: {pesos}")
    return x, phi
