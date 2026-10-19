"""Búsqueda de plantillas densas cerca del estado de 4 clusters C4.

Problema interno (convexo, O'Keeffe sec. III): con tamaños x y fases φ fijos,
    max t   s.t.  P simétrica, 0 ≤ P ≤ 1, P_aa = 1
                  d_a = Σ_b P_ab x_b ≥ t
                  F_a = Σ_b P_ab x_b sin(φ_a − φ_b) = 0
                  c_a = Σ_b P_ab x_b cos(φ_a − φ_b) ≥ eps_c
                  Qᵀ L(P) Q / scale ⪰ eps · I        (L ≻ 0 fuera de la rotación)
es un SDP. Cerca de C4 los modos de cluster son de orden h², así que se reescala por scale = h²
(si no, el margen mata los estados; ver O'Keeffe sec. VI).

Problema externo: las fases son φ = j π/2 + h s (j = cuarto de vuelta de cada pieza), y se
optimizan los tamaños x (softmax) y los desplazamientos s, con un método sin derivadas.
"""
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


def _perp_basis(m):
    Q, _ = np.linalg.qr(np.eye(m) - 1.0 / m)
    return Q[:, : m - 1]


def inner_sdp(x, phi, scale=1.0, eps=1e-6, eps_c=1e-3, forbid=None, solver="CLARABEL"):
    """Devuelve (μ, P) o (None, None) si es infactible. forbid: máscara de pares con P = 0."""
    m = len(x)
    D = phi[:, None] - phi[None, :]
    S, C = np.sin(D), np.cos(D)
    Wl = np.outer(x, x) * C
    np.fill_diagonal(Wl, 0)
    P = cp.Variable((m, m), symmetric=True)
    t = cp.Variable()
    K = cp.multiply(P, Wl)
    Lm = cp.diag(cp.sum(K, axis=1)) - K
    Q = _perp_basis(m)
    cons = [P >= 0, P <= 1, cp.diag(P) == 1,
            P @ x >= t,
            cp.multiply(P, S) @ x == 0,
            cp.multiply(P, C) @ x >= eps_c,
            (Q.T @ Lm @ Q) / scale >> eps * np.eye(m - 1)]
    if forbid is not None:
        cons.append(cp.multiply(P, forbid.astype(float)) == 0)
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        return None, None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None
    return float(t.value), np.array(P.value)


class SplitFamily:
    """Familia de plantillas: cada cuarto de vuelta j ∈ {0,1,2,3} se parte en pieces[j] piezas.

    Parámetros libres: logits de tamaños (uno por pieza) y desplazamientos s (uno por pieza;
    las piezas de un cuarto sin partir quedan en s = 0). φ = j π/2 + h s.
    """

    def __init__(self, pieces, h):
        self.pieces = list(pieces)
        self.h = h
        self.quarter = np.repeat(np.arange(4), self.pieces)
        self.m = len(self.quarter)
        self.split = np.array([self.pieces[j] > 1 for j in self.quarter])
        # pares de cuartos opuestos nunca se conectan en las plantillas tipo C4
        self.forbid = (np.abs(self.quarter[:, None] - self.quarter[None, :]) == 2)

    def n_params(self):
        return self.m + int(self.split.sum())

    def decode(self, z):
        logits, s_free = z[: self.m], z[self.m:]
        x = np.exp(logits - logits.max())
        x /= x.sum()
        s = np.zeros(self.m)
        s[self.split] = s_free
        phi = self.quarter * np.pi / 2 + self.h * s
        return x, phi

    def mu(self, z, eps=1e-6):
        x, phi = self.decode(z)
        if np.min(x) < 1e-3:
            return -1.0
        mu, _ = inner_sdp(x, phi, scale=self.h**2, eps=eps, forbid=self.forbid)
        return -1.0 if mu is None else mu


def optimize_family(fam, z0, eps=1e-6, maxiter=400, seed=None):
    rng = np.random.default_rng(seed)
    best = (fam.mu(z0, eps), z0)
    res = minimize(lambda z: -fam.mu(z, eps), z0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-7, "adaptive": True})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
    return best


def s7_like_start(fam, rng=None, noise=0.0, w=1.3982781340204612, q=0.6505691669447988):
    """Arranque "estilo S7" para cualquier patrón: tamaños por cuarto como en S7 (2a, 2a, 2a, 1 − 6a),
    repartidos en partes iguales entre las piezas, y desplazamientos equiespaciados en [−r, r], con
    r = w en el cuarto 1 y r = 1 en los demás. Con noise > 0 agrega ruido gaussiano (reinicios)."""
    rng = np.random.default_rng(rng)
    a = 1 / (7 + q)
    totals = np.array([2 * a, 2 * a, 2 * a, 1 - 6 * a])
    x = np.repeat(totals / np.array(fam.pieces), fam.pieces)
    s = []
    for j, k in enumerate(fam.pieces):
        if k > 1:
            r = w if j == 1 else 1.0
            s.extend(np.linspace(-r, r, k))
    z = np.concatenate([np.log(x), np.array(s)])
    return z + noise * rng.standard_normal(len(z))


def random_start(fam, rng):
    # tamaños parecidos a C4 repartidos entre las piezas + desplazamientos simétricos alrededor de 0
    logits = np.log(np.repeat(0.25 / np.array(fam.pieces), fam.pieces)) + 0.1 * rng.standard_normal(fam.m)
    s = []
    for j, k in enumerate(fam.pieces):
        if k > 1:
            s.extend(np.sort(rng.uniform(-1.5, 1.5, k)))
    return np.concatenate([logits, np.array(s)])
